"""Score *detections*, not just overlap, and save the instances for tabulation.

Why this is a separate axis from Dice
------------------------------------
Dice on the mask asks "how well do the voxels agree". A user asking the tool for
nuclei asks a different question: "did you find the nucleus, and where is it?" On
a marker-style target those two can disagree sharply in both directions -- a
model can return a good detection with sloppy edges, or crisp edges on the wrong
objects -- and only detection metrics expose that.

The instances come from the *unchanged* post-processing in the CLI
(``extract_nuclei_instances``), so what is scored here is exactly what a user
gets in the ``.nuclei.tsv``. Nothing is reimplemented.

How matching works
------------------
Predicted instance centroids and ground-truth instance centroids are matched
greedily by physical distance under a tolerance. Greedy nearest-neighbour, not
optimal assignment, because it is the stricter behaviour: it cannot reshuffle
pairs to raise its own score, so a reported precision cannot be an artefact of a
lucky assignment.

The tolerance is reported in microns and chosen from the object size, not as a
convenient round number: a nucleus here is ~14.8 um across, so a match radius of
half that is the scale at which "same nucleus" is unambiguous. Distances are
computed in microns using the volume's own voxel size, so the anisotropy
(3.0 um in Z against 1.0 in XY) does not distort the comparison.

    python scripts/detect_metrics.py --artifacts /out --out /out/detection.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("PYTHONPATH", "/work"))

from nuxnet_inference_package.cli_pred import extract_nuclei_instances

# Same target grid the CLI resamples to; centroids are reported on this grid.
TARGET_UM = np.array([3.0, 1.0, 1.0])


def instances_of(mask: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Centroids (z, y, x) of instances, via the CLI's own post-processing."""
    found = extract_nuclei_instances(prediction=mask.astype(np.uint8),
                                     nuclei_label=1, radius=radius)
    if not found:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array([[i["z"], i["y"], i["x"]] for i in found], dtype=np.float64)


def match(found: np.ndarray, truth: np.ndarray, voxel_um: np.ndarray,
          tolerance_um: float) -> dict:
    """Greedy nearest-neighbour matching in physical microns."""
    if len(found) == 0 or len(truth) == 0:
        return {
            "matched": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "median_um": None, "p90_um": None, "distances": [],
        }

    from scipy.spatial import cKDTree

    # Scale BEFORE querying: matching in voxels would treat a 3 um Z step as equal
    # to a 1 um XY step and quietly bias every distance.
    found_um = found * voxel_um
    truth_um = truth * voxel_um
    tree = cKDTree(truth_um)

    used = np.zeros(len(truth_um), dtype=bool)
    distances = []
    for point in found_um:
        nearby = tree.query_ball_point(point, tolerance_um)
        free = [i for i in nearby if not used[i]]
        if not free:
            continue
        best = min(free, key=lambda i: float(np.linalg.norm(truth_um[i] - point)))
        used[best] = True
        distances.append(float(np.linalg.norm(truth_um[best] - point)))

    matched = len(distances)
    precision = matched / max(len(found_um), 1)
    recall = matched / max(len(truth_um), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    distances = np.array(distances)
    return {
        "matched": matched,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "median_um": float(np.median(distances)) if distances.size else None,
        "p90_um": float(np.percentile(distances, 90)) if distances.size else None,
        "distances": distances.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True,
                        help="directory written by eval_numorph.py")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tolerance-um", type=float, default=None,
                        help="match radius; default is half a nucleus (7.4 um)")
    parser.add_argument("--radius", type=float, default=2.0,
                        help="neighbour radius for instance clustering, as in the CLI")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="probability threshold; must match the reported Dice operating point")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    summary = json.loads((artifacts / "summary.json").read_text())
    tolerance = args.tolerance_um if args.tolerance_um is not None else 7.4

    records = []
    for result in summary:
        stem = result["volume"].replace(".ome.tif", "")
        truth_path = artifacts / f"{stem}.truth_grid.npy"
        pred_path = artifacts / f"{stem}.pred_input.npy"
        if not truth_path.is_file():
            continue

        # Score on the *training grid*, where both masks are defined and where the
        # projection has not yet smeared anything. The input-grid mask is projected
        # from this one, so measuring here isolates the network from the export.
        truth = np.load(truth_path) > 0
        prob = np.load(artifacts / f"{stem}.prob.npy").astype(np.float32)
        pred = prob >= args.threshold
        if pred.shape != truth.shape:
            print(f"skip {stem}: pred {pred.shape} != truth {truth.shape}")
            continue

        truth_inst = instances_of(truth)
        pred_inst = instances_of(pred, radius=args.radius)
        matched = match(pred_inst, truth_inst, TARGET_UM, tolerance)

        record = {
            "volume": result["volume"],
            "group": result["group"],
            "shape": list(pred.shape),
            "n_truth": int(len(truth_inst)),
            "n_pred": int(len(pred_inst)),
            "matched": matched["matched"],
            "precision": round(matched["precision"], 4),
            "recall": round(matched["recall"], 4),
            "f1": round(matched["f1"], 4),
            "median_um": matched["median_um"],
            "p90_um": matched["p90_um"],
            "tolerance_um": tolerance,
            "dice_train": round(
                float(2.0 * np.logical_and(pred, truth).sum()
                      / max(pred.sum() + truth.sum(), 1)), 4
            ),
            # Kept for the distance histogram; trimmed to a manageable size.
            "distances": matched["distances"][:5000],
        }
        # A handful of real centroids, so the table in the PDF is inspectable and
        # a reader can see what a row of the .nuclei.tsv actually looks like.
        #
        # Stratified across z rather than the first few: components are ordered by
        # their lowest voxel index, so "the first 12" are all from the top of the
        # stack and, because the target is a single-slice disc, every one of them
        # has centroid z = 0.0 exactly. That reads as a bug to anyone who has not
        # internalised the marker convention, and it hides what the rows look like
        # elsewhere in the volume.
        if len(pred_inst):
            order = np.argsort(pred_inst[:, 0])
            picks = np.linspace(0, len(order) - 1, min(10, len(order))).astype(int)
            sample = pred_inst[order[picks]]
            record["example_centroids"] = [
                {"z": round(float(z), 2), "y": round(float(y), 2), "x": round(float(x), 2)}
                for z, y, x in sample
            ]
        else:
            record["example_centroids"] = []
        records.append(record)
        print(json.dumps({k: v for k, v in record.items()
                          if k not in ("distances", "example_centroids")}))

    out = Path(args.out) if args.out else artifacts / "detection.json"
    out.write_text(json.dumps(records, indent=2))

    if records:
        p = np.array([r["precision"] for r in records])
        r = np.array([r["recall"] for r in records])
        f = np.array([r["f1"] for r in records])
        d = np.array([r["median_um"] for r in records if r["median_um"] is not None])
        print(f"\nmatched volumes: {len(records)}")
        print(f"precision  mean {p.mean():.3f}  min {p.min():.3f}  max {p.max():.3f}")
        print(f"recall     mean {r.mean():.3f}  min {r.min():.3f}  max {r.max():.3f}")
        print(f"F1         mean {f.mean():.3f}  min {f.min():.3f}  max {f.max():.3f}")
        if d.size:
            print(f"median match distance  {np.median(d):.2f} um")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
