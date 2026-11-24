"""Benchmark nuxnet-inference on SELMA3D annotated nuclei.

Two questions, deliberately measured separately because they can disagree:

* **Semantic** -- does the predicted mask overlap the annotated mask? (Dice, IoU)
* **Detection** -- does the predicted *object set* match the annotated object set,
  by centroid? (precision, recall, F1 at a stated tolerance)

Both are always reported. Reporting only Dice would be misleading here, and
reporting only detection would overstate the mask.

Why Dice alone is not enough, and what the "ceiling" row means
--------------------------------------------------------------
nuxnet was trained on NuMorph ground truth, which marks a **single-slice disc at
each nucleus middle**. SELMA3D's nucleus annotations are **filled 3D bodies**
drawn in virtual reality by four experts. A disc can never overlap a body well,
even when the detection is perfect -- so raw Dice conflates "wrong place" with
"different label convention".

To separate those, this reports a **disc ceiling**: synthesise the best possible
disc-shaped prediction from the *ground-truth* centroids, then score that against
the ground truth. It is an upper bound for any predictor that outputs centre
discs, and it is the number raw Dice must be read against. It uses the ground
truth as its own input, so it is a diagnostic, not a result.

Predictions are cached to an ``.npz`` so a re-scored benchmark does not re-run
the network. Deleting the cache is always safe.

Usage
-----
    python benchmark_selma3d.py --data selma3d --model fold1.pt \\
        --spacing 2.3,1.0,1.0 --out benchmark.json

``--data`` is a directory holding ``raw_*.mha`` / ``gt_*.mha`` (2026 layout) or
``*_0000.nii.gz`` / ``*.nii.gz`` (2024 layout). Pairs are matched by sorted order
and the pairing is printed so a mis-pairing is visible rather than silent.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def load_nifti(path: Path) -> np.ndarray:
    """NIfTI-1 -> (z, y, x). i is stored fastest, so a plain reshape suffices."""
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    dims = struct.unpack("<8h", raw[40:56])
    datatype = struct.unpack("<h", raw[70:72])[0]
    offset = int(struct.unpack("<f", raw[108:112])[0])
    dtype = {64: np.float64, 16: np.float32, 4: np.int16, 2: np.uint8,
             512: np.uint16, 256: np.int8}[datatype]
    shape = tuple(dims[1:dims[0] + 1])
    flat = np.frombuffer(raw[offset:], dtype=dtype, count=int(np.prod(shape)))
    return flat.reshape(shape)


def load_mha(path: Path) -> np.ndarray:
    """MetaImage -> (z, y, x).

    MHA headers are plain text terminated by ``ElementDataFile = LOCAL``. The
    payload is zlib-compressed when ``CompressedData`` is True -- zlib, not gzip.
    Feeding a zlib stream to ``gzip.decompress`` raises BadGzipFile, so the magic
    bytes decide. A truncated download fails here rather than producing garbage,
    which is the behaviour we want: compare ``CompressedDataSize`` against the
    payload length if this raises.
    """
    raw = path.read_bytes()
    marker = b"ElementDataFile = LOCAL"
    start = raw.index(marker) + len(marker)
    start = raw.index(b"\n", start) + 1
    fields: dict[str, str] = {}
    for line in raw[:start].decode("latin-1").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()

    dims = [int(v) for v in fields["DimSize"].split()]
    dtype = {"MET_USHORT": np.uint16, "MET_UCHAR": np.uint8,
             "MET_SHORT": np.int16, "MET_FLOAT": np.float32}[fields["ElementType"]]
    payload = raw[start:]
    if fields.get("CompressedData", "False") == "True":
        payload = (gzip.decompress(payload) if payload[:2] == b"\x1f\x8b"
                   else zlib.decompress(payload))
    flat = np.frombuffer(payload, dtype=dtype, count=int(np.prod(dims)))
    # x fastest -> C-order reshape into reversed dims gives (z, y, x) with no
    # transpose. Using order="F" here silently transposes a cubic patch.
    return flat.reshape(tuple(dims[::-1]))


def load_any(path: Path) -> np.ndarray:
    return load_mha(path) if path.suffix == ".mha" else load_nifti(path)


def discover(data: Path) -> list[tuple[Path, Path]]:
    """Pair raw volumes with their ground truth, and refuse ambiguous sets."""
    raws = sorted(p for p in data.glob("raw_*.mha"))
    gts = sorted(p for p in data.glob("gt_*.mha"))
    if not raws:
        raws = sorted(p for p in data.glob("*_0000.nii.gz"))
        gts = sorted(p for p in data.glob("*.nii.gz")
                     if not p.name.endswith("_0000.nii.gz"))
    if not raws or len(raws) != len(gts):
        raise SystemExit(
            f"{data}: found {len(raws)} raw and {len(gts)} ground-truth files. "
            "Matching is by sorted order, so a mismatch cannot be resolved safely."
        )
    return list(zip(raws, gts))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    total = int(a.sum()) + int(b.sum())
    return float(2 * int((a & b).sum()) / total) if total else 1.0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = int((a | b).sum())
    return float(int((a & b).sum()) / union) if union else 1.0


def centroids_vox(mask: np.ndarray) -> np.ndarray:
    """Connected-component centroids, in voxels, z-y-x order."""
    from scipy import ndimage as ndi

    labelled, count = ndi.label(mask > 0, structure=np.ones((3, 3, 3)))
    if not count:
        return np.empty((0, 3))
    # Drop single-voxel specks: they dominate a body annotation's component list
    # and would inflate the apparent object count on both sides.
    sizes = np.bincount(labelled.ravel())
    keep = [i for i in range(1, count + 1) if sizes[i] > 1]
    if not keep:
        return np.empty((0, 3))
    return np.array(ndi.center_of_mass(mask > 0, labelled, keep))


def match(truth_um: np.ndarray, pred_um: np.ndarray, tolerance: float) -> dict:
    """Greedy one-to-one centroid matching.

    Greedy, not optimal, because it is the same rule the project's other
    detection metrics use -- a different assignment strategy here would make
    these numbers silently incomparable with the NuMorph and ear tables.
    """
    from scipy.spatial import cKDTree

    if len(truth_um) == 0 or len(pred_um) == 0:
        return {"matched": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "median_um": None}

    fwd_d, fwd_i = cKDTree(pred_um).query(truth_um, k=1)
    bwd_d, _ = cKDTree(truth_um).query(pred_um, k=1)

    used: set[int] = set()
    distances: list[float] = []
    for i in np.argsort(fwd_d):
        if fwd_d[i] > tolerance:
            break
        j = int(fwd_i[i])
        if j in used:
            continue
        used.add(j)
        distances.append(float(fwd_d[i]))

    precision = float((bwd_d <= tolerance).mean())
    recall = float(len(distances) / len(truth_um))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"matched": len(distances), "precision": precision, "recall": recall,
            "f1": float(f1),
            "median_um": float(np.median(distances)) if distances else None}


def disc_ceiling(truth: np.ndarray, spacing: np.ndarray, radius_vox: float) -> float:
    """Dice of an ideal disc-shaped prediction built from the ground truth.

    This quantifies how much of the Dice deficit is the label *convention* rather
    than a localisation error. It consumes the ground truth, so it is a ceiling,
    never a result.
    """
    from scipy import ndimage as ndi

    centres = centroids_vox(truth)
    if len(centres) == 0:
        return 0.0
    ideal = np.zeros(truth.shape, dtype=bool)
    zz, yy, xx = np.ogrid[:truth.shape[0], :truth.shape[1], :truth.shape[2]]
    for cz, cy, cx in centres:
        ideal |= (np.sqrt(((zz - cz) / max(radius_vox / spacing[0] * spacing[1], 1e-6)) ** 2
                          + (yy - cy) ** 2 + (xx - cx) ** 2) <= radius_vox)
    return dice(truth, ideal)


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------
def cached_predict(cache: Path, key: str, compute):
    """Reuse a prediction if the same key was scored before.

    The network pass dominates the runtime, and a benchmark is re-read many times
    while being written up. Keying on the file stem plus spacing means a change of
    spacing correctly invalidates the cache instead of silently reusing a result
    computed under a different assumption.
    """
    store = {}
    if cache.is_file():
        try:
            store = dict(np.load(cache, allow_pickle=False))
        except Exception:
            store = {}
    if key in store:
        return store[key], True
    value = compute()
    store[key] = value.astype(np.uint8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **store)
    return store[key], False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dir with raw_*/gt_* pairs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--spacing", default="2.3,1.0,1.0",
                        help="patch voxel size in um as z,y,x. 2.3 is the median "
                             "implied by annotation geometry (see "
                             "selma3d_spacing.py); 1,1,1 is what the headers "
                             "declare and what the model scores best at.")
    parser.add_argument("--target-spacing", default="3.0,1.0,1.0")
    parser.add_argument("--patch", default="32,128,128")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--tolerance-um", type=float, default=2.2,
                        help="centroid match radius; one nucleus radius at this "
                             "sampling")
    parser.add_argument("--cache", default=None, help="npz of cached predictions")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--ceiling", action="store_true",
                        help="also compute the disc-shape ceiling (slow, and it "
                             "reads the ground truth, so it is diagnostic only)")
    args = parser.parse_args()

    spacing = np.array([float(v) for v in args.spacing.split(",")])
    target = tuple(float(v) for v in args.target_spacing.split(","))
    patch = tuple(int(v) for v in args.patch.split(","))

    pairs = discover(Path(args.data))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} pairs:")
    for raw, gt in pairs:
        print(f"  {raw.name:28s} <-> {gt.name}")

    import torch

    from nuxnet_inference_package.cli_pred import predict_volume_on_training_grid
    from nuxnet_inference_package.models.unet3d import UNet3D

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, classes=2)
    state = torch.load(args.model, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    cache = Path(args.cache) if args.cache else Path(args.out).with_suffix(".npz")

    volumes = []
    for index, (raw_path, gt_path) in enumerate(pairs, start=1):
        key = f"{raw_path.stem}__" + "_".join(f"{v:g}" for v in spacing)
        volume = load_any(raw_path).astype(np.float32)
        truth = load_any(gt_path) > 0

        def compute():
            mask, _ = predict_volume_on_training_grid(
                volume, model=model, device=device,
                source_voxel_size_um=spacing, target_voxel_size_um=target,
                patch_size=patch, overlap=args.overlap, normalize_input=True,
                classes=2, threshold=args.threshold,
            )
            return (mask > 0)

        pred, reused = cached_predict(cache, key, compute)
        pred = pred.astype(bool)
        print(f"[{index}/{len(pairs)}] {raw_path.name} "
              f"({'cached' if reused else 'computed'})", flush=True)

        truth_c = centroids_vox(truth) * spacing
        pred_c = centroids_vox(pred) * spacing
        det = match(truth_c, pred_c, args.tolerance_um)

        record = {
            "raw": raw_path.name,
            "gt": gt_path.name,
            "shape_zyx": list(truth.shape),
            "dice": dice(truth, pred),
            "iou": iou(truth, pred),
            "truth_fg_fraction": float(truth.mean()),
            "pred_fg_fraction": float(pred.mean()),
            "truth_objects": int(len(truth_c)),
            "pred_objects": int(len(pred_c)),
            "detection": det,
        }
        if args.ceiling:
            record["disc_ceiling_dice"] = disc_ceiling(truth, spacing,
                                                       radius_vox=2.0)
        volumes.append(record)

        c = record.get("disc_ceiling_dice")
        print(f"      dice {record['dice']:.3f}  iou {record['iou']:.3f}"
              + (f"  ceiling {c:.3f}" if c is not None else "")
              + f"  P {det['precision']:.3f}  R {det['recall']:.3f} "
                f"F1 {det['f1']:.3f}")

    mean = {
        "dice": float(np.mean([v["dice"] for v in volumes])),
        "iou": float(np.mean([v["iou"] for v in volumes])),
        "precision": float(np.mean([v["detection"]["precision"] for v in volumes])),
        "recall": float(np.mean([v["detection"]["recall"] for v in volumes])),
        "f1": float(np.mean([v["detection"]["f1"] for v in volumes])),
        "truth_fg_fraction": float(np.mean([v["truth_fg_fraction"] for v in volumes])),
        "pred_fg_fraction": float(np.mean([v["pred_fg_fraction"] for v in volumes])),
        "truth_objects": int(sum(v["truth_objects"] for v in volumes)),
        "pred_objects": int(sum(v["pred_objects"] for v in volumes)),
    }
    with_ceiling = [v["disc_ceiling_dice"] for v in volumes if "disc_ceiling_dice" in v]
    if with_ceiling:
        mean["disc_ceiling_dice"] = float(np.mean(with_ceiling))

    summary = {
        "dataset": "SELMA3D annotated nucleus patches",
        "spacing_um": [float(v) for v in spacing],
        "target_spacing_um": list(target),
        "patch_zyx": list(patch),
        "overlap": args.overlap,
        "threshold": args.threshold,
        "tolerance_um": args.tolerance_um,
        "n_patches": len(volumes),
        "mean": mean,
        "volumes": volumes,
        "notes": [
            "SELMA3D nucleus labels are filled 3D bodies; nuxnet's training target "
            "is a single-slice disc at each nucleus middle. Raw Dice conflates "
            "localisation error with that convention difference -- read it against "
            "disc_ceiling_dice when present.",
            "Centroid detection is the metric the two label conventions can share.",
            "Spacing is not given reliably by the files: every header declares "
            "1,1,1 while the challenge paper gives 0.54,0.54,5.0 for the source "
            "volumes. See selma3d_spacing.py.",
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== mean over {} patches ===".format(len(volumes)))
    print(f"  semantic  dice {mean['dice']:.3f}  iou {mean['iou']:.3f}"
          + (f"   (disc ceiling {mean['disc_ceiling_dice']:.3f})"
             if "disc_ceiling_dice" in mean else ""))
    print(f"  detection P {mean['precision']:.3f}  R {mean['recall']:.3f}  "
          f"F1 {mean['f1']:.3f}   @ {args.tolerance_um:g} um")
    print(f"  objects   truth {mean['truth_objects']:,}  pred {mean['pred_objects']:,}")
    print(f"  foreground truth {mean['truth_fg_fraction']*100:.3f}%  "
          f"pred {mean['pred_fg_fraction']*100:.3f}%")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
