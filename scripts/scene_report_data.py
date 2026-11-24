"""Quantitative checks for a real-data scene the model has never seen.

NuMorph volumes have ground-truth masks, so their report pages can compute Dice.
A whole-mount tissue scene has no ground truth, and pretending otherwise would be
the single most misleading thing a report could do. What *is* available:

* **The point annotations shipped with the acquisition.** They are an independent
  detector's opinion, so agreement with them measures similarity to that detector,
  not correctness. Reported as such.
* **The input channel itself.** The model saw one channel, so its mask must sit on
  that channel's bright voxels. This is the strongest available evidence that the
  mask is on the right structures and in the right place -- a channel or axis mix-up
  still produces a plausible-looking volume, but it cannot produce a mask that
  co-locates with the channel it was given.
* **Geometry.** Foreground fraction, object size distribution and voxels per object
  catch degenerate outputs (empty, saturated, one giant blob) that a percentage
  alone would not.

Everything is measured from the store, so a stale JSON cannot disagree with the
mask it describes. Writes one JSON file for the report to read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _match(reference: np.ndarray, candidate: np.ndarray, tolerance: float) -> dict:
    """Greedy one-to-one match between two Nx3 point sets, in microns.

    Greedy rather than optimal on purpose: it is the same rule the existing
    detection metrics use, so numbers stay comparable between datasets. Reporting a
    different assignment strategy here would make this table and the NuMorph tables
    silently incommensurable.
    """
    from scipy.spatial import cKDTree

    if len(reference) == 0 or len(candidate) == 0:
        return {"matched": 0, "distances": np.array([])}
    tree = cKDTree(candidate)
    distances, indices = tree.query(reference, k=1)
    order = np.argsort(distances)
    used: set[int] = set()
    matched_distances: list[float] = []
    for i in order:
        if distances[i] > tolerance:
            break
        j = int(indices[i])
        if j in used:
            continue
        used.add(j)
        matched_distances.append(float(distances[i]))
    return {"matched": len(matched_distances),
            "distances": np.array(matched_distances)}


def _component_summary(mask: np.ndarray, slab: int = 16) -> dict:
    """Connected-component statistics of the mask, read in z-slabs.

    Slab-wise labelling undercounts objects that cross a slab boundary. The figure
    reported is therefore a *lower bound* on component count and is labelled as one;
    the alternative is holding a 500 M voxel label volume in memory to avoid a 5%
    error in a descriptive statistic.
    """
    from scipy import ndimage as ndi

    structure = np.ones((3, 3, 3), dtype=int)
    sizes: list[int] = []
    for z0 in range(0, mask.shape[0], slab):
        block = np.asarray(mask[z0:z0 + slab])
        labelled, n = ndi.label(block, structure=structure)
        if n:
            counts = np.bincount(labelled.ravel())[1:]
            sizes.extend(int(v) for v in counts)
    sizes_arr = np.array(sizes, dtype=np.int64)
    if sizes_arr.size == 0:
        return {"components_lower_bound": 0, "voxels_per_component": None}
    return {
        "components_lower_bound": int(sizes_arr.size),
        "voxels_per_component": {
            "median": float(np.median(sizes_arr)),
            "p25": float(np.percentile(sizes_arr, 25)),
            "p75": float(np.percentile(sizes_arr, 75)),
            "p99": float(np.percentile(sizes_arr, 99)),
            "max": int(sizes_arr.max()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--out", required=True, help="JSON to write")
    parser.add_argument("--level", default="0")
    parser.add_argument("--layer", default="labels/nuclei")
    parser.add_argument("--channel", default=None,
                        help="channel index or omero label; default = the [MODEL INPUT] one")
    parser.add_argument("--points-group", default=None,
                        help="points group to compare against, e.g. points_CD8")
    parser.add_argument("--tolerance-um", type=float, default=7.4,
                        help="match tolerance; 7.4 um is one nucleus radius at this "
                             "dataset's XY sampling and matches the ear report")
    parser.add_argument("--planes", type=int, default=12)
    parser.add_argument("--name", default=None,
                        help="short dataset name for the report, e.g. 'TE24 ear'")
    parser.add_argument("--wall-clock", default=None,
                        help="inference wall clock to print, e.g. '24.1 min on one T4'")
    args = parser.parse_args()

    import zarr

    root = zarr.open_group(args.zarr, mode="r")
    image = root[args.level]
    mask = root[args.layer][args.level]
    channels = root.attrs.get("omero", {}).get("channels", [])
    labels = [c.get("label", "") for c in channels]

    # ---- which channel is the model's input -------------------------------
    if args.channel is not None:
        if isinstance(args.channel, str) and not args.channel.lstrip("-").isdigit():
            if args.channel not in labels:
                raise SystemExit(f"channel {args.channel!r} not in {labels}")
            channel = labels.index(args.channel)
        else:
            channel = int(args.channel)
    else:
        marked = [i for i, text in enumerate(labels)
                  if "MODEL INPUT" in text.upper()]
        channel = marked[0] if marked else 0

    entry = root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    scale = np.array(entry[0]["scale"], dtype=float)[-3:]
    translate = np.array(entry[1]["translation"], dtype=float)[-3:]

    result: dict = {
        "store": str(args.zarr),
        "name": args.name or Path(args.zarr).stem,
        "layer": args.layer,
        "level": args.level,
        "dtype": str(np.asarray(mask[:1]).dtype),
        "wall_clock": args.wall_clock,
        "shape_zyx": [int(v) for v in mask.shape],
        "voxel_um_zyx": [float(v) for v in scale],
        "translation_um_zyx": [float(v) for v in translate],
        "extent_um_zyx": [float(s * n) for s, n in zip(scale, mask.shape)],
        "channels": labels,
        "channel_index": channel,
        "channel_label": labels[channel] if channels else "?",
        "tolerance_um": args.tolerance_um,
    }

    # ---- geometry ---------------------------------------------------------
    total = int(np.prod(mask.shape))
    foreground = 0
    for z0 in range(0, mask.shape[0], 32):
        foreground += int(np.asarray(mask[z0:z0 + 32]).sum())
    result["foreground_voxels"] = foreground
    result["foreground_fraction"] = foreground / total
    result["values"] = sorted(np.unique(
        np.asarray(mask[: min(32, mask.shape[0])])).tolist())

    result.update(_component_summary(mask))
    if result.get("voxels_per_component"):
        median_vox = result["voxels_per_component"]["median"]
        # Volume of an ellipsoid with the acquisition's own voxel size. A single-slice
        # disc-like target has an expected volume near the XY voxel volume, so this
        # ratio says whether objects are discs or filled bodies.
        result["voxel_volume_um3"] = float(np.prod(scale))
        result["median_component_volume_um3"] = float(median_vox * np.prod(scale))

    # ---- co-location with the input channel --------------------------------
    # The model only ever saw this channel. Its output must sit on that channel's
    # bright voxels; if it does not, either the mask or the channel selection is wrong.
    #
    # Thresholding "bright" and counting hits is the obvious test and it is the wrong
    # one for this target. The training label is a single-slice disc at each nucleus
    # middle, so a mask voxel is 1-5 voxels -- it need not include the brightest voxel
    # of the nucleus it marks, and a fixed top-1% threshold scored a correct mask at
    # 0.03 precision on the ear. A percentile rank is scale-free and asks the question
    # that matters: are mask voxels high in their own plane's distribution?
    ranks: list[float] = []
    hits = false_pos = 0
    bright_n = 0
    sampled = 0
    plane_indices = np.linspace(0, mask.shape[0] - 1, args.planes).astype(int)
    for z in plane_indices:
        plane = np.asarray(image[channel, z] if image.ndim == 4 else image[z],
                           dtype=np.float64)
        m = np.asarray(mask[z]).astype(bool)
        if plane.shape != m.shape:
            raise SystemExit(f"plane {z}: image {plane.shape} vs mask {m.shape}")
        if not m.any():
            continue
        # Percentile of each mask voxel within this plane. Ranking is computed once
        # per plane, then sampled at the mask, which is O(n log n) rather than O(n*m).
        flat = plane.ravel()
        order = np.argsort(flat, kind="stable")
        rank = np.empty(flat.size, dtype=np.float64)
        rank[order] = np.arange(flat.size, dtype=np.float64) / max(flat.size - 1, 1)
        ranks.append(float(rank[m.ravel()].mean()))
        # A generous bright set for the hit-rate figure: the top 5% of the plane.
        bright = flat >= np.percentile(flat, 95)
        hits += int(np.count_nonzero(m.ravel() & bright))
        false_pos += int(np.count_nonzero(~m.ravel() & bright))
        bright_n += int(bright.sum())
        sampled += int(flat.size)
    recall = hits / max(hits + false_pos, 1)
    result["colocation"] = {
        "planes": int(len(plane_indices)),
        "sampled_voxels": sampled,
        "mean_percentile_rank_of_mask_voxels": float(np.mean(ranks)) if ranks else None,
        "bright_definition": "above the 95th percentile of each sampled plane",
        "mask_on_bright": hits,
        "bright_outside_mask": false_pos,
        "bright_voxels": bright_n,
        "fraction_of_bright_covered": recall,
    }

    # Mean intensity under the mask vs elsewhere, on the same sampled planes. This is
    # the direct "is the mask on something brighter than background" test and needs no
    # threshold choice at all.
    on_sum = off_sum = 0.0
    on_n = off_n = 0
    for z in plane_indices:
        plane = np.asarray(image[channel, z] if image.ndim == 4 else image[z],
                           dtype=np.float64)
        m = np.asarray(mask[z]).astype(bool)
        on_sum += float(plane[m].sum())
        on_n += int(m.sum())
        off_sum += float(plane[~m].sum())
        off_n += int((~m).sum())
    result["intensity"] = {
        "on_mask_mean": on_sum / max(on_n, 1),
        "off_mask_mean": off_sum / max(off_n, 1),
        "ratio": (on_sum / max(on_n, 1)) / max(off_sum / max(off_n, 1), 1e-9),
    }

    # ---- against the point annotations ------------------------------------
    group = args.points_group
    if group is None:
        candidates = sorted(k for k in root.keys() if k.startswith("points_"))
        if candidates:
            group = candidates[0]
    if group and group in root:
        points_um = np.stack([
            np.asarray(root[group]["z"][:], dtype=float),
            np.asarray(root[group]["y"][:], dtype=float),
            np.asarray(root[group]["x"][:], dtype=float),
        ], axis=1)
        # Centroids of the mask, in microns, computed from the same slab labelling as
        # the component summary. A centroid is the mean of voxel indices, so it can
        # land just outside the crop; those are dropped rather than clamped, because a
        # clamped centroid would report a false match at the boundary.
        from scipy import ndimage as ndi

        structure = np.ones((3, 3, 3), dtype=int)
        centroids_vox: list[np.ndarray] = []
        for z0 in range(0, mask.shape[0], 16):
            block = np.asarray(mask[z0:z0 + 16])
            labelled, n = ndi.label(block, structure=structure)
            if not n:
                continue
            ids = np.arange(1, n + 1)
            local = np.array(ndi.center_of_mass(block, labelled, ids))
            local[:, 0] += z0
            centroids_vox.append(local)
        centroids = (np.vstack(centroids_vox) * scale + translate
                     if centroids_vox else np.empty((0, 3)))
        forward = _match(points_um, centroids, args.tolerance_um)
        backward = _match(centroids, points_um, args.tolerance_um)
        distances = forward["distances"]
        result["points"] = {
            "group": group,
            "n_points": int(len(points_um)),
            "n_components": int(len(centroids)),
            "points_with_component": int(forward["matched"]),
            "components_with_point": int(backward["matched"]),
            "precision": float(backward["matched"] / max(len(centroids), 1)),
            "recall": float(forward["matched"] / max(len(points_um), 1)),
            "f1": float(
                2 * forward["matched"]
                / max(len(points_um) + len(centroids), 1)
            ),
            "median_distance_um": float(np.median(distances)) if distances.size else None,
            "p90_distance_um": float(np.percentile(distances, 90)) if distances.size else None,
            "note": (
                "Point annotations are one detector's output, not ground truth. "
                "Agreement measures similarity to that detector."
            ),
        }
    else:
        result["points"] = None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")

    print(f"mask      {result['shape_zyx']}  foreground {result['foreground_fraction']*100:.4f}%")
    print(f"values    {result['values']}")
    print(f"components {result['components_lower_bound']:,} (lower bound, slab-wise)")
    if result.get("voxels_per_component"):
        print(f"voxels/component median {result['voxels_per_component']['median']:.1f}")
    rank = result["colocation"]["mean_percentile_rank_of_mask_voxels"]
    print(f"colocation mean rank {rank:.3f}  "
          f"covers {result['colocation']['fraction_of_bright_covered']:.3f} of top-5% voxels")
    print(f"intensity on/off mask {result['intensity']['ratio']:.2f}x")
    if result["points"]:
        p = result["points"]
        print(f"{p['group']}: {p['n_points']:,} points vs {p['n_components']:,} components")
        print(f"  precision {p['precision']:.3f}  recall {p['recall']:.3f}  f1 {p['f1']:.3f}")
        if p["median_distance_um"] is not None:
            print(f"  median match distance {p['median_distance_um']:.2f} um")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
