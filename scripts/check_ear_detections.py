"""Is the extra detection real? Test each predicted component against the nucleus channel.

The Imaris comparison on its own is not interpretable. Imaris "Spots - Nuclei" is a
detector with its own parameters, not ground truth, so agreement with it measures
*similarity to another algorithm*, not accuracy. The number that matters is whether a
predicted component sits on something that looks like a nucleus.

This resolves the ambiguity directly. For every predicted component it reports the
mean intensity of the nucleus channel (SYTOX) in a small window around its centroid,
split into three groups:

* **matched**    -- an Imaris spot is within tolerance. Both agree.
* **unmatched**  -- no Imaris spot within tolerance. Either a false positive or a
                    nucleus Imaris missed. Intensity decides which.
* **tissue**     -- random voxels with signal but no prediction, as a floor.

If unmatched components carry intensity comparable to matched ones, they are nuclei
that Imaris did not call, and precision against Imaris is a measure of disagreement
between two detectors rather than of error. If they sit near the tissue floor, they
are genuine false positives and the number should be read as a shortfall.

It also reports the assignment multiplicity -- how many predicted components fall to
the same Imaris spot -- because over-segmentation and false positives look identical
in a precision figure and are fixed differently.

    python scripts/check_ear_detections.py \\
        --zarr /zarr/ear1_1-6m.lite.zarr --layer nuclei_mask_roi \\
        --spots /data/roi_crop.spots.npz \\
        --z-range 48:264 --y-range 1480:2170 --x-range 400:1030
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NUCLEUS_CHANNEL = 0
NUCLEUS_DIAMETER_UM = 14.8


def span(text: str) -> tuple[int, int]:
    start, _, stop = text.partition(":")
    return int(start), int(stop)


def load_spots(path: str, z_range, y_range, x_range, key: str = "Points0") -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    x = data[f"{key}__x_px"]
    y = data[f"{key}__y_px"]
    z = data[f"{key}__z_px"]
    keep = (
        (z >= z_range[0]) & (z < z_range[1])
        & (y >= y_range[0]) & (y < y_range[1])
        & (x >= x_range[0]) & (x < x_range[1])
    )
    return np.stack([z[keep], y[keep], x[keep]], axis=1)


def component_centroids(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3)))
    if count == 0:
        return np.zeros((0, 3))
    return np.array(ndi.center_of_mass(mask, labels, range(1, count + 1)))


def sample_intensity(volume: np.ndarray, points: np.ndarray, radius: int = 2) -> np.ndarray:
    """Mean intensity in a cube around each point, skipping points out of bounds."""
    if len(points) == 0:
        return np.zeros(0, dtype=np.float32)
    shape = np.array(volume.shape)
    index = np.rint(points).astype(int)
    out = np.zeros(len(index), dtype=np.float32)
    for i, (z, y, x) in enumerate(index):
        z0, z1 = max(0, z - radius), min(shape[0], z + radius + 1)
        y0, y1 = max(0, y - radius), min(shape[1], y + radius + 1)
        x0, x1 = max(0, x - radius), min(shape[2], x + radius + 1)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            out[i] = 0.0
            continue
        out[i] = float(volume[z0:z1, y0:y1, x0:x1].mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--channel", type=int, default=NUCLEUS_CHANNEL)
    parser.add_argument("--spots", required=True)
    parser.add_argument("--spots-key", default="Points0")
    parser.add_argument("--z-range", required=True)
    parser.add_argument("--y-range", required=True)
    parser.add_argument("--x-range", required=True)
    parser.add_argument("--voxel-um", default="5.005,3.5455,3.5455")
    parser.add_argument("--tolerance-um", type=float, default=NUCLEUS_DIAMETER_UM / 2)
    parser.add_argument("--radius", type=int, default=2,
                        help="half-width of the intensity sampling cube, source voxels")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    voxel = np.array([float(v) for v in args.voxel_um.split(",")])
    z_range, y_range, x_range = span(args.z_range), span(args.y_range), span(args.x_range)

    import zarr

    root = zarr.open_group(args.zarr, mode="r")
    group = root[args.layer] if args.layer else root
    mask = np.asarray(group["0"][:]) > 0
    print(f"mask        {mask.shape}  {mask.sum():,} voxels")

    # The nucleus image, cropped to the same region. Shape must match the mask, which is
    # on the source grid, so it does.
    #
    # Build the index by axis NAME rather than a positional tuple: the store is
    # [t, c, z, y, x], but a [c, z, y, x] store would silently return the wrong plane
    # from the same hardcoded index.
    series_attrs = dict(root["0"].attrs.get("multiscales", [{}])[0])
    series_axes = [a["name"] for a in series_attrs.get("axes", [])]
    array = root["0"]["0"]
    index = []
    for axis, length in zip(series_axes, array.shape):
        if axis == "t":
            index.append(0)
        elif axis == "c":
            index.append(args.channel)
        elif axis == "z":
            index.append(slice(z_range[0], z_range[1]))
        elif axis == "y":
            index.append(slice(y_range[0], y_range[1]))
        elif axis == "x":
            index.append(slice(x_range[0], x_range[1]))
        else:
            index.append(0)
    image = np.asarray(array[tuple(index)])
    if image.ndim > 3:
        image = np.squeeze(image)
    print(f"channel {args.channel}   {image.shape}  {image.dtype}  "
          f"min {image.min()}  max {image.max()}")

    found = component_centroids(mask)
    points = load_spots(args.spots, z_range, y_range, x_range, args.spots_key)
    local = points - np.array([z_range[0], y_range[0], x_range[0]])
    print(f"components  {len(found):,}")
    print(f"Imaris      {len(local):,}")

    # --- distance from every component to its nearest Imaris spot ---------------
    from scipy.spatial import cKDTree

    found_um = found * voxel
    truth_um = local * voxel
    tree_truth = cKDTree(truth_um)
    distance, nearest = tree_truth.query(found_um)
    matched = distance <= args.tolerance_um

    # How many components claim the same Imaris spot -> over-segmentation vs false positive.
    counts = np.bincount(nearest[matched], minlength=len(local))
    multiplicity = counts[counts > 0]

    # --- intensity test --------------------------------------------------------
    r = args.radius
    intensity = sample_intensity(image, found, radius=r)

    # Tissue floor: voxels with signal, sampled on a coarse grid, away from any prediction.
    stride = 8
    coarse = image[::stride, ::stride, ::stride]
    coarse_mask = mask[::stride, ::stride, ::stride]
    tissue = coarse[(coarse > 0) & (~coarse_mask)]
    nonzero = image[image > 0]

    def stats(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"n": 0}
        return {
            "n": int(values.size),
            "mean": round(float(values.mean()), 1),
            "median": round(float(np.median(values)), 1),
            "p25": round(float(np.percentile(values, 25)), 1),
            "p75": round(float(np.percentile(values, 75)), 1),
        }

    groups = {
        "matched": stats(intensity[matched]),
        "unmatched": stats(intensity[~matched]),
        "all_components": stats(intensity),
        "tissue_with_signal": stats(tissue),
        "all_signal": stats(nonzero),
    }

    print()
    print("nucleus-channel intensity at component centroids (mean of a "
          f"{2*r+1}^3 cube)")
    for name, s in groups.items():
        if s["n"] == 0:
            print(f"  {name:22s} n=0")
            continue
        print(f"  {name:22s} n={s['n']:>7,}  mean {s['mean']:>8.1f}  "
              f"median {s['median']:>8.1f}  IQR {s['p25']:.0f}-{s['p75']:.0f}")

    ratio = (groups["unmatched"]["median"] / groups["matched"]["median"]
             if groups["unmatched"]["n"] and groups["matched"]["n"] else float("nan"))
    floor_ratio = (groups["unmatched"]["median"] / groups["tissue_with_signal"]["median"]
                   if groups["unmatched"]["n"] and groups["tissue_with_signal"]["n"] else float("nan"))
    print()
    print(f"  unmatched / matched median        {ratio:.2f}")
    print(f"  unmatched / tissue-signal median  {floor_ratio:.2f}")

    # --- chance baseline -------------------------------------------------------
    # The comparison above could still be argued either way: unmatched components are
    # brighter than random tissue, but so is any point near tissue. The decisive test
    # is an *equal-sized random sample from the same intensity distribution*. Place as
    # many random centroids as there are unmatched components, drawn from voxels that
    # have signal, and measure the same thing. If random points drawn from the tissue
    # distribution score far below the unmatched components, then the unmatched
    # components are not explained by "it just landed in bright tissue" -- they are
    # chosen positions on nucleus-like objects.
    rng = np.random.default_rng(0)
    signal_coords = np.argwhere(image > np.percentile(image[image > 0], 50))
    n_random = int((~matched).sum())
    if signal_coords.size and n_random:
        picks = signal_coords[rng.choice(len(signal_coords), size=n_random,
                                         replace=len(signal_coords) < n_random)]
        random_intensity = sample_intensity(image, picks.astype(float), radius=r)
        random_stats = stats(random_intensity)
        print()
        print("chance baseline (random points on above-median signal voxels, "
              f"n={n_random:,})")
        print(f"  random                median {random_stats['median']:>8.1f}  "
              f"IQR {random_stats['p25']:.0f}-{random_stats['p75']:.0f}")
        print(f"  unmatched components  median "
              f"{groups['unmatched']['median']:>8.1f}  "
              f"IQR {groups['unmatched']['p25']:.0f}-{groups['unmatched']['p75']:.0f}")
        print(f"  unmatched / random    "
              f"{groups['unmatched']['median'] / random_stats['median']:.2f}")
    else:
        random_stats = {"n": 0}

    print()
    print("assignment multiplicity (components per matched Imaris spot)")
    if multiplicity.size:
        print(f"  spots receiving >=1 component  {multiplicity.size:,} of {len(local):,}")
        print(f"  components per such spot       mean {multiplicity.mean():.2f}  "
              f"median {np.median(multiplicity):.0f}  max {multiplicity.max()}")
        for k in (1, 2, 3, 4):
            n = int((multiplicity == k).sum())
            print(f"    exactly {k}: {n:,}")

    print()
    print("nearest-Imaris distance for components (um)")
    for q in (50, 75, 90, 95):
        print(f"  p{q:<3d} {np.percentile(distance, q):8.2f}")

    result = {
        "region": {"z": list(z_range), "y": list(y_range), "x": list(x_range)},
        "channel": args.channel,
        "components": int(len(found)),
        "imaris_spots": int(len(local)),
        "tolerance_um": float(args.tolerance_um),
        "sampling_radius_voxels": r,
        "intensity": groups,
        "unmatched_over_matched_median": round(float(ratio), 3),
        "unmatched_over_tissue_median": round(float(floor_ratio), 3),
        "chance_baseline": (
            {
                "n": random_stats.get("n", 0),
                "median": random_stats.get("median"),
                "p25": random_stats.get("p25"),
                "p75": random_stats.get("p75"),
                "unmatched_over_random_median": round(
                    float(groups["unmatched"]["median"] / random_stats["median"]), 3
                ) if random_stats.get("median") else None,
            }
            if random_stats.get("n") else None
        ),
        "multiplicity": {
            "spots_with_component": int(multiplicity.size),
            "mean": round(float(multiplicity.mean()), 3) if multiplicity.size else None,
            "max": int(multiplicity.max()) if multiplicity.size else None,
        },
        "distance_um_percentiles": {
            f"p{q}": round(float(np.percentile(distance, q)), 2)
            for q in (50, 75, 90, 95)
        },
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2) + "\n")
        print()
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
