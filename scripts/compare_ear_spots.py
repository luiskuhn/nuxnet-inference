"""Compare nuxnet-inference output against Imaris spot labels on a real ear.

This is the one comparison in the repository that is not NuMorph. NuMorph gives
masks; Imaris gives *points*. The two labels were produced by different processes,
so scoring them the same way would be wrong:

* **NuMorph masks** (what the model was trained on) are 2D single-slice discs
  marking a nucleus's middle Z, curated for detection.
* **Imaris "Spots - Nuclei"** are sub-voxel centroid detections carrying a fixed
  4.0 um parameter radius. They mark where a nucleus is, not what shape it has.

So the comparison renders the Imaris points into the same disc format the model
was trained to emit and scores geometric agreement. Comparing a detection to a
segmentation would blame the model for a difference in the label, not in the
prediction.

Two scores, because they answer different questions:

* **Dice** on the rendered discs -- comparable with the NuMorph figures.
* **Detection matching** at a physical tolerance -- did we find the nucleus at
  all? Precision/recall/F1 plus the distance distribution, which is what a
  biologist actually wants to know.

Everything is measured on the **source grid** (the Zarr's own sampling), because
that is where the Imaris spots live and where a user would compare outputs.

    python scripts/compare_ear_spots.py \
        --prediction /zarr/ear1_1-6m.lite.zarr --layer nuclei_mask_roi \
        --spots /data/roi_crop.spots.npz \
        --z-range 48:264 --y-range 1480:2170 --x-range 400:1030
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: The nucleus channel, measured rather than assumed. scripts in the analysis
#: branch ranked the ear's five channels by how well each separates Imaris
#: nucleus positions from surrounding tissue: channel 0 (Nuclei / SYTOX Green)
#: gives d-prime 1.00, while the myeloid markers give negative d-prime, i.e. no
#: nucleus contrast at all.
NUCLEUS_CHANNEL = 0

#: Imaris stores a 4.0 um radius for every spot. It is a detection parameter, not
#: a measured nucleus radius, so it is deliberately NOT used as the disc radius.
IMARIS_PARAMETER_RADIUS_UM = 4.0

#: Measured on 396 isolated nuclei in the parent project (p25 6.9, p75 7.8).
NUCLEUS_DIAMETER_UM = 14.8


def load_spots(path: str, z_range, y_range, x_range, key: str = "Points0") -> np.ndarray:
    """Imaris nucleus spots inside a (z, y, x) pixel window, as float pixels."""
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


def render_discs(shape, points: np.ndarray, radius_xy: int = 1) -> np.ndarray:
    """Render sub-voxel centroids as single-slice discs.

    Single slice, not a sphere: that is the NuMorph convention the model was
    trained to produce. A sphere would compare the model's 2D target against a 3D
    label and depress Dice for a reason that has nothing to do with the model.

    ``radius_xy=1`` gives a 3x3 plus-shape (5 voxels) rather than a 3x3 square,
    because a square overstates area by 9/5 and would inflate Dice.
    """
    mask = np.zeros(shape, dtype=bool)
    if len(points) == 0:
        return mask
    zi = np.rint(points[:, 0]).astype(int)
    yi = np.rint(points[:, 1]).astype(int)
    xi = np.rint(points[:, 2]).astype(int)
    ok = (
        (zi >= 0) & (zi < shape[0])
        & (yi >= 0) & (yi < shape[1])
        & (xi >= 0) & (xi < shape[2])
    )
    zi, yi, xi = zi[ok], yi[ok], xi[ok]
    mask[zi, yi, xi] = True
    if radius_xy >= 1:
        for dz, dy, dx in ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)):
            yy = np.clip(yi + dy, 0, shape[1] - 1)
            xx = np.clip(xi + dx, 0, shape[2] - 1)
            mask[zi, yy, xx] = True
    return mask


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2.0 * np.logical_and(a, b).sum() / total)


def component_centroids(mask: np.ndarray) -> np.ndarray:
    """Centroids of connected components, in voxel coordinates."""
    from scipy import ndimage as ndi

    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3)))
    if count == 0:
        return np.zeros((0, 3))
    return np.array(ndi.center_of_mass(mask, labels, range(1, count + 1)))


def match_detections(found: np.ndarray, truth: np.ndarray,
                     voxel_um: np.ndarray, tolerance_um: float) -> dict:
    """Greedy nearest-neighbour matching in physical microns.

    Greedy rather than optimal assignment because it is the stricter behaviour:
    it cannot steal a neighbour to raise its own count, so a reported precision
    is not an artefact of a lucky assignment.
    """
    if len(found) == 0 or len(truth) == 0:
        return {"matched": 0, "distances": np.array([]),
                "n_found": len(found), "n_truth": len(truth)}

    from scipy.spatial import cKDTree

    # Scale before querying: matching in voxels would treat a 5 um Z step as
    # equal to a 3.5 um XY step and bias every distance.
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
    return {"matched": len(distances), "distances": np.array(distances),
            "n_found": len(found), "n_truth": len(truth)}


def load_mask(path: str, layer: str | None = None) -> np.ndarray:
    """Read a mask from an OME-NGFF layer or an OME-TIFF."""
    if path.endswith(".zarr") or path.endswith(".zarr/"):
        import zarr

        root = zarr.open_group(path, mode="r")
        group = root[layer] if layer else root
        array = group["0"]
        return np.asarray(array[:]) > 0

    import tifffile

    with tifffile.TiffFile(path) as tif:
        volume = tif.series[0].asarray()
    if volume.ndim > 3:
        volume = np.squeeze(volume)
    return volume > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True,
                        help="OME-Zarr store or OME-TIFF containing the mask")
    parser.add_argument("--layer", default=None,
                        help="group inside the Zarr holding the mask")
    parser.add_argument("--spots", default="/data/roi_crop.spots.npz")
    parser.add_argument("--spots-key", default="Points0")
    parser.add_argument("--voxel-um", default="5.005,3.5455,3.5455", help="Z,Y,X")
    parser.add_argument("--z-range", required=True)
    parser.add_argument("--y-range", required=True)
    parser.add_argument("--x-range", required=True)
    parser.add_argument("--tolerance-um", type=float, default=None,
                        help=f"match radius; default is half a nucleus "
                             f"({NUCLEUS_DIAMETER_UM / 2:.1f} um)")
    parser.add_argument("--disc-radius", type=int, default=1)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    voxel = np.array([float(v) for v in args.voxel_um.split(",")])
    tolerance = (args.tolerance_um if args.tolerance_um is not None
                 else NUCLEUS_DIAMETER_UM / 2.0)

    def span(text: str) -> tuple[int, int]:
        start, _, stop = text.partition(":")
        return int(start), int(stop)

    z_range, y_range, x_range = span(args.z_range), span(args.y_range), span(args.x_range)
    shape = (z_range[1] - z_range[0], y_range[1] - y_range[0], x_range[1] - x_range[0])

    prediction = load_mask(args.prediction, args.layer)
    if prediction.shape != shape:
        raise SystemExit(
            f"prediction shape {prediction.shape} != region {shape}. "
            f"For a Zarr layer the array must cover exactly the requested region; "
            f"if the layer spans the whole volume, crop it before scoring."
        )

    points = load_spots(args.spots, z_range, y_range, x_range, args.spots_key)
    local = points - np.array([z_range[0], y_range[0], x_range[0]])
    truth_mask = render_discs(shape, local, radius_xy=args.disc_radius)

    print(f"region        Z {z_range[0]}:{z_range[1]}  Y {y_range[0]}:{y_range[1]}  "
          f"X {x_range[0]}:{x_range[1]}  = {shape}")
    print(f"voxel         Z {voxel[0]:.3f}  Y {voxel[1]:.3f}  X {voxel[2]:.3f} um")
    print(f"model mask    {prediction.sum():,} voxels  "
          f"({100 * prediction.mean():.4f}% foreground)")
    print(f"Imaris truth  {len(points):,} spots -> {truth_mask.sum():,} disc voxels  "
          f"({100 * truth_mask.mean():.4f}%)")
    print()

    overlap = dice(prediction, truth_mask)
    print(f"Dice (rendered marker discs)   {overlap:.4f}")

    found = component_centroids(prediction)
    matched = match_detections(found, local, voxel, tolerance)
    precision = matched["matched"] / max(matched["n_found"], 1)
    recall = matched["matched"] / max(matched["n_truth"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    distances = matched["distances"]

    print()
    print(f"detection at {tolerance:.1f} um tolerance "
          f"(half a {NUCLEUS_DIAMETER_UM:.1f} um nucleus)")
    print(f"  model components   {matched['n_found']:,}")
    print(f"  Imaris spots       {matched['n_truth']:,}")
    print(f"  matched            {matched['matched']:,}")
    print(f"  precision          {precision:.3f}")
    print(f"  recall             {recall:.3f}")
    print(f"  F1                 {f1:.3f}")
    if distances.size:
        print(f"  match distance     median {np.median(distances):.2f} um  "
              f"p90 {np.percentile(distances, 90):.2f} um")

    if len(found) and len(local):
        print()
        print("centroid cloud comparison (a displacement shows up here)")
        print(f"  model  mean z,y,x {np.round(found.mean(axis=0), 2)}")
        print(f"  truth  mean z,y,x {np.round(local.mean(axis=0), 2)}")
        print(f"  delta  mean z,y,x "
              f"{np.round(found.mean(axis=0) - local.mean(axis=0), 2)} voxels")

    result = {
        "region": {"z": list(z_range), "y": list(y_range), "x": list(x_range)},
        "voxel_um_zyx": voxel.tolist(),
        "prediction_voxels": int(prediction.sum()),
        "prediction_foreground_pct": float(100 * prediction.mean()),
        "imaris_spots": int(len(points)),
        "truth_disc_voxels": int(truth_mask.sum()),
        "dice_marker_discs": float(overlap),
        "tolerance_um": float(tolerance),
        "model_components": int(matched["n_found"]),
        "matched": int(matched["matched"]),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "median_match_distance_um": float(np.median(distances)) if distances.size else None,
        "p90_match_distance_um": (float(np.percentile(distances, 90))
                                  if distances.size else None),
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
