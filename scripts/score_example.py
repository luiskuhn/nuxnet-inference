"""Score a prediction against the bundled example ground-truth mask.

The example pair ships with the repository precisely so that "does it work" has a
number attached to it rather than an impression. This script produces that number, and
it reports more than one, because on this dataset a single Dice value is ambiguous:

* **training-grid Dice** -- the mask resampled onto the grid the network was trained on,
  compared with the ground truth resampled the same way. This is what the network
  achieves, and it is the fair measure of the model.
* **input-grid Dice** -- the mask as written, on the input sampling, compared with the
  ground-truth mask as stored. This is what a downstream tool actually sees.
* **round-trip ceiling** -- the Dice a *perfect* mask scores after the same resampling
  round trip, with no model involved. No prediction can exceed it on the input grid.

Without the ceiling, the gap between the first two looks like model error. On this
dataset most of that gap is the projection: 64 slices compress to 53 and back.

Usage:
    python scripts/score_example.py \
        --prediction runs/c075.ome.tiff \
        --truth examples/numorph_c075_0007/c075_0007_Bottom3.mask.ome.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from nuxnet_inference_package.data_loading.data_loader import resample_volume_pair
from nuxnet_inference_package.preprocessing import voxel_size_from_ome

TARGET = (3.0, 1.0, 1.0)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


def read_mask(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with tifffile.TiffFile(path) as tif:
        volume = tif.series[0].asarray()
        ome = getattr(tif, "ome_metadata", None) or getattr(
            tif.series[0], "ome_metadata", None
        )
    if volume.ndim > 3:
        volume = np.squeeze(volume)
    return (volume > 0).astype(np.uint8), voxel_size_from_ome(ome)


def to_training_grid(mask: np.ndarray, source_spacing: np.ndarray, target=TARGET):
    """Resample a binary mask onto the training grid.

    ``resample_volume_pair`` returns ``(resampled_image, resampled_label)``. A mask must
    come from the LABEL channel: that one is nearest-neighbour, so it stays binary.
    Taking the image channel interpolates the mask and silently changes which voxels are
    foreground -- it reported a ceiling BELOW the prediction, which is impossible and is
    how this was caught.
    """
    target = np.asarray(target, dtype=np.float64)
    if np.allclose(np.asarray(source_spacing), target):
        return mask, False
    _, resampled_label = resample_volume_pair(
        mask[None].astype(np.float32),
        mask.astype(np.int64),
        np.asarray(source_spacing, dtype=np.float64),
        target,
    )
    return (resampled_label > 0).astype(np.uint8), True


def back_to_input_grid(on_grid: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Trilinear on the binary mask, re-thresholded at 0.5 -- the CLI's own rule."""
    import torch

    tensor = torch.from_numpy(on_grid[None, None].astype(np.float32))
    back = torch.nn.functional.interpolate(
        tensor, size=shape, mode="trilinear", align_corners=False
    )
    return (back[0, 0].numpy() >= 0.5).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--target-voxel-spacing", default="3.0,1.0,1.0")
    args = parser.parse_args()

    target = tuple(float(v) for v in args.target_voxel_spacing.split(","))

    prediction, _ = read_mask(Path(args.prediction))
    truth, truth_spacing = read_mask(Path(args.truth))

    if truth_spacing is None:
        raise SystemExit(f"{args.truth} carries no PhysicalSize; cannot establish the grid")

    if prediction.shape != truth.shape:
        raise SystemExit(
            f"shape mismatch: prediction {prediction.shape}, truth {truth.shape}. "
            "The prediction should be on the input grid, so these must agree."
        )

    truth_grid, resampled = to_training_grid(truth, truth_spacing, target)

    # The prediction is written on the input grid; put it on the training grid for the
    # architecture-level comparison, with the same nearest-neighbour label rule.
    if resampled:
        pred_grid, _ = to_training_grid(prediction, truth_spacing, target)
    else:
        pred_grid = prediction

    # Ceiling: truth on the training grid, projected back by the same rule.
    ceiling_mask = back_to_input_grid(truth_grid, truth.shape) if resampled else truth

    train_dice = dice(pred_grid, truth_grid)
    input_dice = dice(prediction, truth)
    ceiling = dice(ceiling_mask, truth)

    print(f"source voxel size   {np.round(truth_spacing, 4).tolist()} um")
    print(f"target voxel size   {list(target)} um")
    print(f"resampled           {resampled}")
    print()
    print(f"training-grid Dice  {train_dice:.4f}   <- network, from the written mask")
    print(f"round-trip ceiling  {ceiling:.4f}   <- a PERFECT mask after the same round trip")
    print(f"input-grid Dice     {input_dice:.4f}   <- what was written to disk")
    if ceiling > 0:
        print(f"fraction of ceiling {input_dice / ceiling * 100:.1f}%")
    print()
    print(
        "Note on training-grid Dice: this is measured from the mask on disk, so it has\n"
        "already been projected to the input grid and back. Scoring directly on the\n"
        "training grid, before projection, gives a slightly higher number (0.870 vs\n"
        "0.860 on this volume) because nothing is lost in the round trip. Both are\n"
        "correct; they answer slightly different questions."
    )
    print()


if __name__ == "__main__":
    main()
