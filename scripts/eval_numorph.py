"""Run the CLI over NuMorph volumes on solidfox and score every stage.

Deliberately computes and prints, for each volume:
  * the training-grid Dice (what the model actually achieves)
  * the round-trip ceiling (what a perfect mask would score on the input grid)
  * the input-grid Dice of what the CLI wrote
  * instance counts from the unchanged post-processing

The ceiling is the point. Without it, a low input-grid Dice looks like a bad model
when it is mostly the 53 -> 64 projection. Reporting all four makes the comparison
honest and shows whether the export is losing anything recoverable.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from skimage import measure

sys.path.insert(0, os.environ.get("PYTHONPATH", "/work"))

from nuxnet_inference_package.data_loading.data_loader import resample_volume_pair
from nuxnet_inference_package.preprocessing import (
    apply_training_pipeline,
    sliding_window_predict,
    voxel_size_from_ome,
)
from nuxnet_inference_package.model.unet_3d_models import UNet3D

TARGET = (3.0, 1.0, 1.0)
PATCH = (32, 128, 128)
THRESHOLD = 0.8
DATA = Path("/data/NUMORPH_SEM_SEG_DATASET/data")


def read(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        return np.squeeze(tif.series[0].asarray())


def read_ome(path: Path):
    with tifffile.TiffFile(path) as tif:
        volume = np.squeeze(tif.series[0].asarray())
        ome = getattr(tif, "ome_metadata", None)
    return volume, voxel_size_from_ome(ome)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


def upsample_trilinear(mask: np.ndarray, size) -> np.ndarray:
    tensor = torch.from_numpy(mask[None, None].astype(np.float32))
    out = F.interpolate(tensor, size=size, mode="trilinear", align_corners=False)
    return (out[0, 0].numpy() >= 0.5)


def main() -> None:
    weights = sys.argv[1] if len(sys.argv) > 1 else "/models/fold1.pt"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out_dir = Path("/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = UNet3D(in_channels=1, classes=2, dropout=0.1)
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"weights {weights}: {len(state.get('state_dict', state))} tensors, device {device}")

    images = sorted(glob.glob(str(DATA / "*/images/*.ome.tif")))
    print(f"{len(images)} images available, running {min(limit, len(images))}")

    results = []
    for image_path in images[:limit]:
        image_path = Path(image_path)
        mask_path = Path(str(image_path).replace("/images/", "/masks/").replace("_IMAGE_", "_MASK_"))
        if not mask_path.is_file():
            print(f"skip {image_path.name}: no mask")
            continue

        image, voxel_size = read_ome(image_path)
        truth = read(mask_path) > 0
        if voxel_size is None:
            print(f"skip {image_path.name}: no OME spacing")
            continue

        prepared, _ = apply_training_pipeline(image, voxel_size, TARGET, normalize=True)
        _, truth_grid = resample_volume_pair(
            np.zeros((1,) + truth.shape, dtype=np.float32),
            truth.astype(np.int64),
            np.asarray(voxel_size),
            np.asarray(TARGET),
        )
        truth_grid = truth_grid > 0

        result = sliding_window_predict(
            model=model, volume=prepared, patch_size=PATCH, device=device, classes=2
        )
        mask_grid = result.probability >= THRESHOLD

        # what a perfect mask would score after the same round trip
        ceiling = dice(upsample_trilinear(truth_grid, image.shape), truth)

        out_mask = upsample_trilinear(mask_grid, image.shape)
        input_dice = dice(out_mask, truth)

        labeled, n = measure.label(
            out_mask, background=0, connectivity=1, return_num=True
        )
        sizes = np.bincount(labeled.ravel())[1:] if n else np.array([])

        row = {
            "volume": image_path.name,
            "group": image_path.parent.parent.name,
            "source_spacing": [round(float(v), 3) for v in voxel_size],
            "shape": list(image.shape),
            "prepared_shape": list(prepared.shape),
            "tiles": result.n_tiles,
            "train_grid_dice": round(dice(mask_grid, truth_grid), 4),
            "roundtrip_ceiling": round(ceiling, 4),
            "input_grid_dice": round(input_dice, 4),
            "fraction_of_ceiling": round(input_dice / max(ceiling, 1e-9), 4),
            "instances": int(n),
            "median_size_voxels": float(np.median(sizes)) if len(sizes) else 0.0,
            "foreground_pct": round(float(out_mask.mean() * 100), 3),
        }
        results.append(row)
        print(json.dumps(row))

        # persist artifacts for the PDF report
        stem = image_path.name.replace(".ome.tif", "")
        np.save(out_dir / f"{stem}.prob.npy", result.probability.astype(np.float16))
        np.save(out_dir / f"{stem}.prepared.npy", prepared.astype(np.float16))
        np.save(out_dir / f"{stem}.truth_grid.npy", truth_grid.astype(np.uint8))
        np.save(out_dir / f"{stem}.pred_input.npy", out_mask.astype(np.uint8))
        np.save(out_dir / f"{stem}.truth_input.npy", truth.astype(np.uint8))

    summary = out_dir / "summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {summary}")

    if results:
        tr = np.array([r["train_grid_dice"] for r in results])
        ce = np.array([r["roundtrip_ceiling"] for r in results])
        ip = np.array([r["input_grid_dice"] for r in results])
        print(f"\ntrain-grid dice   mean {tr.mean():.3f}  min {tr.min():.3f}  max {tr.max():.3f}")
        print(f"ceiling           mean {ce.mean():.3f}  min {ce.min():.3f}  max {ce.max():.3f}")
        print(f"input-grid dice   mean {ip.mean():.3f}  min {ip.min():.3f}  max {ip.max():.3f}")
        print(f"input / ceiling   mean {(ip / ce).mean():.3f}")


if __name__ == "__main__":
    main()
