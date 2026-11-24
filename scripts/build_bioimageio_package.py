"""Build a BioImage.IO package from a nuxnet-training run.

The checkpoint alone is not a usable model. A consumer needs three things the state
dict cannot tell them:

1. the preprocessing that puts their volume on the grid the weights expect
   (3.0/1.0/1.0 um), because the network has no concept of physical scale;
2. the fact that the outputs are **logits**, not probabilities;
3. a tensor whose shape is already legal for the network, since every spatial axis
   must be divisible by 4.

This script produces a package that states all three and proves them: it runs the
exported weights to generate the test output, so `bioimageio test` compares against
what the weights actually produce rather than a stale artifact.

Usage:
    python build_bioimageio_package.py \
        --weights numorph_fold1_weights.pt \
        --out numorph-nuclei-unet-fold1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile
import torch

from nuxnet_inference_package.model.unet_3d_models import UNet3D

TARGET_VOXEL_SPACING = [3.0, 1.0, 1.0]
PATCH = (32, 128, 128)
CLASSES = 2
DROPOUT = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tensor_files(out: Path, model: UNet3D) -> dict[str, str]:
    """Create the test tensors and the reference output from the weights themselves.

    The input is a fixed-seed ramp plus a bright blob, chosen so the network sees a
    non-degenerate volume. The output is computed, never hard-coded: if the weights
    or architecture drift, the package's own test fails.
    """
    generator = torch.Generator().manual_seed(0)
    volume = torch.rand(PATCH, generator=generator)
    z = torch.linspace(-1, 1, PATCH[0])[:, None, None]
    volume = (volume * 0.4 + torch.exp(-(z**2) * 4.0) * 0.6).clamp(0, 1)
    volume = volume[None, None].float()

    with torch.no_grad():
        logits = model(volume)

    np.save(out / "test-input.npy", volume.numpy())  # (1, 1, z, y, x) to match declared axes
    np.save(out / "test-output.npy", logits.numpy())  # (1, 2, z, y, x)

    # sample_tensor must be a real image format, not .npy: bioimageio validates it with
    # imageio, which has no .npy backend and fails the whole description with
    # "Could not find a backend to open <bytes>". test_tensor is read by bioimageio.core
    # and may be .npy. The trainer's exports make the same split (sample-*.tif +
    # test-*.npy), so match it.
    tifffile.imwrite(out / "sample-input.tif", volume.numpy()[0, 0])
    tifffile.imwrite(out / "sample-output.tif", logits.numpy()[0])

    # Softmax'd marker probability, which is what a reader of the model card cares about.
    probability = torch.softmax(logits, dim=1)[0, 1].numpy()
    print(
        f"test tensor {tuple(PATCH)}: "
        f"logits range {logits.min():.3f}..{logits.max():.3f}, "
        f"marker probability range {probability.min():.4f}..{probability.max():.4f}"
    )
    return {
        "test-input.npy": sha256(out / "test-input.npy"),
        "test-output.npy": sha256(out / "test-output.npy"),
        "sample-input.tif": sha256(out / "sample-input.tif"),
        "sample-output.tif": sha256(out / "sample-output.tif"),
    }


def write_environment(out: Path) -> None:
    (out / "environment.yml").write_text(
        "name: numorph-nuclei-unet\n"
        "channels:\n"
        "  - pytorch\n"
        "  - conda-forge\n"
        "channel_priority: strict\n"
        "dependencies:\n"
        "  - python=3.12\n"
        "  - pip\n"
        "  - pip:\n"
        "      - torch==2.5.1\n"
        "      - numpy==1.26.4\n"
        "      - tifffile==2024.8.30\n"
    )


def write_readme(out: Path, fold: int) -> None:
    (out / "README.md").write_text(
        f"""---
library_name: pytorch
tags:
  - bioimage.io
  - pytorch
---

# NuMorph nucleus-marker U-Net (fold {fold})

A 3D U-Net that detects nuclei in cleared mouse-brain light-sheet data. It was trained
on the NuMorph semantic-segmentation dataset, where each mask entry is a **2D
single-slice disc** marking the middle Z of one nucleus, stored in a 3D volume. The
model reproduces that target: its output is a **marker** channel, and instances are
recovered by grouping the marker voxels afterwards.

The target is neither a nucleus body nor a one-voxel point. Measured on the bundled
example pair: 684 discs, median 63 voxels each, one slice thick, roughly 14 px across
in XY at 2.5/0.75/0.75 um sampling. A prediction whose components are one or two
voxels is badly under-segmenting, not merely "sparse".

## Read this before using the weights

The network has no concept of physical scale. It was trained on volumes resampled to
{", ".join(f"{v:g}" for v in TARGET_VOXEL_SPACING)} um per voxel in Z, Y, X. Feeding it
a volume at its stored sampling will produce something that looks like nuclei and is
wrong -- measured on the bundled example, whole-volume inference at the stored sampling
scores input-grid Dice 0.456, while resampling to the training grid first scores 0.739.

So the contract is:

1. **Resample** your OME-TIFF to {", ".join(f"{v:g}" for v in TARGET_VOXEL_SPACING)} um
   (Z, Y, X). Preserve the physical extent: new length = old length x source spacing /
   target spacing. Use Gaussian anti-aliasing when downsampling and trilinear
   interpolation, and nearest-neighbour for the mask.
2. **Normalize** each volume to [0, 1] by its own min and max, *after* resampling.
3. **Tile** with a {", ".join(str(v) for v in PATCH)} patch and overlap, padding each
   tile so every spatial axis is a multiple of 4. The decoder upsamples twice, so an
   axis that is not a multiple of 4 makes the decoder's feature map one voxel larger
   than its skip connection and the forward pass fails outright.
4. The network returns **logits** with two channels, `background` and `marker`. Apply
   `softmax(dim=1)` (or `argmax`) yourself; the packaged tensors are unnormalized.
5. Blend overlapping tiles, then threshold. On NuMorph data Dice is flat between 0.4
   and 0.9, so 0.5 is a reasonable default and 0.8 favours precision.

Step 1 is not optional and is not something the state dict can enforce for you.

## What this model is not

The target is a flat 2D disc marking a nucleus's middle slice, not a filled 3D nucleus.
If you need nucleus volumes, this model gives you detections to seed a watershed; do
not treat its foreground as a segmentation of nucleus volume. A predicted component
occupies one slice, so its voxel count measures the disc, not the nucleus.

Two consequences worth stating plainly:

* **It is a detection target, not a 3D segmentation target.** Expect one slice per
  object even when the nucleus spans several.
* **The mask is stored in 3D but curated in 2D.** Slices between markers carry no
  information, which is why per-slice metrics on this dataset are naturally bimodal.

## Validation

`bioimageio test` checks the package format, reloads the state dict, and reproduces
`test-output.npy`. That is a technical check only. The scientific numbers below come
from 32 NuMorph image/mask pairs with a held-out fold:

| grid | Dice | note |
|---|---|---|
| training grid ({", ".join(f"{v:g}" for v in TARGET_VOXEL_SPACING)} um) | 0.801 mean | what the network achieves |
| input grid | 0.749 mean | after projecting back to source sampling |

The gap is the export resampling, not model error: a *perfect* mask projected from the
training grid back to a 2.5/0.75/0.75 um input grid scores only 0.883, because the
projection cannot preserve which slice was which.

## Citation

Replace with a resolvable citation before publication.
"""
    )


def write_rdf(out: Path, hashes: dict[str, str], fold: int) -> None:
    weights_sha = sha256(out / "weights.pt")
    architecture_sha = sha256(out / "nuxnet_inference_package" / "model" / "unet_3d_models.py")
    environment_sha = sha256(out / "environment.yml")
    readme_sha = sha256(out / "README.md")

    (out / "rdf.yaml").write_text(
        f"""format_version: 0.5.12
type: model
name: NuMorph nucleus-marker U-Net (fold {fold})
version: 1.0.0
description: >-
  3D U-Net detecting nucleus middle-slice markers in cleared mouse-brain light-sheet microscopy.
authors:
  - name: Luis Kuhn
    github_user: luiskuhn
maintainers:
  - name: Luis Kuhn
    github_user: luiskuhn
license: MIT
documentation:
  source: README.md
  sha256: {readme_sha}
tags:
  - 3d
  - pytorch
  - unet
  - nucleus
  - segmentation
  - light-sheet-microscopy
cite:
  - text: REPLACE WITH RESOLVABLE MODEL CITATION
    url: https://github.com/luiskuhn/nuxnet-training
inputs:
  - id: input
    description: >-
      Normalized volume resampled to {", ".join(f"{v:g}" for v in TARGET_VOXEL_SPACING)} um per voxel (Z,Y,X); axes must be multiples of 4.
    axes:
      - type: batch
        size: 1
      - type: channel
        channel_names:
          - intensity
      - id: z
        type: space
        size:
          min: 4
          step: 4
      - id: y
        type: space
        size:
          min: 4
          step: 4
      - id: x
        type: space
        size:
          min: 4
          step: 4
    data:
      type: float32
      range:
        - 0.0
        - 1.0
    sample_tensor:
      source: sample-input.tif
      sha256: {hashes["sample-input.tif"]}
    test_tensor:
      source: test-input.npy
      sha256: {hashes["test-input.npy"]}
outputs:
  - id: logits
    description: >-
      Raw background and marker logits; apply softmax(dim=1). Marker is a 2D disc target.
    axes:
      - type: batch
      - type: channel
        channel_names:
          - background
          - marker
      - id: z
        type: space
        size:
          tensor_id: input
          axis_id: z
      - id: y
        type: space
        size:
          tensor_id: input
          axis_id: y
      - id: x
        type: space
        size:
          tensor_id: input
          axis_id: x
    data:
      type: float32
      range:
        - null
        - null
    sample_tensor:
      source: sample-output.tif
      sha256: {hashes["sample-output.tif"]}
    test_tensor:
      source: test-output.npy
      sha256: {hashes["test-output.npy"]}
weights:
  pytorch_state_dict:
    source: weights.pt
    sha256: {weights_sha}
    architecture:
      source: nuxnet_inference_package/model/unet_3d_models.py
      callable: UNet3D
      kwargs:
        in_channels: 1
        classes: {CLASSES}
        dropout: {DROPOUT}
      sha256: {architecture_sha}
    dependencies:
      source: environment.yml
      sha256: {environment_sha}
    pytorch_version: 2.5.1
config:
  project:
    target_voxel_spacing_zyx_um: {json.dumps(TARGET_VOXEL_SPACING)}
    patch_size_zyx: {json.dumps(list(PATCH))}
    normalization: per-volume min-max after resampling
    output: logits, softmax required
    nuclei_target: 2D single-slice disc per nucleus, median 63 voxels, not a filled body
"""
    )


def write_sums(out: Path) -> None:
    lines = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256(path)}  {path.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--repo", default=".", help="nuxnet-inference checkout to copy the architecture from")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copy(args.weights, out / "weights.pt")

    # The architecture must ship with the weights, at the path rdf.yaml names, or the
    # package cannot be reloaded by a consumer who does not have our source tree.
    architecture_source = Path(args.repo) / "nuxnet_inference_package" / "model" / "unet_3d_models.py"
    if not architecture_source.is_file():
        raise SystemExit(f"architecture not found: {architecture_source}")
    architecture_target = out / "nuxnet_inference_package" / "model" / "unet_3d_models.py"
    architecture_target.parent.mkdir(parents=True)
    shutil.copy(architecture_source, architecture_target)

    model = UNet3D(in_channels=1, classes=CLASSES, dropout=DROPOUT)
    state = torch.load(out / "weights.pt", map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=True)
    model.eval()
    print(f"loaded {len(state.get('state_dict', state))} tensors strictly from {args.weights}")

    hashes = write_tensor_files(out, model)
    write_environment(out)
    write_readme(out, args.fold)
    write_rdf(out, hashes, args.fold)
    write_sums(out)

    print(f"\npackaged {out}")
    for entry in sorted(out.rglob("*")):
        if entry.is_file():
            print(f"  {entry.relative_to(out)}  {entry.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
