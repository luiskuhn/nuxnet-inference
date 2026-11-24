"""Build the example image/mask pair that ships with nuxnet-inference.

Why a script instead of a bare binary: the pair in ``examples/numorph_c075_0007/``
has to stay honest about where it came from, and a reader of the repository should be
able to see exactly what was copied, what was recompressed, and what was stripped. It
is also how the files get regenerated if the source dataset is ever re-released.

Provenance
----------
The pair is a genuine NuMorph semantic-segmentation sample, taken from the dataset
that ``nuxnet-training`` trains on (``NUMORPH_SEM_SEG_DATASET.zip``, linked from the
training repo). It is redistributed here so that the inference tool can be tested and
scored against a ground truth without the user having to find the dataset first.

The reuse terms of that dataset are recorded as PENDING in the training repository.
That is a real constraint, not boilerplate: if the terms turn out to forbid
redistribution, this fixture must be removed and replaced with the synthetic
alternative described in ``examples/README.md``. Until then it is included for
testing, with attribution.

What this script changes
------------------------
* ``zlib`` compression. The source stores uncompressed float32, which is 12.5 MiB for
  a single 64x224x224 volume. Compression is lossless, so the voxels are identical.
* Nothing else. Values, dtype, shape, OME axes and the PhysicalSize calibration are
  preserved as they were, because the point of the fixture is to exercise the real
  parsing path -- including the spacing read that everything downstream depends on.

Usage
-----
    python scripts/build_example_data.py \
        --image /path/to/..._IMAGE_....ome.tif \
        --mask  /path/to/..._MASK_....ome.tif \
        --out   examples/numorph_c075_0007
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import tifffile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_size(ome_metadata: str | None) -> dict[str, float] | None:
    """Pull Z/Y/X physical size out of the OME-XML, in that order."""
    if not ome_metadata:
        return None
    sizes = {}
    for axis in ("Z", "Y", "X"):
        match = re.search(rf'PhysicalSize{axis}="([0-9.]+)"', ome_metadata)
        if not match:
            return None
        sizes[axis] = float(match.group(1))
    return sizes


def read_pair(image_path: Path, mask_path: Path):
    with tifffile.TiffFile(image_path) as tif:
        series = tif.series[0]
        image = series.asarray()
        image_meta = tif.ome_metadata
        image_axes = series.axes

    with tifffile.TiffFile(mask_path) as tif:
        series = tif.series[0]
        mask = series.asarray()
        mask_axes = series.axes

    if image_axes != mask_axes:
        raise SystemExit(f"axes disagree: image {image_axes}, mask {mask_axes}")
    if not set("ZYX").issubset(set(image_axes)):
        raise SystemExit(f"expected ZYX axes, got {image_axes}")
    if image.shape != mask.shape:
        raise SystemExit(f"shape mismatch: image {image.shape}, mask {mask.shape}")

    spacing = physical_size(image_meta)
    if spacing is None:
        raise SystemExit(
            "the image carries no PhysicalSize calibration; refusing to build a fixture "
            "that would not exercise the spacing path"
        )
    return image, mask, image_axes, spacing


def describe(image: np.ndarray, mask: np.ndarray, spacing: dict[str, float]) -> dict:
    """Facts about the pair that the model card and the examples README quote."""
    from scipy import ndimage as ndi

    labels, count = ndi.label(mask > 0, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())[1:] if count else np.array([], dtype=int)

    # NuMorph masks are 2D middle-Z markers, so every component should be one slice thick.
    thickness = []
    for index in range(1, count + 1):
        thickness.append(len(np.unique(np.argwhere(labels == index)[:, 0])))
    thickness = np.array(thickness) if thickness else np.array([], dtype=int)

    return {
        "image_shape_zyx": [int(v) for v in image.shape],
        "image_dtype": str(image.dtype),
        "image_value_range": [float(image.min()), float(image.max())],
        "mask_dtype": str(mask.dtype),
        "mask_unique_values": [int(v) for v in np.unique(mask)],
        "voxel_size_um_zyx": [spacing["Z"], spacing["Y"], spacing["X"]],
        "foreground_percent": round(float(mask.mean()) * 100, 4),
        "mask_components": int(count),
        "component_size_voxels": {
            "min": int(sizes.min()) if sizes.size else 0,
            "median": float(np.median(sizes)) if sizes.size else 0.0,
            "max": int(sizes.max()) if sizes.size else 0,
        },
        "component_z_thickness": {
            "min": int(thickness.min()) if thickness.size else 0,
            "median": float(np.median(thickness)) if thickness.size else 0.0,
            "max": int(thickness.max()) if thickness.size else 0,
        },
    }


def write_volume(path: Path, volume: np.ndarray, axes: str, spacing: dict[str, float]) -> None:
    metadata: dict[str, object] = {"axes": axes}
    metadata.update(
        {
            "PhysicalSizeZ": spacing["Z"],
            "PhysicalSizeY": spacing["Y"],
            "PhysicalSizeX": spacing["X"],
            "PhysicalSizeZUnit": "\u00b5m",
            "PhysicalSizeYUnit": "\u00b5m",
            "PhysicalSizeXUnit": "\u00b5m",
        }
    )
    tifffile.imwrite(path, volume, ome=True, metadata=metadata, compression="zlib")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--name",
        default=None,
        help="output basename; defaults to the source basename with the long prefix removed",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    image, mask, axes, spacing = read_pair(image_path, mask_path)

    if args.name:
        stem = args.name
    else:
        stem = image_path.name
        for suffix in (".ome.tif", ".ome.tiff", ".tif", ".tiff"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        stem = stem.replace("NUMORPH_SEM_SEG_", "").replace("_IMAGE_", "__image__")
        stem = stem.replace("__image__", "")

    image_out = out / f"{stem}.image.ome.tif"
    mask_out = out / f"{stem}.mask.ome.tif"

    write_volume(image_out, image, axes, spacing)
    write_volume(mask_out, mask, axes, spacing)

    facts = describe(image, mask, spacing)
    facts["source"] = {
        "dataset": "NuMorph semantic segmentation (NUMORPH_SEM_SEG_DATASET)",
        "original_image_name": image_path.name,
        "original_mask_name": mask_path.name,
        "transform": "zlib-compressed, voxels unchanged",
    }

    (out / "sample.json").write_text(json.dumps(facts, indent=2) + "\n")

    print(f"wrote {image_out.name}  {image_out.stat().st_size / 1024:.0f} KiB")
    print(f"wrote {mask_out.name}  {mask_out.stat().st_size / 1024:.0f} KiB")
    print()
    print(json.dumps(facts, indent=2))
    print()
    for path in (image_out, mask_out):
        print(f"sha256 {sha256(path)}  {path.name}")


if __name__ == "__main__":
    main()
