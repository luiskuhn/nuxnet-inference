"""Build a local OME-Zarr scene: the ear ROI image plus the nuclei mask, side by side.

Why a separate export rather than just copying the layer
-------------------------------------------------------
The production run wrote ``nuclei_mask_roi`` into the *source* store, which is a
bioformats2raw layout: the image lives inside a series group ``0/`` and the mask
sits next to it as a sibling. That is correct and what the pipeline should do, but
it is awkward to open interactively:

* the image series group carries a 5-axis ``[t, c, z, y, x]`` multiscales entry,
  while the mask carries a 3-axis ``[z, y, x]`` one, and a viewer has to be told
  how to relate them;
* the source store is 4.8 GB and lives on a remote machine.

So this writes a *new, small* store containing

    ear1_1-6m_roi.zarr/
      image/     [z, y, x] nucleus channel, ROI crop only, uint16, multiscale
      nuclei/    [z, y, x] binary mask,                 uint8,  multiscale
      spots/     Imaris nucleus detections as a points table (optional)

Both image and mask use the same 3-axis convention and the same absolute physical
transform, so a viewer overlays them without any per-layer fiddling. The transform
is absolute (not region-relative): ``translation`` is where the ROI starts in the
original specimen, so the crop stays in its true position and coordinates match the
source store.

Sizes: the ROI is 216 x 690 x 630 at 5.005 / 3.5455 / 3.5454 um. That is 94 M voxels,
about 188 MB as uint16 -- small enough to copy to a laptop and open in napari.

    python scripts/export_roi_scene.py \\
        --zarr /zarr/ear1_1-6m.lite.zarr --layer nuclei_mask_roi \\
        --out /zarr/ear1_1-6m_roi.zarr \\
        --z-range 48:264 --y-range 1480:2170 --x-range 400:1030
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def span(text: str) -> tuple[int, int]:
    start, _, stop = text.partition(":")
    return int(start), int(stop)


def axis_metadata(source_scale, source_translation, region_start, level_scales,
                  name: str = "image"):
    """Multiscales entry for a 3-axis [z, y, x] group with absolute placement.

    `translation` is the physical position of voxel (0, 0, 0) of this crop within
    the specimen, i.e. the volume origin plus region_start scaled by the source
    spacing. Keeping it absolute is what lets the crop be dropped into a scene
    without a manual offset, and it means image and mask agree by construction.
    """
    translation = [
        float(source_translation[i] + region_start[i] * source_scale[i])
        for i in range(3)
    ]
    datasets = []
    for level, scale in enumerate(level_scales):
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [float(v) for v in scale]},
                    {"type": "translation", "translation": translation},
                ],
            }
        )
    return {
        "version": "0.4",
        "name": name,
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": datasets,
        "type": "gaussian",
    }


def pyramid(volume: np.ndarray, levels: int, reducer: str) -> list[np.ndarray]:
    """Downsample by max-pooling 2x per level on the last three axes.

    Max, not mean, for the mask: averaging a sparse binary layer drives nuclei below
    any threshold and they vanish when zoomed out, which reads as missing detections.
    For the image, mean is the honest thumbnail.
    """
    out = [volume]
    current = volume
    for _ in range(levels - 1):
        if current.shape[0] < 2 or current.shape[1] < 2 or current.shape[2] < 2:
            break
        if reducer == "max":
            z, y, x = current.shape[:3]
            trimmed = current[: z - z % 2, : y - y % 2, : x - x % 2]
            current = trimmed.reshape(z // 2, 2, y // 2, 2, x // 2, 2).max(axis=(1, 3, 5))
        else:
            z, y, x = current.shape[:3]
            trimmed = current[: z - z % 2, : y - y % 2, : x - x % 2]
            current = trimmed.reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5))
            current = current.astype(volume.dtype)
        out.append(current)
    return out


def open_store(out: Path, mode: str = "a"):
    """Open the scene on a store that writes NESTED chunk keys, ``0/1/2/3``.

    This is not a cosmetic choice, and getting it wrong is silent.

    OME-NGFF 0.2 moved chunk keys from flat (``0.0.0``) to nested (``0/0/0``), and
    ``ome_zarr`` -- which napari-ome-zarr uses to read every store -- hardcodes
    ``dimension_separator="/"`` when it opens one (``FormatV02.init_store``). If the
    chunks on disk are flat but the reader looks them up nested, every lookup misses
    and zarr answers with ``fill_value``. For these arrays that is 0, so a viewer
    shows a correct-looking, correctly-shaped, completely blank image. No exception,
    no warning, no truncation: the read simply succeeds and reports zeros.

    ``zarr.open_group(path)`` does not infer the separator from the ``.zarray`` on
    read in this version, so the store has to carry it.
    """
    from zarr.storage import FSStore

    import zarr

    return zarr.open_group(
        store=FSStore(str(out), mode=mode, dimension_separator="/", auto_mkdir=True),
        mode=mode,
    )


def verify_scene(out: Path, expected_mask_foreground: int | None) -> None:
    """Read the scene back the way napari does and fail if it is blank.

    Checking shapes is what let a broken store look finished: the shape and dtype
    were right and every voxel was zero. So this asserts on *content*, through the
    same reader a viewer uses.
    """
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    nodes = [n for n in Reader(parse_url(str(out)))() if n.data]
    if len(nodes) < 2:
        raise SystemExit(f"verify: expected image and mask, reader returned {len(nodes)}")

    image = np.asarray(nodes[0].data[0][:])
    if image.max() == 0:
        raise SystemExit(
            "verify: the image reads as all zeros. The store probably uses flat "
            "chunk keys (0.0.0) while the reader expects nested ones (0/0/0)."
        )
    print(f"verify image      {image.shape}  range {image.min()}..{image.max()}")

    mask = np.asarray(nodes[1].data[0][:])
    foreground = int((mask > 0).sum())
    if foreground == 0:
        raise SystemExit("verify: the mask reads as all zeros")
    if expected_mask_foreground is not None and foreground != expected_mask_foreground:
        raise SystemExit(
            f"verify: mask foreground {foreground:,} != expected "
            f"{expected_mask_foreground:,}; the pyramid or crop is wrong"
        )
    print(f"verify mask       {mask.shape}  foreground {foreground:,}")

    under = image[mask > 0].mean()
    off = image[mask == 0].mean()
    ratio = under / off if off else float("inf")
    print(f"verify intensity  under mask {under:.0f} vs elsewhere {off:.0f} = {ratio:.2f}x")
    if ratio <= 1.0:
        raise SystemExit("verify: the mask does not sit on brighter pixels than background")


def write_group(parent, name: str | None, levels: list[np.ndarray],
                scale, translation, level_scales, colour: str, label: str,
                compressor, ngff_type: str = "gaussian",
                window: tuple[float, float] | None = None) -> None:
    """Write one multiscale image into `parent`, optionally under a child group.

    `name=None` writes the arrays directly into `parent`, which is what an OME-NGFF
    image at the root of a store requires: the root group itself must carry the
    multiscales metadata and the numbered array levels. A root that instead holds
    child groups with the arrays inside them is not an image to a reader, and
    napari-ome-zarr returns zero layers for it.

    `window` is the display range a viewer applies on open. It must come from the
    data, not from a constant: this channel spans 0..22815, and a hardcoded 0..1
    window clips every value above 1 to pure white, so the image opens as a
    saturated black-and-white blob rather than a intensity image.
    """
    node = parent if name is None else parent.require_group(name)
    node.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]

    for level, data in enumerate(levels):
        chunk = tuple(min(int(c), 64) for c in data.shape)
        # dimension_separator="/" so the chunk keys land nested (0/1/2), matching
        # what ome_zarr hardcodes when it opens the store. See open_store().
        array = node.create_dataset(
            str(level), data=data, chunks=chunk, compressor=compressor,
            overwrite=True, fill_value=0, dimension_separator="/",
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]

    # Build the entry as a plain dict first. Mutating a value already stored in
    # node.attrs does not persist, because zarr serialises on assignment: doing
    # node.attrs["multiscales"][0]["name"] = label silently loses the label.
    entry = axis_metadata(scale, translation, (0, 0, 0), level_scales, name=label)
    entry["type"] = ngff_type
    node.attrs["multiscales"] = [entry]
    start, end = window if window is not None else (0.0, 1.0)
    node.attrs["omero"] = {
        "channels": [{"color": colour, "label": label,
                      "window": {"start": float(start), "end": float(end),
                                 "min": float(start), "max": float(end)}}]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, help="source OME-Zarr store")
    parser.add_argument("--layer", default=None, help="mask group inside the store")
    parser.add_argument("--out", required=True, help="destination .zarr scene")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--z-range", required=True)
    parser.add_argument("--y-range", required=True)
    parser.add_argument("--x-range", required=True)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--spots", default=None,
                        help="optional .npz of Imaris spots to add as a points table")
    parser.add_argument("--spots-key", default="Points0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import zarr
    from numcodecs import Blosc

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    source = zarr.open_group(args.zarr, mode="r")
    series = source["0"]

    # Read the metadata of the image series: it is the authority on axes and spacing.
    attributes = dict(series.attrs.get("multiscales", [{}])[0])
    axes = [a["name"] for a in attributes["axes"]]
    transforms = attributes["datasets"][0]["coordinateTransformations"]
    full_scale = list(transforms[0]["scale"])
    full_translation = [0.0] * (len(full_scale) - 3)
    for transform in transforms:
        if transform["type"] == "translation":
            full_translation = list(transform["translation"])

    def pick(values):
        """Select z, y, x from a 5-axis list by axis name, never by position."""
        return [values[axes.index(a)] for a in ("z", "y", "x")]

    scale = pick(full_scale)
    source_translation = pick(full_translation)
    print(f"source axes       {axes}")
    print(f"source scale      Z {scale[0]:.4f}  Y {scale[1]:.4f}  X {scale[2]:.4f} um")
    print(f"source translation Z {source_translation[0]:.2f}  "
          f"Y {source_translation[1]:.2f}  X {source_translation[2]:.2f} um")

    z_range, y_range, x_range = span(args.z_range), span(args.y_range), span(args.x_range)
    region = (z_range[0], y_range[0], x_range[0])
    shape = (z_range[1] - z_range[0], y_range[1] - y_range[0], x_range[1] - x_range[0])

    # Absolute physical origin of voxel (0,0,0) of this crop. Computed once, here,
    # because it is the transform BOTH layers must carry: if image and mask compute
    # it independently and disagree by one axis, they overlay in the array but not in
    # the specimen. axis_metadata then receives a zero region start and simply writes
    # this value through.
    translation = [
        float(source_translation[i] + region[i] * scale[i]) for i in range(3)
    ]
    print(f"crop origin       Z {translation[0]:.2f}  Y {translation[1]:.2f}  "
          f"X {translation[2]:.2f} um  (absolute)")
    print(f"region            Z {z_range[0]}:{z_range[1]}  Y {y_range[0]}:{y_range[1]}  "
          f"X {x_range[0]}:{x_range[1]}  = {shape}")

    # ---- image ---------------------------------------------------------------
    image = np.asarray(
        series["0"][0, args.channel,
                     z_range[0]:z_range[1], y_range[0]:y_range[1],
                     x_range[0]:x_range[1]]
    )
    print(f"image             {image.shape}  {image.dtype}  "
          f"range {image.min()}..{image.max()}")
    if image.shape != shape:
        raise SystemExit(f"image crop {image.shape} != requested {shape}")

    # ---- mask ----------------------------------------------------------------
    if args.layer:
        mask = np.asarray(source[args.layer]["0"][:])
        print(f"mask              {mask.shape}  {mask.dtype}  "
              f"foreground {int(mask.sum()):,}")
        if mask.shape != shape:
            raise SystemExit(f"mask {mask.shape} != region {shape}")
    else:
        mask = None

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    image_compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    group = open_store(out, mode="a")

    # ---- image at the ROOT ---------------------------------------------------
    # OME-NGFF: the root group of an image store holds the multiscales metadata and
    # the numbered array levels directly. Putting them in a child "image" group makes
    # the store a plain container of groups, which napari-ome-zarr skips entirely.
    print("writing image levels...")
    image_levels = pyramid(image, args.levels, reducer="mean")
    level_scales = [[s * (2 ** level) for s in scale]
                    for level in range(len(image_levels))]

    # Display window from the data. A hardcoded 0..1 clips this channel (0..22815)
    # to pure white on open. The 99.5th percentile as the top is the same stretch
    # open_layers.py uses for the normalised display: a few hot voxels would
    # otherwise push every nucleus into the dark end.
    image_window = (float(np.percentile(image, 1.0)),
                    float(np.percentile(image, 99.5)))
    print(f"image window      {image_window[0]:.0f}..{image_window[1]:.0f} "
          f"(1st..99.5th percentile)")
    write_group(group, None, image_levels, scale, translation, level_scales,
                "FFFFFF", "nuclei (SYTOX)", image_compressor,
                window=image_window)

    # ---- mask as a label image ----------------------------------------------
    # Nested under labels/, not beside the image. That is the OME-NGFF place for a
    # segmentation of an image, and it is what tells a reader to overlay rather than
    # open a second, unrelated image. A binary layer's window is 0..1 by definition.
    if mask is not None:
        print("writing mask levels...")
        labels = group.require_group("labels")
        labels.attrs["labels"] = ["nuclei"]
        mask_levels = pyramid(mask.astype(np.uint8), args.levels, reducer="max")
        write_group(labels, "nuclei", mask_levels, scale, translation, level_scales,
                    "FF00FF", "nuclei mask", compressor, ngff_type="label",
                    window=(0.0, 1.0))

    # ---- optional spots ------------------------------------------------------
    if args.spots:
        data = np.load(args.spots, allow_pickle=True)
        px = np.stack([
            data[f"{args.spots_key}__z_px"],
            data[f"{args.spots_key}__y_px"],
            data[f"{args.spots_key}__x_px"],
        ], axis=1)
        keep = (
            (px[:, 0] >= z_range[0]) & (px[:, 0] < z_range[1])
            & (px[:, 1] >= y_range[0]) & (px[:, 1] < y_range[1])
            & (px[:, 2] >= x_range[0]) & (px[:, 2] < x_range[1])
        )
        local = px[keep] - np.array(region)
        # Absolute physical coordinates, so the points land in the same frame as
        # the image. `translation` is already the absolute origin of the crop, so
        # the region offset must NOT be added again here.
        physical = np.stack([
            local[:, 0] * scale[0] + translation[0],
            local[:, 1] * scale[1] + translation[1],
            local[:, 2] * scale[2] + translation[2],
        ], axis=1)
        group = open_store(out, mode="a")
        node = group.require_group("spots")
        node.create_dataset("z", data=physical[:, 0].astype(np.float32), overwrite=True)
        node.create_dataset("y", data=physical[:, 1].astype(np.float32), overwrite=True)
        node.create_dataset("x", data=physical[:, 2].astype(np.float32), overwrite=True)
        node.attrs["points_table"] = {
            "axes": ["z", "y", "x"],
            "units": "micrometer",
            "frame": "absolute specimen coordinates",
            "label": "Imaris nuclei spots",
            "n": int(len(local)),
        }
        print(f"spots             {len(local):,} points written (absolute µm)")

    # Root-level annotations only. The multiscales entry now belongs to the root
    # itself (written above), so anything added here must not add or replace it.
    group.attrs["roi"] = {
        "z": list(z_range), "y": list(y_range), "x": list(x_range),
        "source": str(args.zarr),
        "channel": args.channel,
    }
    group.attrs["scene"] = {
        "name": "ear1_1-6m ROI",
        "description": "image and nuclei mask in absolute specimen coordinates",
        "layers": ["image", "nuclei"] + (["spots"] if args.spots else []),
    }

    print()
    print(f"wrote {out}")
    for child in sorted(p.name for p in out.iterdir()):
        print(f"  {child}")

    # Read it back through the reader a viewer uses, and fail loudly if blank.
    print()
    verify_scene(out, int(mask.sum()) if mask is not None else None)


if __name__ == "__main__":
    main()
