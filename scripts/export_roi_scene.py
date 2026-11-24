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
                  name: str = "image", n_channels: int = 1):
    """Multiscales entry for a [z, y, x] or [c, z, y, x] group with absolute placement.

    `translation` is the physical position of voxel (0, 0, 0) of this crop within
    the specimen, i.e. the volume origin plus region_start scaled by the source
    spacing. Keeping it absolute is what lets the crop be dropped into a scene
    without a manual offset, and it means image and mask agree by construction.

    With more than one channel the axis list gains a leading channel axis and every
    scale/translation gets a leading 1.0/0.0, because NGFF requires one entry per
    axis. Getting that length wrong makes the reader reject the store.
    """
    translation = [
        float(source_translation[i] + region_start[i] * source_scale[i])
        for i in range(3)
    ]
    datasets = []
    for level, scale in enumerate(level_scales):
        if n_channels > 1:
            entry_scale = [1.0] + [float(v) for v in scale]
            entry_translation = [0.0] + translation
        else:
            entry_scale = [float(v) for v in scale]
            entry_translation = translation
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": entry_scale},
                    {"type": "translation", "translation": entry_translation},
                ],
            }
        )
    axes = [{"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}]
    if n_channels > 1:
        axes.insert(0, {"name": "c", "type": "channel"})
    return {
        "version": "0.4",
        "name": name,
        "axes": axes,
        "datasets": datasets,
        "type": "gaussian",
    }


def pyramid(volume: np.ndarray, levels: int, reducer: str) -> list[np.ndarray]:
    """Downsample by 2x per level on the spatial axes, keeping any channel axis.

    Max, not mean, for the mask: averaging a sparse binary layer drives nuclei below
    any threshold and they vanish when zoomed out, which reads as missing detections.
    For the image, mean is the honest thumbnail.

    A 4-axis input is treated as (c, z, y, x) and pooled only over z, y, x, so the
    channel count is preserved and a multi-channel store keeps one layer per channel
    at every level.
    """
    out = [volume]
    current = volume
    for _ in range(levels - 1):
        spatial = current.shape[-3:]
        if min(spatial) < 2:
            break
        # Work on a trailing (z, y, x) view so 3-axis and 4-axis inputs share the code.
        leading = current.shape[:-3]
        z, y, x = spatial
        trimmed = current[..., : z - z % 2, : y - y % 2, : x - x % 2]
        shape = leading + (z // 2, 2, y // 2, 2, x // 2, 2)
        blocks = trimmed.reshape(shape)
        # Reduce the size-2 axes only, i.e. the odd positions after any leading
        # (channel) axes. Starting at len(leading) would include the first halved
        # spatial axis and collapse the whole volume to (2, 2, 2) per level.
        axes = tuple(range(len(leading) + 1, len(shape), 2))
        if reducer == "max":
            current = blocks.max(axis=axes)
        else:
            current = blocks.mean(axis=axes).astype(volume.dtype)
        out.append(current)
    return out


def write_points(parent, name: str, local_voxels: np.ndarray, scale,
                 translation, label: str, extra: dict | None = None) -> None:
    """Write a points table, converting voxel indices to absolute microns.

    Both point layers go through here so they cannot disagree about the frame. The
    conversion is ``physical = index * scale + translation`` with ``translation`` the
    already-absolute origin of the crop, so the region offset must NOT be added again
    -- doing so shifts the layer in the specimen while leaving the arrays aligned,
    which is the failure that is hardest to see by eye.
    """
    import zarr  # noqa: F401  (kept local to match the rest of the module)

    local = np.asarray(local_voxels, dtype=np.float64)
    physical = np.stack(
        [local[:, i] * scale[i] + translation[i] for i in range(3)], axis=1
    )
    node = parent.require_group(name)
    for i, axis in enumerate(("z", "y", "x")):
        node.create_dataset(
            axis, data=physical[:, i].astype(np.float32), overwrite=True,
            dimension_separator="/",
        )
    for key, values in (extra or {}).items():
        node.create_dataset(
            key, data=np.asarray(values), overwrite=True, dimension_separator="/",
        )
    node.attrs["points_table"] = {
        "axes": ["z", "y", "x"],
        "units": "micrometer",
        "frame": "absolute specimen coordinates",
        "label": label,
        "n": int(len(local)),
    }


def channel_metadata(store: Path, series, n_channels: int) -> list[dict]:
    """Channel names and colours, preferring the store's `omero`, then the OME XML.

    Both sources are optional in practice: a bioformats2raw conversion may or may not
    write `omero` on the series group, while the original acquisition always records
    channel names in ``OME/METADATA.ome.xml``. Falling back to the XML is what keeps
    the layer names meaningful ("Nuclei", "Veins") instead of "channel 0", which would
    defeat the point of naming which channel the model saw.

    The XML also carries the channel colour, as a packed RGBA int (signed 32-bit in
    practice, so 100598527 and -15990785 both appear). Decoding it yields exactly the
    same values the `omero` block would have held -- 05FF02, FF0605, ... -- so the
    fallback keeps the acquisition's colouring instead of inventing a palette.
    Channels are returned in axis order, which is what `--channels` indexes.
    """
    import xml.etree.ElementTree as ET

    omero = series.attrs.get("omero", {}).get("channels", [])
    if omero:
        return [dict(c) for c in omero]

    base = Path(str(store).rstrip("/"))
    for candidate in (base / "OME" / "METADATA.ome.xml", base / "METADATA.ome.xml"):
        if not candidate.exists():
            continue
        try:
            root = ET.fromstring(candidate.read_text(errors="ignore"))
        except ET.ParseError:
            break
        # Tolerate the namespace variants bioformats2raw emits.
        found = []
        for element in root.iter():
            if element.tag.split("}")[-1] != "Channel":
                continue
            entry = {"label": (element.get("Name") or element.get("ID") or "").strip()}
            packed = element.get("Color")
            if packed:
                try:
                    # Packed as RGBA in a signed 32-bit int; mask to unsigned and
                    # drop the alpha byte.
                    entry["color"] = f"{(int(packed) & 0xFFFFFFFF) >> 8 & 0xFFFFFF:06X}"
                except ValueError:
                    pass
            found.append(entry)
        if found:
            return found
        break
    return [{} for _ in range(n_channels)]


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
                scale, translation, level_scales, colour, label: str,
                compressor, ngff_type: str = "gaussian",
                window: tuple[float, float] | None = None,
                channel_labels: list[str] | None = None,
                channel_colours: list[str] | None = None,
                channel_windows: list[tuple[float, float]] | None = None) -> None:
    """Write one multiscale image into `parent`, optionally under a child group.

    `name=None` writes the arrays directly into `parent`, which is what an OME-NGFF
    image at the root of a store requires: the root group itself must carry the
    multiscales metadata and the numbered array levels. A root that instead holds
    child groups with the arrays inside them is not an image to a reader, and
    napari-ome-zarr returns zero layers for it.

    For a multi-channel image the caller passes per-channel `channel_labels`,
    `channel_colours` and `channel_windows`, and the `omero` block must list one
    entry per channel or the reader names the layers wrongly. For a binary label
    image the single `colour`/`label`/`window` arguments are used instead.

    `window` is the display range a viewer applies on open. It must come from the
    data, not from a constant: this channel spans 0..22815, and a hardcoded 0..1
    window clips every value above 1 to pure white, so the image opens as a
    saturated black-and-white blob rather than an intensity image.
    """
    node = parent if name is None else parent.require_group(name)
    n_channels = levels[0].shape[0] if levels[0].ndim == 4 else 1
    dims = ["c", "z", "y", "x"] if n_channels > 1 else ["z", "y", "x"]
    node.attrs["_ARRAY_DIMENSIONS"] = dims

    for level, data in enumerate(levels):
        chunk = tuple(min(int(c), 64) for c in data.shape)
        # dimension_separator="/" so the chunk keys land nested (0/1/2), matching
        # what ome_zarr hardcodes when it opens the store. See open_store().
        array = node.create_dataset(
            str(level), data=data, chunks=chunk, compressor=compressor,
            overwrite=True, fill_value=0, dimension_separator="/",
        )
        array.attrs["_ARRAY_DIMENSIONS"] = dims

    # Build the entry as a plain dict first. Mutating a value already stored in
    # node.attrs does not persist, because zarr serialises on assignment: doing
    # node.attrs["multiscales"][0]["name"] = label silently loses the label.
    entry = axis_metadata(scale, translation, (0, 0, 0), level_scales, name=label,
                          n_channels=n_channels)
    entry["type"] = ngff_type
    node.attrs["multiscales"] = [entry]

    if channel_labels is not None:
        colours = channel_colours or ["FFFFFF"] * len(channel_labels)
        windows = channel_windows or [(0.0, 1.0)] * len(channel_labels)
        channels = []
        for i, (chan_label, chan_colour, chan_window) in enumerate(
                zip(channel_labels, colours, windows)):
            start, end = chan_window
            channels.append({
                "color": chan_colour.lstrip("#"),
                "label": chan_label,
                "active": True,
                "window": {"start": float(start), "end": float(end),
                           "min": float(start), "max": float(end)},
            })
        node.attrs["omero"] = {"channels": channels}
    else:
        start, end = window if window is not None else (0.0, 1.0)
        node.attrs["omero"] = {
            "channels": [{"color": colour.lstrip("#"), "label": label,
                          "window": {"start": float(start), "end": float(end),
                                     "min": float(start), "max": float(end)}}]
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, help="source OME-Zarr store")
    parser.add_argument("--layer", default=None, help="mask group inside the store")
    parser.add_argument("--out", required=True, help="destination .zarr scene")
    parser.add_argument("--channel", type=int, default=0,
                        help="the channel the model was run on; marked as such in the "
                             "layer name")
    parser.add_argument("--channels", default=None,
                        help="comma-separated channels to include, default just "
                             "--channel (for example '0,1,2,3,4')")
    parser.add_argument("--z-range", required=True)
    parser.add_argument("--y-range", required=True)
    parser.add_argument("--x-range", required=True)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--spots", default=None,
                        help="optional .npz of Imaris spots to add as a points table")
    parser.add_argument("--spots-key", default="Points0")
    parser.add_argument("--centroids", default=None,
                        help="optional .nuclei.tsv from the inference CLI, added as a "
                             "points layer in the same frame as the image")
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
    # Read every requested channel as one (c, z, y, x) stack, so the store holds a
    # single multiscale image with a channel axis rather than unrelated images.
    # napari-ome-zarr turns that into one layer per channel, named and coloured from
    # the `omero` block, which is what makes "which channel was the model input"
    # answerable in the layer list rather than in the README.
    channels = [int(c) for c in args.channels.split(",")] if args.channels else [args.channel]
    image = np.asarray(
        series["0"][0, channels,
                     z_range[0]:z_range[1], y_range[0]:y_range[1],
                     x_range[0]:x_range[1]]
    )
    print(f"image             {image.shape}  {image.dtype}  "
          f"range {image.min()}..{image.max()}  channels {channels}")
    if image.shape[1:] != shape:
        raise SystemExit(f"image crop {image.shape[1:]} != requested {shape}")

    # Input channel, by identity, not by position: the model was trained on one
    # marker and after this it is one of several layers.
    input_index = channels.index(args.channel)
    print(f"model input       channel {args.channel} = index {input_index} in this stack")

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

    # Per-channel display: name, colour and window all come from the source's own
    # `omero` block where it has one, so the scene shows the channels the way the
    # acquisition did. The channel the model actually saw is renamed to say so -- the
    # point of carrying the other channels is to judge the segmentation against the
    # anatomy, and that requires knowing which layer produced it.
    source_channels = channel_metadata(Path(args.zarr), series, len(channels))
    channel_labels, channel_colours, channel_windows = [], [], []
    for position, channel_index in enumerate(channels):
        meta = source_channels[channel_index] if channel_index < len(source_channels) else {}
        name = meta.get("label") or f"channel {channel_index}"
        if channel_index == args.channel:
            name = f"{name} [MODEL INPUT]"
        plane = image[position]
        # Window from the data, never a constant: a hardcoded 0..1 clips this
        # channel (0..22815) to pure white. The 99.5th percentile top matches the
        # stretch open_layers.py uses, so a few hot voxels cannot push every nucleus
        # into the dark end.
        window = (float(np.percentile(plane, 1.0)), float(np.percentile(plane, 99.5)))
        colour = str(meta.get("color", "FFFFFF"))
        channel_labels.append(name)
        channel_colours.append(colour)
        channel_windows.append(window)
        marker = " <- model input" if channel_index == args.channel else ""
        print(f"  channel {channel_index}: {name:34s} {colour}  "
              f"window {window[0]:.0f}..{window[1]:.0f}{marker}")

    write_group(group, None, image_levels, scale, translation, level_scales,
                "FFFFFF", channel_labels[0], image_compressor,
                channel_labels=channel_labels,
                channel_colours=channel_colours,
                channel_windows=channel_windows)

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

    # ---- optional point layers ----------------------------------------------
    # Two independent detectors, so they are separate groups rather than merged:
    # merging would hide exactly the disagreement the comparison exists to show.
    group = open_store(out, mode="a")
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
        write_points(group, "spots", local, scale, translation,
                     label="Imaris nuclei spots")
        print(f"spots             {len(local):,} points written (absolute um)")

    if args.centroids:
        # The CLI's own instance centroids, in ROI-local voxel coordinates. Same
        # grid and same frame as the image, because the CLI returns the mask on the
        # input grid and localises on that mask -- so no resampling is applied here.
        table = np.genfromtxt(args.centroids, delimiter="\t", names=True,
                              dtype=None, encoding="utf-8")
        local = np.stack([table["z"], table["y"], table["x"]], axis=1).astype(np.float64)
        # A centroid is a mean of voxel indices, so it can fall a hair outside the
        # crop when a nucleus touches the boundary; keep only points inside.
        keep = (
            (local[:, 0] >= 0) & (local[:, 0] < shape[0])
            & (local[:, 1] >= 0) & (local[:, 1] < shape[1])
            & (local[:, 2] >= 0) & (local[:, 2] < shape[2])
        )
        inside = local[keep]
        write_points(group, "centroids", inside, scale, translation,
                     label="model nuclei centroids",
                     extra={"size_voxels": table["size_voxels"][keep]})
        print(f"centroids         {len(inside):,} of {len(local):,} points written "
              f"({len(local) - len(inside):,} outside the crop)")

    # Root-level annotations only. The multiscales entry now belongs to the root
    # itself (written above), so anything added here must not add or replace it.
    group.attrs["roi"] = {
        "z": list(z_range), "y": list(y_range), "x": list(x_range),
        "source": str(args.zarr),
        "channel": args.channel,
    }
    layers = ["image", "nuclei"]
    if args.spots:
        layers.append("spots")
    if args.centroids:
        layers.append("centroids")
    group.attrs["scene"] = {
        "name": "ear1_1-6m ROI",
        "description": "image and nuclei mask in absolute specimen coordinates",
        "layers": layers,
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
