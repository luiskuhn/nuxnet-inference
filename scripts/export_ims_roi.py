"""Cut a subvolume from a large Imaris .ims and write it as an OME-Zarr scene.

Why this exists
---------------
An ``.ims`` here can be hundreds of gigabytes, and only a small region is needed to
test the inference tool. Copying the whole file to a machine with a GPU is wasteful,
and the native resolution is what the test has to exercise, so a downsampled export
would not answer the question.

This reads a rectangular region straight out of the HDF5 at **ResolutionLevel 0**
and writes it as OME-NGFF: image levels at the root, one ``omero`` entry per
channel so napari-ome-zarr names and colours them correctly, and optional point
annotations copied from the file's own ``Scene`` group.

The Imaris layout
-----------------
::

    DataSet/ResolutionLevel {0..6}/TimePoint 0/Channel {c}/Data   (z, y, x) uint16
    DataSetInfo/Channel {c}          Name, Color as fixed-width char arrays
    DataSetInfo/Image                ExtMin0/1/2, ExtMax0/1/2 -> physical extent
    Scene/Content/Points {i}/CoordsXYZR                            (n, 4) float32

Spacing is derived from the extent and the array shape, not guessed. Imaris stores
the extent as ASCII in fixed-width char arrays, and there is no ``PhysicalSize`` in
the OME XML for this file, so the extent is the only calibration available.

Points are already in the file's absolute microns, and the crop's translation is
computed the same way as the mask's elsewhere in this repo --
``physical = index * scale + translation`` -- so the two agree by construction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _decode(value) -> str:
    """Imaris attributes are fixed-width char arrays; join and strip them."""
    if isinstance(value, np.ndarray) and value.dtype.kind == "S":
        return b"".join(value.ravel().tolist()).decode(errors="replace").strip()
    return str(value)


def open_ims(path: Path):
    import h5py

    return h5py.File(str(path), "r")


def image_spacing(handle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxel spacing, origin, and the real image shape, in ``Z, Y, X``.

    The datasets are **zero-padded up to whole HDF5 chunks**, so their shape is not
    the image size: level 0 is stored as ``(960, 11136, 7296)`` while the image is
    ``(950, 11043, 7181)``. Dividing the physical extent by the padded shape gives
    0.9596 um instead of the true 0.975 um -- a 1.6% calibration error on every axis,
    which is exactly the kind of quiet scale error that makes a mask look slightly
    wrong without anything obviously failing.

    The declared size lives in ``DataSetInfo/Image`` as ``X``, ``Y``, ``Z`` and is
    the only correct divisor. The padding is verified to be zeros, so cropping to the
    declared size loses no data.
    """
    info = handle["DataSetInfo/Image"].attrs
    lo = np.array([float(_decode(info[f"ExtMin{i}"])) for i in range(3)])
    hi = np.array([float(_decode(info[f"ExtMax{i}"])) for i in range(3)])
    # Imaris extents are X, Y, Z; the arrays are Z, Y, X.
    lo_zyx = lo[::-1]
    hi_zyx = hi[::-1]

    declared = np.array([int(_decode(info[axis])) for axis in ("Z", "Y", "X")])
    stored = np.array([
        handle["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"].shape[i]
        for i in range(3)
    ])
    shape = np.minimum(declared, stored) if declared.size == 3 else stored
    spacing = (hi_zyx - lo_zyx) / shape
    return spacing, lo_zyx, shape


def channel_metadata(handle) -> list[dict]:
    """Name and display colour for every channel, plus a data-derived window."""
    channels = []
    for index in range(_channel_count(handle)):
        attrs = handle[f"DataSetInfo/Channel {index}"].attrs
        name = _decode(attrs.get("Name", f"Channel {index}"))
        colour = _decode(attrs.get("Color", "1.0 1.0 1.0"))
        rgb = _colour_to_hex(colour)
        window = _channel_window(handle, index)
        channels.append({"label": name, "colour": rgb, "window": window})
    return channels


def _channel_count(handle) -> int:
    level = handle["DataSet/ResolutionLevel 0/TimePoint 0"]
    return len([k for k in level.keys() if k.startswith("Channel")])


def _colour_to_hex(text: str) -> str:
    """``'1.0 0.0 0.0'`` -> ``'FF0000'``.

    Imaris stores the colour as three floats in 0..1. A channel with an all-zero
    colour would render invisible, so fall back to white rather than black.
    """
    try:
        parts = [float(v) for v in text.split()][:3]
        if len(parts) != 3 or max(parts) <= 0:
            return "FFFFFF"
        return "".join(f"{int(round(min(max(v, 0.0), 1.0) * 255)):02X}" for v in parts)
    except ValueError:
        return "FFFFFF"


def _channel_window(handle, index: int) -> tuple[float, float]:
    """1st..99.5th percentile of a strided sample of the channel.

    A window from min/max would let one hot pixel flatten the whole channel; this
    is the same percentile rule the rest of the repo uses for display.
    """
    data = handle[f"DataSet/ResolutionLevel 4/TimePoint 0/Channel {index}/Data"]
    sample = np.asarray(data[:: max(1, data.shape[0] // 16),
                            :: max(1, data.shape[1] // 64),
                            :: max(1, data.shape[2] // 64)])
    low, high = np.percentile(sample, (1.0, 99.5))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def read_region(handle, index: int, z0, z1, y0, y1, x0, x1) -> np.ndarray:
    data = handle[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {index}/Data"]
    return np.asarray(data[z0:z1, y0:y1, x0:x1])


def pyramid(volume: np.ndarray, levels: int) -> list[np.ndarray]:
    """Max-pooled 2x reduction per level, on the spatial axes of a CZYX array.

    Max rather than mean so a sparse marker does not fade below the display window
    when zoomed out. The reduce axes start at ``1``: including the channel axis
    would collapse ``(c, z, y, x)`` to ``(1, ...)`` and napari then rejects the
    pyramid as out of order.
    """
    out = [volume]
    for _ in range(1, levels):
        current = out[-1]
        shape = tuple(max(1, s // 2) for s in current.shape[1:])
        if min(shape) < 2:
            break
        reduced = current[
            :,
            : shape[0] * 2, : shape[1] * 2, : shape[2] * 2,
        ].reshape(current.shape[0], shape[0], 2, shape[1], 2, shape[2], 2)
        out.append(reduced.max(axis=(2, 4, 6)))
    return out


def read_points(handle) -> list[dict]:
    """The file's point annotations, in absolute microns, as ``Z, Y, X``.

    Imaris writes ``CoordsXYZR`` in X, Y, Z order. Everything downstream in this
    repo indexes by Z, Y, X, so the transposition happens here, once, at the
    boundary. Clipping against a Z, Y, X box while the points are still X, Y, Z
    silently keeps nothing, because the X interval never overlaps the Z one.
    """
    content = handle.get("Scene/Content")
    if content is None:
        return []
    sets = []
    for key in sorted(k for k in content.keys() if k.startswith("Points")):
        node = content[key]
        if "CoordsXYZR" not in node:
            continue
        name = _decode(node.attrs.get("Name", key))
        xyz = np.asarray(node["CoordsXYZR"][:, :3], dtype=np.float64)
        sets.append({"name": name, "coords": xyz[:, ::-1]})
    return sets


def clip_points(points: np.ndarray, lower, upper, margin=0.0) -> np.ndarray:
    """Keep points inside the physical box, with an optional margin."""
    low = np.asarray(lower) - margin
    high = np.asarray(upper) + margin
    return points[np.all((points >= low) & (points <= high), axis=1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ims", required=True)
    parser.add_argument("--out", required=True, help="output OME-Zarr store")
    parser.add_argument("--z-range", required=True, help="Z0:Z1 at ResolutionLevel 0")
    parser.add_argument("--y-range", required=True)
    parser.add_argument("--x-range", required=True)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--marker-channel", default=None,
                        help="channel name to mark [MODEL INPUT], e.g. CD8")
    parser.add_argument("--points", action="store_true",
                        help="copy the file's point annotations as points groups")
    parser.add_argument("--points-margin-um", type=float, default=0.0)
    parser.add_argument(
        "--points-shift-um", default="0,0,0",
        help=(
            "Z,Y,X offset in um added to every point before it is written. Default is "
            "no offset: the point annotations are copied exactly as the file stores "
            "them."
        ),
    )
    args = parser.parse_args()

    import zarr
    from numcodecs import Blosc

    # Reuse the exporter's store/layer writers rather than reimplementing them: they
    # carry the fixes this repo already paid for (nested chunk keys, an `omero` block
    # with one entry per channel, per-axis scale and translation).
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from export_roi_scene import axis_metadata, open_store, write_group  # noqa: F401

    handle = open_ims(Path(args.ims))
    spacing, origin, image_shape = image_spacing(handle)
    channels = channel_metadata(handle)
    print(f"spacing Z {spacing[0]:.4f}  Y {spacing[1]:.4f}  X {spacing[2]:.4f} um")
    print(f"origin  Z {origin[0]:.1f}  Y {origin[1]:.1f}  X {origin[2]:.1f} um")
    print(f"image   Z {image_shape[0]}  Y {image_shape[1]}  X {image_shape[2]} (declared)")

    z0, z1 = (int(v) for v in args.z_range.split(":"))
    y0, y1 = (int(v) for v in args.y_range.split(":"))
    x0, x1 = (int(v) for v in args.x_range.split(":"))
    # Clamp to the declared image size. The stored arrays run to whole HDF5 chunks,
    # so a range taken from the dataset shape would pad the crop with a strip of
    # zeros that is not part of the image.
    requested = (z0, z1, y0, y1, x0, x1)
    z0, z1 = max(0, z0), min(z1, int(image_shape[0]))
    y0, y1 = max(0, y0), min(y1, int(image_shape[1]))
    x0, x1 = max(0, x0), min(x1, int(image_shape[2]))
    if (z0, z1, y0, y1, x0, x1) != requested:
        print(f"clamped Z {z0}:{z1}  Y {y0}:{y1}  X {x0}:{x1} to the declared image size")
    print(f"region  Z {z0}:{z1}  Y {y0}:{y1}  X {x0}:{x1}")

    volume = np.stack([
        read_region(handle, index, z0, z1, y0, y1, x0, x1)
        for index in range(len(channels))
    ])
    print(f"read    {volume.shape} {volume.dtype} = {volume.nbytes / 1e9:.2f} GB")

    # Physical position of the crop's first voxel, so every downstream layer shares
    # one frame: physical = index * scale + translation.
    translation = (
        origin[0] + z0 * spacing[0],
        origin[1] + y0 * spacing[1],
        origin[2] + x0 * spacing[2],
    )

    if args.marker_channel:
        for channel in channels:
            if channel["label"].lower() == args.marker_channel.lower():
                channel["label"] = f"{channel['label']} [MODEL INPUT]"
                break
        else:
            raise SystemExit(
                f"--marker-channel {args.marker_channel!r} is not one of "
                f"{[c['label'] for c in channels]}"
            )

    levels = pyramid(volume, args.levels)
    level_scales = [
        (spacing[0] * (2 ** i), spacing[1] * (2 ** i), spacing[2] * (2 ** i))
        for i in range(len(levels))
    ]
    print(f"pyramid {[tuple(int(v) for v in lv.shape) for lv in levels]}")

    store = Path(args.out)
    group = open_store(store, mode="w")
    write_group(
        group, None, levels, tuple(spacing), translation, level_scales,
        "FFFFFF", "image", compressor=Blosc(cname="zstd", clevel=5, shuffle=0),
        channel_labels=[c["label"] for c in channels],
        channel_colours=[c["colour"] for c in channels],
        channel_windows=[c["window"] for c in channels],
    )

    if args.points:
        root = open_store(store, mode="a")
        lower = np.array(translation)
        upper = lower + np.array([(z1 - z0), (y1 - y0), (x1 - x0)]) * np.array(spacing)
        shift = np.array([float(v) for v in args.points_shift_um.split(",")])
        if shift.shape != (3,):
            raise SystemExit("--points-shift-um needs three comma-separated values, Z,Y,X")
        for annotation in read_points(handle):
            # Shift BEFORE clipping: the clip box is in image coordinates, and a point
            # that is outside the crop but lands inside it after correction is one the
            # viewer needs. Clipping first would silently drop it.
            kept = clip_points(annotation["coords"] + shift, lower, upper,
                               margin=args.points_margin_um)
            if len(kept) == 0:
                print(f"points  {annotation['name']}: none inside the crop")
                continue
            node = root.require_group(f"points_{annotation['name']}")
            for axis, column in zip("zyx", range(3)):
                node.create_dataset(axis, data=kept[:, column],
                                    overwrite=True, chunks=(min(len(kept), 65536),))
            print(f"points  {annotation['name']}: {len(kept):,} of "
                  f"{len(annotation['coords']):,} inside the crop"
                  + (f" (shift {tuple(shift)} um)" if shift.any() else ""))

    # Provenance. The store keeps enough of the source acquisition that it can be
    # traced back to the .ims without that file: which image it came from, when it
    # was recorded, and exactly which voxel range was cut. Written under a namespaced
    # key so it cannot collide with NGFF's own attributes.
    root = open_store(store, mode="a")
    info = handle["DataSetInfo/Image"].attrs
    root.attrs["source"] = {
        "format": "Imaris .ims (HDF5)",
        "file": Path(args.ims).name,
        "image_id": _decode(info.get("ID", "")),
        "image_name": _decode(info.get("Name", "")),
        "recording_date": _decode(info.get("RecordingDate", "")),
        "original_format": _decode(info.get("OriginalFormat", "")),
        "imaris_version": _decode(
            info.get("OriginalFormatFileIOVersion", "")
        ),
        "resolution_level": 0,
        "voxel_range_zyx": [[z0, z1], [y0, y1], [x0, x1]],
        "declared_image_shape_zyx": [int(v) for v in image_shape],
        "voxel_size_um_zyx": [round(float(v), 6) for v in spacing],
        "origin_um_zyx": [round(float(v), 6) for v in origin],
        "channels": [
            {"index": i, "name": c["label"], "colour": c["colour"]}
            for i, c in enumerate(channels)
        ],
        "notes": (
            "ResolutionLevel 0, cropped to the declared image size. The stored HDF5 "
            "arrays are zero-padded to whole chunk boundaries, so their shape is "
            "larger than the image; pixel size is extent / declared size, not "
            "extent / array shape."
        ),
    }

    total = sum(lv.nbytes for lv in levels)
    print(f"wrote   {store}  ({total / 1e9:.2f} GB of arrays)")


if __name__ == "__main__":
    main()
