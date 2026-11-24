"""Run the inference CLI over an OME-Zarr image, chunk by chunk, and stitch the
result back into the same store as a new OME-NGFF layer.

Why chunking is not optional here
---------------------------------
The CLI resamples whatever volume it is given onto the training grid
(3.0/1.0/1.0 um). For an image acquired at a coarser sampling that is an
*up*-sampling. ``ear1_1-6m`` is 5.005 / 3.545 / 3.545 um, so resampling expands
it 1.67 x 3.55 x 3.55 = **21x** in voxel count. One full channel is
1353 x 3162 x 2422 uint16 (about 20 GB on disk) and becomes roughly 82 GB of
float32 before tiling even starts. It does not fit, and no amount of swap makes
that a good idea.

So: read a window, write it as an OME-TIFF carrying its true physical spacing,
let the CLI do what it does, and paste the mask back.

Three separate things called "overlap", and none of them is the others
---------------------------------------------------------------------
* ``--halo`` here gives each chunk *context* beyond the region it is responsible
  for. A nucleus straddling a boundary is then seen whole by at least one chunk
  instead of being predicted as a crescent. Needed because the model is
  convolutional: its receptive field extends past the patch it scores. The halo
  is discarded after prediction; only the chunk's core is pasted. At the edge of
  the requested region the halo is clamped, never zero-padded -- inventing black
  would create an edge the model would react to.
* ``--overlap`` passed to the CLI is the sliding-window overlap *inside* a chunk.
  It is not a chunking parameter and must not be confused with the halo.
* The chunk grid itself uses no overlap: chunks are disjoint and each owns its
  core exclusively, so no voxel is written twice and no seam blending is needed.

Why one CLI call per z-slab rather than one per chunk
----------------------------------------------------
A subprocess per chunk means loading the checkpoint thousands of times, and it
serialises reads, resampling and GPU work. The CLI already accepts a *directory*
of OME-TIFFs and loops over them with the model resident, so the chunks of one
z-slab go through in a single call. That also gives natural resume granularity:
a slab is either done or it is not.

Normalization must be shared, not per chunk
-------------------------------------------
The trainer normalized each training volume by its own min/max. Applied per
chunk that becomes a per-chunk range, and identical tissue gets different
contrast depending on which chunk it fell into. Measured on this ear, chunking a
256x256x32 region and normalizing each tile independently changed ~1% of voxels
and dropped Dice between tiled and untiled results to **0.458**. So one range is
derived from the whole requested region and passed to every chunk via
``--normalization-range``.

Where the stitched layer goes
-----------------------------
The mask is written as a new group *inside the same store*, with the same
physical scale as the source **and a translation offset** so a viewer places it
over the region it came from rather than at the origin. In OME-NGFF the
transform is

    physical = index * scale + translation

so a region starting at source index ``z0`` needs translation
``t_source + z0 * scale``. Without that term the layer looks correct in
isolation and misregistered in every viewer, which is the worst kind of wrong.
A 2x and 4x max-pooled pyramid is added because a 1.35 x 3.16 x 2.42 k mask is
unpleasant to scrub through in napari at full resolution only.

Usage
-----
    # full ear, into the ear's own store
    python scripts/predict_zarr_chunks.py \
        --zarr    /zarr/ear1_1-6m.lite.zarr \
        --channel 0 \
        --model   /models/fold1.pt \
        --layer-name nuclei_mask

    # a small region, to check the plumbing without waiting for the whole ear
    python scripts/predict_zarr_chunks.py ... \
        --z-range 500:532 --y-range 1024:1280 --x-range 1792:2048 \
        --chunk 32,128,128

    # plan and cost estimate only
    python scripts/predict_zarr_chunks.py ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# The CLI's own post-processing, imported rather than reimplemented, so there is
# exactly one instance-clustering implementation in the codebase.
from nuxnet_inference_package.cli_pred import (
    extract_nuclei_instances,
    write_instances_tsv,
)


def receptive_field_voxels() -> int:
    """Receptive field of UNet3D in voxels of the training grid.

    Computed from the layer table rather than guessed, and asserted against the
    architecture so the halo default cannot silently drift away from it.

    Rule: ``r = r + (k - 1) * j`` then ``j = j * stride``, where each nearest
    upsample halves the jump rather than changing the field.
    """
    layers = [
        (3, 1), (3, 1),                    # inc: 2 x conv3d
        (2, 2), (3, 1), (3, 1),            # down1: 2x2 stride-2 conv, then 2 x conv3d
        (2, 2), (3, 1), (3, 1),            # down2: same
        (3, 1), (3, 1),                    # mid: 2 x conv3d
        (1, 0.5), (3, 1), (3, 1),          # up1: nearest x2, then 2 x conv3d
        (1, 0.5), (3, 1), (3, 1),          # up2: same
        (1, 1),                            # outc: 1x1x1
    ]
    field, jump = 1.0, 1.0
    for kernel, stride in layers:
        field += (kernel - 1) * jump
        jump *= stride
    return int(round(field))


def halo_for_receptive_field(source_um, target_um) -> tuple[int, int, int]:
    """Smallest integer halo, per axis, that covers half the receptive field.

    Half the field, not the whole field: the voxel at the edge of a chunk's core
    needs complete context on *each* side, so half the receptive field either way
    completes the window. Rounded up, because half a voxel of missing context is
    precisely the case that clips a nucleus.

    This exists because the hand-picked default was wrong. The receptive field is
    60 training-grid voxels, i.e. 180 um in Z, so half of it is 90 um, which at
    5.005 um per source voxel is 18 voxels. The default halo of 8 covered only
    40 um, so any nucleus within 18 source voxels of a chunk boundary in Z was
    predicted without full context. In XY the default of 32 was generous but the
    required figure is 9, so the Z axis was the only real failure.
    """
    field = receptive_field_voxels()
    half_um = field / 2.0 * np.asarray(target_um, dtype=float)
    return tuple(int(np.ceil(v)) for v in half_um / np.asarray(source_um, dtype=float))


# ---------------------------------------------------------------------------
# zarr access
# ---------------------------------------------------------------------------
def open_level(store: str, level: str = "0"):
    """Open one multiscale level of an OME-Zarr -> (array, multiscales, group)."""
    import zarr

    root = zarr.open_group(store, mode="r")

    # bioformats2raw layout: groups "0", "1", ... each carrying a multiscales attr.
    #
    # Membership alone is not enough to distinguish the two layouts. A store whose
    # multiscales sits on the ROOT group has an *array* called "0", and
    # ``"0" in root`` is equally true there, so the test used to take this branch
    # and then index into that array looking for a child "0" -- which fails with
    # "invalid 'fields' argument". Require the entry to be a group that actually
    # declares multiscales before treating the store as bioformats2raw-shaped.
    candidate = root[level] if level in root else None
    if hasattr(candidate, "attrs") and not hasattr(candidate, "shape") \
            and candidate.attrs.get("multiscales"):
        metadata = dict(candidate.attrs["multiscales"][0])
        if "0" in candidate:
            return candidate["0"], metadata, candidate

    # Single-level layout: multiscales sits on the root group itself.
    metadata = dict(root.attrs.get("multiscales", [{}])[0])
    if not metadata:
        raise SystemExit(f"no multiscales metadata found in {store}")
    # Merge the sibling `omero` block in. Channel names live there, not inside
    # multiscales, and `--channel CD8` resolves against `metadata["omero"]`. Copy
    # rather than alias so callers cannot accidentally mutate store attributes.
    if root.attrs.get("omero"):
        metadata["omero"] = dict(root.attrs["omero"])
    return root["0"], metadata, root


def axis_names(metadata: dict) -> list[str]:
    return [axis["name"] for axis in metadata["axes"]]


def spacing_by_axis(metadata: dict) -> dict[str, float]:
    """Per-axis scale in um, keyed by name.

    Read by name, never by position: the order depends on what wrote the file,
    and ``scale[-3:]`` is only ZYX when the axes happen to end in ZYX.
    """
    transform = metadata["datasets"][0]["coordinateTransformations"][0]
    if transform.get("type") != "scale":
        raise SystemExit("expected a scale transformation as the first transform")
    names = axis_names(metadata)
    scale = transform["scale"]
    if len(scale) != len(names):
        raise SystemExit(f"scale has {len(scale)} entries for {len(names)} axes")
    return {name: float(value) for name, value in zip(names, scale)}


def translation_by_axis(metadata: dict) -> dict[str, float]:
    """Per-axis translation in um, defaulting to 0 where the file has none."""
    transforms = metadata["datasets"][0]["coordinateTransformations"]
    names = axis_names(metadata)
    for transform in transforms:
        if transform.get("type") == "translation":
            values = transform["translation"]
            return {name: float(value) for name, value in zip(names, values)}
    return {name: 0.0 for name in names}


def resolve_channel(metadata: dict, channel) -> int:
    """Accept a channel index or an omero label such as 'Nuclei'."""
    if isinstance(channel, int):
        return channel
    text = str(channel)
    if text.lstrip("-").isdigit():
        return int(text)
    omero = (metadata.get("omero") or {}).get("channels", [])
    for index, entry in enumerate(omero):
        if str(entry.get("label", "")).lower() == text.lower():
            return index
    labels = [entry.get("label") for entry in omero] or "none declared"
    raise SystemExit(f"channel {channel!r} not found; omero labels are {labels}")


def read_volume(array, axes, channel_index, z_range, y_range, x_range,
                stride=(1, 1, 1)) -> np.ndarray:
    """Read a (z, y, x) window, collapsing time and selecting one channel."""
    index: list[object] = []
    for axis in axes:
        name = axis.lower()
        if name == "t":
            index.append(0)
        elif name == "c":
            index.append(channel_index)
        elif name == "z":
            index.append(slice(z_range[0], z_range[1], stride[0]))
        elif name == "y":
            index.append(slice(y_range[0], y_range[1], stride[1]))
        elif name == "x":
            index.append(slice(x_range[0], x_range[1], stride[2]))
        else:
            raise SystemExit(f"unsupported axis {axis!r} in {axes}")
    return np.asarray(array[tuple(index)])


# ---------------------------------------------------------------------------
# argument helpers
# ---------------------------------------------------------------------------
def parse_range(text: str | None, limit: int, label: str) -> tuple[int, int]:
    if not text:
        return 0, limit
    if ":" in text:
        start, _, stop = text.partition(":")
        start = int(start) if start else 0
        stop = int(stop) if stop else limit
    else:
        start, stop = 0, int(text)
    if not 0 <= start < stop <= limit:
        raise SystemExit(f"--{label}-range {text} is outside 0:{limit}")
    return start, stop


def parse_triplet(text: str, allow_zero: bool = False) -> tuple[int, int, int]:
    parts = [int(p) for p in text.split(",")]
    low = 0 if allow_zero else 1
    if len(parts) != 3 or any(p < low for p in parts):
        what = "non-negative" if allow_zero else "positive"
        raise SystemExit(f"expected three {what} comma-separated integers, got {text!r}")
    return parts[0], parts[1], parts[2]


def chunk_starts(start: int, stop: int, size: int) -> list[int]:
    return list(range(start, stop, size))


def extended(core_start: int, core_stop: int, halo: int,
             region_start: int, region_stop: int) -> tuple[int, int]:
    """Grow by the halo, clamped to the requested region."""
    return max(region_start, core_start - halo), min(region_stop, core_stop + halo)


# ---------------------------------------------------------------------------
# occupancy: do not ask the model to segment nothing
# ---------------------------------------------------------------------------
def chunk_occupancy(array, axes, channel_index, z0, z1, y0, y1, x0, x1,
                    chunk, cache_path: Path) -> np.ndarray:
    """Boolean grid: does a chunk contain any non-zero voxel?

    A light-sheet ear is 7% tissue in a mostly-empty box; on a 32x256x256 grid
    fewer than a fifth of the chunks hold anything. Feeding the rest to the
    network is pure waste, and it is not harmless waste: a correctly-normalized
    all-zero tile lies outside anything the model saw in training.
    """
    if cache_path.exists():
        cached = np.load(cache_path)
        expected = (len(chunk_starts(z0, z1, chunk[0])),
                    len(chunk_starts(y0, y1, chunk[1])),
                    len(chunk_starts(x0, x1, chunk[2])))
        if cached.shape == expected:
            return cached.astype(bool)

    nz = len(chunk_starts(z0, z1, chunk[0]))
    ny = len(chunk_starts(y0, y1, chunk[1]))
    nx = len(chunk_starts(x0, x1, chunk[2]))
    occupied = np.zeros((nz, ny, nx), dtype=bool)

    print(f"occupancy   scanning {nz}x{ny}x{nx} chunks for signal ...", flush=True)
    for iz, cz in enumerate(chunk_starts(z0, z1, chunk[0])):
        # Read one z-chunk at a time: the plane stack is the smallest unit that
        # still lets zarr decompress each chunk exactly once per pass.
        slab = read_volume(array, axes, channel_index,
                           (cz, min(cz + chunk[0], z1)), (y0, y1), (x0, x1))
        any_z = slab.any(axis=0)
        for iy, cy in enumerate(chunk_starts(y0, y1, chunk[1])):
            band = any_z[cy - y0:min(cy + chunk[1], y1) - y0]
            for ix, cx in enumerate(chunk_starts(x0, x1, chunk[2])):
                occupied[iz, iy, ix] = bool(
                    band[:, cx - x0:min(cx + chunk[2], x1) - x0].any()
                )
        del slab, any_z
    np.save(cache_path, occupied)
    return occupied


# ---------------------------------------------------------------------------
# the CLI call
# ---------------------------------------------------------------------------
def cli_command(args, chunk_dir: Path, mask_dir: Path) -> list[str]:
    """Build one CLI invocation over a directory of chunk TIFFs."""
    base = [args.cli] if args.cli else [args.python, "-m",
                                        "nuxnet_inference_package.cli_pred"]
    command = base + [
        "predict",
        "--input", str(chunk_dir),
        "--output", str(mask_dir),
        "--model", str(args.model),
        "--classes", str(args.classes),
        "--in-channels", "1",
        "--dropout-rate", str(args.dropout_rate),
        "--target-voxel-spacing", args.target_voxel_spacing,
        "--patch-size", args.patch_size,
        "--overlap", str(args.overlap),
        "--threshold", str(args.threshold),
        "--normalization-range",
        f"{args.normalization_range[0]!r},{args.normalization_range[1]!r}",
    ]
    # The CLI declares --cuda/--no-cuda with default False, so GPU use is opt-in.
    # Forgetting it silently runs on CPU; measured, that was 425 s per chunk
    # against ~12 s on a T4. Make the intent explicit either way.
    command.append("--no-cuda" if args.cpu else "--cuda")
    return command


def write_chunk_tiff(path: Path, volume: np.ndarray, spacing) -> None:
    import tifffile

    tifffile.imwrite(
        path, np.ascontiguousarray(volume), ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": float(spacing[0]),
            "PhysicalSizeY": float(spacing[1]),
            "PhysicalSizeX": float(spacing[2]),
            "PhysicalSizeZUnit": "\u00b5m",
            "PhysicalSizeYUnit": "\u00b5m",
            "PhysicalSizeXUnit": "\u00b5m",
        },
    )


def read_mask(path: Path) -> np.ndarray:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        volume = tif.series[0].asarray()
    if volume.ndim > 3:
        volume = np.squeeze(volume)
    return (volume > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# writing the layer
# ---------------------------------------------------------------------------
def mask_axes() -> list[dict]:
    return [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]


def write_multiscales(group, name: str, scale_zyx, translation_zyx,
                      levels: int) -> None:
    """Declare exactly the levels that exist on disk.

    ``levels`` must be the number of datasets actually present. Listing a level
    that has not been written yet makes the store unreadable by ngff-validating
    viewers, so the attribute is rewritten after the pyramid is built.

    ``type`` is ``label``, not ``gaussian``: this is a discrete label image, and
    the ngff spec uses that field to tell a viewer to apply a label colormap
    rather than a continuous one. ``name`` is the leaf, so a nested layer such as
    ``labels/nuclei`` is called "nuclei" in a viewer's layer list rather than
    "labels/nuclei".
    """
    leaf = name.rsplit("/", 1)[-1]
    datasets = []
    for level in range(levels):
        lscale = [float(s) * (2 ** level) for s in scale_zyx]
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {"type": "scale", "scale": lscale},
                {"type": "translation", "translation": [float(t) for t in translation_zyx]},
            ],
        })
    group.attrs["multiscales"] = [{
        "version": "0.4",
        "name": leaf,
        "axes": mask_axes(),
        "datasets": datasets,
        "type": "label",
        "metadata": {"method": "nuxnet-inference sliding-window, thresholded"},
    }]
    # A binary mask with a sane display window and a colour that reads on grey.
    group.attrs["omero"] = {
        "channels": [{
            "label": leaf,
            "color": "FF00FF",
            "window": {"start": 0, "end": 1, "min": 0, "max": 1},
        }],
    }


def ensure_mask_group(store: str, name: str, shape, scale_zyx, translation_zyx,
                      chunk, fresh: bool):
    """Open or create the mask group's level 0 array inside the store.

    ``name`` may be a nested path such as ``labels/nuclei``. OME-NGFF puts label
    images under a ``labels`` group with a sibling ``labels`` attribute, and
    napari-ome-zarr only offers them as addable layers when that convention is
    followed. A nested path is also deliberately NOT inside the image series
    group: ``<store>/labels/<name>`` is a self-contained multiscale image in its
    own right, so it opens as a separate layer instead of being mistaken for a
    level of the intensity image.
    """
    import zarr

    root = zarr.open_group(store, mode="a")
    group = root.require_group(name)
    if fresh and "0" in group:
        del group["0"]
    if "0" not in group:
        group.create_dataset(
            "0", shape=shape, dtype="u1",
            chunks=tuple(min(c, s) for c, s in zip(chunk, shape)),
            compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=0),
        )
    array = group["0"]
    if tuple(array.shape) != tuple(shape):
        raise SystemExit(
            f"existing layer {name!r} has shape {array.shape}, this run needs {shape}; "
            f"rerun with --fresh to replace it"
        )
    # Declare the label so readers treat it as an instance/semantic label image
    # rather than another intensity channel. Without this the mask opens with a
    # continuous colormap and every nucleus looks like a different shade.
    if "/" in name:
        parent_path, label = name.rsplit("/", 1)
        parent = root[parent_path]
        declared = list(parent.attrs.get("labels", []))
        if label not in declared:
            declared.append(label)
            parent.attrs["labels"] = declared
    # Level 0 only for now; the pyramid builder adds the rest and then restates
    # the attribute so it never advertises a level that is not on disk.
    write_multiscales(group, name, scale_zyx, translation_zyx, levels=1)
    return group, array


def build_pyramid(store: str, name: str, shape, scale_zyx, translation_zyx,
                  chunk, levels: int, slab: int = 32) -> None:
    """Max-pool the mask into coarser levels for fast viewing.

    Max, not mean: a mask is binary and a nucleus must not fade to 0.25 grey
    because it occupied a quarter of a 2x cell. Averaging a label map is how
    small nuclei disappear from the overview.
    """
    import zarr

    root = zarr.open_group(store, mode="a")
    group = root[name]
    for level in range(1, levels):
        factor = 2 ** level
        lshape = tuple(max(1, -(-s // factor)) for s in shape)
        if str(level) in group:
            del group[str(level)]
        out = group.create_dataset(
            str(level), shape=lshape, dtype="u1",
            chunks=tuple(min(c, s) for c, s in zip(chunk, lshape)),
            compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=0),
        )
        # Build each level from the one above it, two voxels at a time. Pooling
        # from level 0 directly would read the full-resolution mask once per
        # level; chaining keeps every level a single cheap pass.
        src = group[str(level - 1)]
        for z in range(0, lshape[0], slab):
            z_stop = min(z + slab, lshape[0])
            # One extra slice on each side of the source so the 2x block that
            # straddles the boundary is complete.
            s0, s1 = z * 2, min(z_stop * 2, src.shape[0])
            block = np.asarray(src[s0:s1])
            if block.shape[0] % 2:
                block = np.concatenate(
                    [block, np.zeros((1, *block.shape[1:]), dtype=block.dtype)]
                )
            if block.shape[1] % 2:
                block = np.concatenate(
                    [block, np.zeros((block.shape[0], 1, block.shape[2]), dtype=block.dtype)],
                    axis=1,
                )
            if block.shape[2] % 2:
                block = np.concatenate(
                    [block, np.zeros((block.shape[0], block.shape[1], 1), dtype=block.dtype)],
                    axis=2,
                )
            pooled = block.reshape(
                block.shape[0] // 2, 2,
                block.shape[1] // 2, 2,
                block.shape[2] // 2, 2,
            ).max(axis=(1, 3, 5))
            out[z:z_stop] = pooled[: z_stop - z, : lshape[1], : lshape[2]]
            del block, pooled
        print(f"  level {level}: {lshape}", flush=True)
    write_multiscales(group, name, scale_zyx, translation_zyx, levels=levels)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunked inference over an OME-Zarr using the nuxnet-pred CLI."
    )
    parser.add_argument("--zarr", required=True,
                        help="OME-Zarr store; the mask layer is written into it")
    parser.add_argument("--level", default="0", help="multiscale level (default 0)")
    parser.add_argument("--channel", default="0",
                        help="channel index or omero label, e.g. 0 or Nuclei")
    parser.add_argument("--model", required=True, help="checkpoint .pt")
    parser.add_argument("--layer-name", default="nuclei_mask",
                        help="new group written inside --zarr")
    parser.add_argument("--z-range", default=None)
    parser.add_argument("--y-range", default=None)
    parser.add_argument("--x-range", default=None)
    parser.add_argument("--chunk", default="32,128,128", help="chunk Z,Y,X")
    parser.add_argument("--halo", default=None,
                        help="context Z,Y,X grown around each chunk, then discarded. "
                             "Default: derived from the network's receptive field")
    parser.add_argument("--target-voxel-spacing", default="3.0,1.0,1.0")
    parser.add_argument("--patch-size", default="32,128,128")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="sliding-window overlap INSIDE a chunk")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--normalization-range", default=None, metavar="LOW,HIGH",
                        help="min/max shared by every chunk; default derives it "
                             "from the whole region")
    parser.add_argument("--normalization-percentiles", default="0,100", metavar="LOW,HIGH")
    parser.add_argument("--levels", type=int, default=3,
                        help="multiscale levels for the mask layer (1 = none)")
    parser.add_argument("--skip-empty", dest="skip_empty", action="store_true",
                        default=True, help="do not send all-background chunks to the CLI")
    parser.add_argument("--no-skip-empty", dest="skip_empty", action="store_false")
    parser.add_argument("--fresh", action="store_true",
                        help="recompute from scratch instead of resuming")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and a cost estimate, then stop")
    parser.add_argument("--max-slab-chunks", type=int, default=0,
                        help="cap chunks per CLI call (0 = all in the slab)")
    parser.add_argument("--cpu", action="store_true", help="force CPU in the CLI")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter used to run the CLI module")
    parser.add_argument("--cli", default=None,
                        help="path to an installed nuxnet-pred executable")
    parser.add_argument("--nuclei-tsv", default=None,
                        help="optional TSV of centroids from the finished mask")
    parser.add_argument("--nuclei-label", type=int, default=1)
    parser.add_argument("--neighbor-radius", type=float, default=2.0)
    parser.add_argument("--work", default=None,
                        help="directory for chunk TIFFs and the occupancy cache")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    array, metadata, _ = open_level(args.zarr, args.level)
    axes = axis_names(metadata)
    spacing = spacing_by_axis(metadata)
    translation = translation_by_axis(metadata)
    channel_index = resolve_channel(metadata, args.channel)

    shape = {axis.lower(): size for axis, size in zip(axes, array.shape)}
    for required in ("z", "y", "x"):
        if required not in shape:
            raise SystemExit(f"axes {axes} do not include {required}")

    z0, z1 = parse_range(args.z_range, shape["z"], "z")
    y0, y1 = parse_range(args.y_range, shape["y"], "y")
    x0, x1 = parse_range(args.x_range, shape["x"], "x")

    source = np.array([spacing["z"], spacing["y"], spacing["x"]], dtype=float)
    target = np.array([float(v) for v in args.target_voxel_spacing.split(",")],
                      dtype=float)
    region = np.array([z1 - z0, y1 - y0, x1 - x0], dtype=float)
    resampled = np.round(region * source / target).astype(int)
    gigabytes = float(np.prod(resampled)) * 4 / 1e9

    scale_zyx = (spacing["z"], spacing["y"], spacing["x"])
    translation_zyx = (
        translation["z"] + z0 * spacing["z"],
        translation["y"] + y0 * spacing["y"],
        translation["x"] + x0 * spacing["x"],
    )

    chunk = parse_triplet(args.chunk)
    if args.halo is not None:
        halo = parse_triplet(args.halo, allow_zero=True)
    else:
        # Derived, not guessed. The hand-picked 8,32,32 under-covered Z badly:
        # the network's receptive field is 60 training-grid voxels (180 um in Z),
        # so half of it is 90 um = 18 source voxels, not 8.
        halo = halo_for_receptive_field(source, target)
        print(f"halo        {halo} (derived: half of a "
              f"{receptive_field_voxels()}-voxel receptive field, converted per axis)")

    print(f"source      {args.zarr}  level {args.level}  axes {axes}")
    print(f"channel     {args.channel} -> index {channel_index}")
    print(f"spacing     Z {source[0]:.4f}  Y {source[1]:.4f}  X {source[2]:.4f} um")
    print(f"region      Z {z0}:{z1}  Y {y0}:{y1}  X {x0}:{x1}"
          f"  = {tuple(region.astype(int))}")
    print(f"on-grid     {tuple(resampled)} voxels = {gigabytes:.2f} GB as float32"
          f" if loaded whole")
    print(f"chunk       {chunk}  halo {halo}  "
          f"-> {tuple(c + 2 * h for c, h in zip(chunk, halo))} read per chunk")

    work = Path(args.work) if args.work else Path(
        tempfile.mkdtemp(prefix="nuxnet-chunks-")
    )
    work.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # One normalization range for the whole region.
    #
    # The trainer normalized each training volume by its own min/max. Applied to
    # chunks that becomes a per-chunk range, and identical tissue gets different
    # contrast depending on which chunk it fell into. Derived here on a strided
    # subsample: min/max converge long before a full pass, and a full pass is the
    # thing this script exists to avoid.
    # ------------------------------------------------------------------
    if args.normalization_range:
        low, high = (float(v) for v in args.normalization_range.split(","))
        print(f"normalize   shared range {low:g}..{high:g} (given)")
    else:
        p_low, p_high = (float(v) for v in args.normalization_percentiles.split(","))
        sample = read_volume(array, axes, channel_index,
                             (z0, z1), (y0, y1), (x0, x1), stride=(2, 4, 4))
        if p_low == 0.0 and p_high == 100.0:
            low, high = float(sample.min()), float(sample.max())
        else:
            low, high = (float(v) for v in np.percentile(sample, [p_low, p_high]))
        del sample
        print(f"normalize   shared range {low:g}..{high:g} "
              f"(one range for every chunk, not per chunk)")
    if high <= low:
        raise SystemExit("normalization range is degenerate; the region has no signal")
    args.normalization_range = (low, high)

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    zs = chunk_starts(z0, z1, chunk[0])
    ys = chunk_starts(y0, y1, chunk[1])
    xs = chunk_starts(x0, x1, chunk[2])

    occupied = None
    if args.skip_empty:
        cache = work / f"occupancy_{z0}_{z1}_{y0}_{y1}_{x0}_{x1}_{chunk[0]}_{chunk[1]}_{chunk[2]}.npy"
        occupied = chunk_occupancy(array, axes, channel_index,
                                   z0, z1, y0, y1, x0, x1, chunk, cache)
        n_total = int(occupied.size)
        n_occ = int(occupied.sum())
        print(f"occupancy   {n_occ}/{n_total} chunks contain signal "
              f"({100 * n_occ / n_total:.1f}%); the rest are left as background")

    # Estimated GPU work: every chunk costs its halo-padded on-grid tile count.
    from nuxnet_inference_package.preprocessing import _window_starts

    pad = np.array([c + 2 * h for c, h in zip(chunk, halo)], dtype=float)
    padded_grid = np.round(pad * source / target).astype(int)
    patch = [int(v) for v in args.patch_size.split(",")]
    step = [max(1, int(round(p * (1 - args.overlap)))) for p in patch]
    tiles = int(np.prod([
        len(_window_starts(d, p, s))
        for d, p, s in zip(padded_grid, patch, step)
    ]))
    n_runs = int(occupied.sum()) if occupied is not None else int(np.prod(
        [len(zs), len(ys), len(xs)]))
    per_chunk_s = tiles * 0.08 if not args.cpu else tiles * 0.6
    estimate = n_runs * per_chunk_s
    print(f"cost        {tiles} tiles/chunk  x {n_runs} chunks  "
          f"~{estimate / 3600:.1f} h "
          f"({'CPU' if args.cpu else 'GPU'}, from measured per-tile rate)")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    # ------------------------------------------------------------------
    # Create the layer, then fill it one z-slab at a time
    # ------------------------------------------------------------------
    layer_shape = (z1 - z0, y1 - y0, x1 - x0)
    group, layer = ensure_mask_group(args.zarr, args.layer_name, layer_shape,
                                     scale_zyx, translation_zyx, chunk,
                                     fresh=args.fresh)
    progress_path = Path(args.zarr) / args.layer_name / "progress.json"
    done_slabs: set[int] = set()
    if args.fresh:
        progress_path.unlink(missing_ok=True)
    elif progress_path.exists():
        done_slabs = set(json.loads(progress_path.read_text()).get("done_slabs", []))
        if done_slabs:
            print(f"resume      {len(done_slabs)}/{len(zs)} z-slabs already done")

    def save_progress() -> None:
        progress_path.write_text(json.dumps({
            "done_slabs": sorted(done_slabs),
            "shape": list(layer_shape),
            "chunk": list(chunk),
            "halo": list(halo),
            "threshold": args.threshold,
            "normalization_range": list(args.normalization_range),
        }, indent=2) + "\n")

    started = time.time()
    processed = 0
    total_core = int(occupied.sum()) if occupied is not None else int(np.prod(
        [len(zs), len(ys), len(xs)]))

    try:
        for iz, cz in enumerate(zs):
            if iz in done_slabs:
                continue
            cz_stop = min(cz + chunk[0], z1)

            # Which (iy, ix) in this slab have signal?
            if occupied is not None:
                cells = [(iy, ix) for iy, cy in enumerate(ys)
                         for ix, cx in enumerate(xs) if occupied[iz, iy, ix]]
            else:
                cells = [(iy, ix) for iy in range(len(ys)) for ix in range(len(xs))]
            if not cells:
                layer[cz - z0:cz_stop - z0] = np.zeros(
                    (cz_stop - cz, y1 - y0, x1 - x0), dtype=np.uint8
                )
                done_slabs.add(iz)
                save_progress()
                print(f"slab {iz + 1}/{len(zs)}  z {cz}:{cz_stop}  empty, written as 0")
                continue

            slab_mask = np.zeros((cz_stop - cz, y1 - y0, x1 - x0), dtype=np.uint8)

            # Process this slab in batches so one CLI call stays bounded.
            batch_size = args.max_slab_chunks or len(cells)
            for batch_start in range(0, len(cells), batch_size):
                batch = cells[batch_start:batch_start + batch_size]
                chunk_dir = work / f"in_{iz:04d}_{batch_start:04d}"
                mask_dir = work / f"out_{iz:04d}_{batch_start:04d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                mask_dir.mkdir(parents=True, exist_ok=True)

                boxes = []
                for iy, ix in batch:
                    cy, cx = ys[iy], xs[ix]
                    cy_stop, cx_stop = min(cy + chunk[1], y1), min(cx + chunk[2], x1)
                    ez = extended(cz, cz_stop, halo[0], z0, z1)
                    ey = extended(cy, cy_stop, halo[1], y0, y1)
                    ex = extended(cx, cx_stop, halo[2], x0, x1)
                    volume = read_volume(array, axes, channel_index, ez, ey, ex)
                    name = f"c_{iz:04d}_{iy:04d}_{ix:04d}.ome.tif"
                    write_chunk_tiff(chunk_dir / name, volume, source)
                    boxes.append((iy, ix, cy, cy_stop, cx, cx_stop, ez, ey, ex))
                    del volume

                completed = subprocess.run(
                    cli_command(args, chunk_dir, mask_dir),
                    capture_output=True, text=True,
                )
                if completed.returncode != 0:
                    tail = "\n".join((completed.stderr or completed.stdout
                                      ).strip().splitlines()[-15:])
                    raise SystemExit(
                        f"CLI failed for slab {iz} batch at chunk {batch_start}:\n{tail}"
                    )

                for iy, ix, cy, cy_stop, cx, cx_stop, ez, ey, ex in boxes:
                    produced = mask_dir / f"c_{iz:04d}_{iy:04d}_{ix:04d}.ome.tiff"
                    if not produced.exists():
                        raise SystemExit(f"CLI wrote no mask for {produced.name}")
                    predicted = read_mask(produced)
                    # Keep only the core; the halo has served its purpose.
                    core_mask = predicted[
                        cz - ez[0]:cz_stop - ez[0],
                        cy - ey[0]:cy_stop - ey[0],
                        cx - ex[0]:cx_stop - ex[0],
                    ]
                    expected = (cz_stop - cz, cy_stop - cy, cx_stop - cx)
                    if core_mask.shape != expected:
                        raise SystemExit(
                            f"chunk {produced.name}: mask core {core_mask.shape} "
                            f"!= expected {expected}"
                        )
                    slab_mask[:, cy - y0:cy_stop - y0,
                              cx - x0:cx_stop - x0] = core_mask
                    processed += 1
                    del predicted, core_mask

                if not args.keep_temp:
                    shutil.rmtree(chunk_dir, ignore_errors=True)
                    shutil.rmtree(mask_dir, ignore_errors=True)

            layer[cz - z0:cz_stop - z0] = slab_mask
            del slab_mask
            done_slabs.add(iz)
            save_progress()

            elapsed = time.time() - started
            rate = processed / max(elapsed, 1e-6)
            remaining = max(0, total_core - processed)
            eta = remaining / rate if rate > 0 else 0.0
            print(f"slab {iz + 1}/{len(zs)}  z {cz}:{cz_stop}  "
                  f"{len(cells)} chunks  "
                  f"{processed}/{total_core} total  "
                  f"{elapsed / 60:.1f} min elapsed  "
                  f"ETA {eta / 3600:.1f} h", flush=True)
    finally:
        if not args.keep_temp and args.work is None:
            shutil.rmtree(work, ignore_errors=True)

    if args.levels > 1:
        print("pyramid")
        build_pyramid(args.zarr, args.layer_name, layer_shape, scale_zyx,
                      translation_zyx, chunk, args.levels)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    layer = group["0"]
    print(f"\nlayer       {args.zarr}/{args.layer_name}")
    print(f"shape       {layer.shape}  dtype {layer.dtype}  levels {args.levels}")
    print(f"transform   scale {tuple(round(v, 4) for v in scale_zyx)} um  "
          f"translation {tuple(round(v, 2) for v in translation_zyx)} um")

    if args.nuclei_tsv:
        # Read the layer back in z-slabs: a full 10 GB read just to count
        # centroids is avoidable, but the unchanged post-processing function
        # takes one array, so the mask is materialised once here.
        full = np.asarray(layer[:])
        instances = extract_nuclei_instances(
            prediction=full, nuclei_label=args.nuclei_label,
            radius=args.neighbor_radius,
        )
        table = write_instances_tsv(instances=instances,
                                    output_prefix=Path(args.nuclei_tsv))
        print(f"nuclei      {table}  {len(instances)} rows")
        print("note        centroids are voxels of the SOURCE grid, offset so "
              "index 0 is the region origin")


if __name__ == "__main__":
    main()
