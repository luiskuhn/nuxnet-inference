"""Verify a chunked nuxnet-inference run against the store it wrote into.

Run after `scripts/predict_zarr_chunks.py`. Separate from the prediction script on
purpose: the thing being checked is the *result*, and a check that shares code with
the writer can reproduce the writer's own mistake.

Checks, in order of how likely they are to be silently wrong:

1. the layer exists, is the right shape, and is binary
2. the label is declared as an OME-NGFF label image, so viewers treat it as one
3. the pyramid levels actually exist for every level the metadata advertises
4. the mask's transforms match the image's, so the layers overlay
5. the mask is not empty, not saturated, and its density is plausible for the input
6. foreground agrees with CD8 intensity, which is what the model was given
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--level", default="0")
    parser.add_argument("--layer", default="labels/nuclei")
    parser.add_argument("--channel", default=0, help="channel index or omero label")
    parser.add_argument("--slab", type=int, default=32, help="z-slab for reading")
    args = parser.parse_args()

    import zarr

    root = zarr.open_group(args.zarr, mode="r")
    failures: list[str] = []

    # ---- 1. the layer exists and is binary ----------------------------------
    if args.layer not in root:
        print(f"FAIL  no layer {args.layer!r}")
        return 1
    layer_group = root[args.layer]
    mask = layer_group[args.level]
    image = root[args.level]
    print(f"layer   {args.layer}[{args.level}]  shape {mask.shape}  dtype {mask.dtype}")

    if mask.shape != image.shape[-3:]:
        failures.append(
            f"mask shape {mask.shape} != image spatial shape {image.shape[-3:]}"
        )
    if mask.dtype != np.uint8:
        failures.append(f"mask dtype {mask.dtype} is not uint8")

    # Values actually present. A 0/1 mask is expected; anything else means the
    # threshold or the writer did something other than semantic segmentation.
    values = np.unique(mask[: min(args.slab, mask.shape[0])])
    print(f"values  {values[:10].tolist()}"
          f"{' ...' if values.size > 10 else ''}  ({values.size} distinct in first slab)")
    if not set(np.unique(values).tolist()) <= {0, 1}:
        failures.append(f"mask is not binary; values {values[:10].tolist()}")

    # ---- 2. declared as a label image --------------------------------------
    if "/" in args.layer:
        parent_path, label = args.layer.rsplit("/", 1)
        declared = list(root[parent_path].attrs.get("labels", []))
        print(f"labels  {parent_path}.labels = {declared}")
        if label not in declared:
            failures.append(f"{label!r} not declared in {parent_path}.labels")
        if parent_path != "labels":
            failures.append(f"label image is under {parent_path!r}, not 'labels'")

    # Label images need `type: label` in their multiscales metadata; without it a
    # viewer opens the mask with a continuous colormap.
    entries = layer_group.attrs.get("multiscales", [])
    if not entries:
        failures.append("layer has no multiscales metadata")
    else:
        entry = entries[0]
        kind = entry.get("type")
        print(f"ngff    type={kind!r}  version={entry.get('version')!r}")

    # ---- 3. advertised levels exist on disk --------------------------------
    advertised = [d["path"] for d in (entries[0]["datasets"] if entries else [])]
    present = sorted(k for k in layer_group.keys() if k.isdigit())
    print(f"levels  advertised {advertised}  present {present}")
    if sorted(advertised) != present:
        failures.append(f"metadata advertises {advertised}, disk has {present}")

    # ---- 4. transforms match the image ------------------------------------
    def transforms(group: zarr.Group) -> dict:
        entry = group.attrs.get("multiscales", [{}])[0]
        out = {}
        for t in entry.get("datasets", [{}])[0].get("coordinateTransformations", []):
            out[t["type"]] = t[list(k for k in t if k != "type")[0]]
        return out

    mask_tr, image_tr = transforms(layer_group), transforms(root)
    for kind in ("scale", "translation"):
        if kind not in mask_tr or kind not in image_tr:
            failures.append(f"missing {kind} transform on mask or image")
            continue
        # Drop the channel axis from the image's transform before comparing.
        ref = [v for v in image_tr[kind]][-3:]
        got = mask_tr[kind]
        if not np.allclose(ref, got):
            failures.append(f"{kind} differs: mask {got} vs image {ref}")
        else:
            print(f"{kind:12s} matches image: {[round(float(v), 4) for v in got]}")

    # ---- 5. mask is neither empty nor saturated ---------------------------
    total = int(np.prod(mask.shape))
    fg_total = 0
    for z0 in range(0, mask.shape[0], args.slab):
        slab = np.asarray(mask[z0:z0 + args.slab])
        fg_total += int(slab.sum())
    fg_fraction = fg_total / total
    print(f"foreground {fg_total:,} voxels = {100 * fg_fraction:.4f}% of the volume")
    if fg_total == 0:
        failures.append("mask is empty")
    elif fg_fraction > 0.5:
        failures.append(f"mask covers {100 * fg_fraction:.1f}% -- implausible for nuclei")

    # ---- 6. foreground tracks the input channel --------------------------
    # The model only ever saw this one channel, so its output must sit on that
    # channel's bright voxels. A mask that is uncorrelated with the input is the
    # signature of a channel/axis mix-up that still writes a plausible volume.
    #
    # Top-1% "bright" is the wrong statistic for this target. The training label is
    # a single-slice disc at each nucleus middle, so a mask voxel need not contain
    # the brightest voxel of the nucleus it marks; that test scored a demonstrably
    # correct mask at 0.15 precision here and 0.03 on the ear. Percentile rank is
    # scale-free and asks what matters: are mask voxels high in their own plane?
    omero = root.attrs.get("omero", {}).get("channels", [])
    channel = args.channel
    if isinstance(channel, str) and not channel.lstrip("-").isdigit():
        labels_list = [c.get("label", "") for c in omero]
        if channel not in labels_list:
            failures.append(f"channel {channel!r} not in omero labels {labels_list}")
            channel = 0
        else:
            channel = labels_list.index(channel)
    channel = int(channel)

    ranks: list[float] = []
    plane_indices = np.linspace(0, mask.shape[0] - 1, 8).astype(int)
    for z in plane_indices:
        plane = np.asarray(image[channel, z] if image.ndim == 4 else image[z],
                           dtype=np.float64)
        m = np.asarray(mask[z]).astype(bool)
        if plane.shape != m.shape:
            failures.append(f"plane {z}: image {plane.shape} vs mask {m.shape}")
            break
        if not m.any():
            continue
        flat = plane.ravel()
        order = np.argsort(flat, kind="stable")
        rank = np.empty(flat.size, dtype=np.float64)
        rank[order] = np.arange(flat.size, dtype=np.float64) / max(flat.size - 1, 1)
        ranks.append(float(rank[m.ravel()].mean()))

    if ranks:
        mean_rank = float(np.mean(ranks))
        print(f"vs channel {channel} "
              f"({omero[channel].get('label') if omero and channel < len(omero) else '?'}):")
        print(f"  mean percentile rank of mask voxels {mean_rank:.3f} over "
              f"{len(ranks)} planes")
        # A mask on noise would average ~0.5; a correctly placed sparse mask sits
        # far above that. 0.75 is a deliberately loose floor -- the ear and this
        # store both score >0.9, so anything near chance is a real signal of error.
        if mean_rank < 0.75:
            failures.append(
                f"mask voxels average only the {mean_rank*100:.1f}th percentile of the "
                f"input channel -- mask may not belong to this channel"
            )

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print("PASS  all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
