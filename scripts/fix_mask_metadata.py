"""Repair the mask layer's OME-NGFF metadata after a chunked run.

The chunked writer on solidfox was copied before two metadata fixes landed, so the
layer it produced is a correct *array* with wrong *attributes*:

* ``type`` is ``gaussian``; a discrete label image must declare ``label`` or a
  viewer applies a continuous colormap and each nucleus renders as a different
  shade of one ramp.
* ``name`` is ``labels/nuclei``; a viewer's layer list should show ``nuclei``.

Nothing here touches voxel data. It rewrites attributes only, which is why it can
safely run against a finished store.
"""

import json
import sys

import numpy as np
import zarr


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/w/data/te24_1-66m_roi.zarr"
    root = zarr.open_group(path, mode="a")

    print("ROOT multiscales:")
    print(json.dumps(root.attrs.get("multiscales"), indent=1))

    layer = root["labels/nuclei"]
    image_ct = root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    # The image is (c,z,y,x) and its transform carries four entries; the mask is
    # (z,y,x). Copy the last three only. Taking the whole list would give the mask
    # a transform of the wrong length, which is exactly the kind of metadata error
    # that looks fine in one viewer and misplaces the layer in another.
    image_scale = next(t["scale"] for t in image_ct if t["type"] == "scale")[-3:]
    image_translate = next(t["translation"] for t in image_ct
                           if t["type"] == "translation")[-3:]

    entries = layer.attrs["multiscales"]
    entry = entries[0]
    datasets = entry["datasets"]

    # Keep exactly the levels that exist on disk, and force every one of them to
    # carry the image's own scale and translation so the layers overlay.
    present = sorted(k for k in layer.keys() if k.isdigit())
    rebuilt = []
    for level in present:
        scale = [float(s) * (2 ** int(level)) for s in image_scale]
        rebuilt.append({
            "path": level,
            "coordinateTransformations": [
                {"type": "scale", "scale": scale},
                {"type": "translation", "translation": [float(t) for t in image_translate]},
            ],
        })

    layer.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "nuclei",
        "axes": entry.get("axes", [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]),
        "datasets": rebuilt,
        "type": "label",
        "metadata": entry.get("metadata", {}),
    }]
    layer.attrs["omero"] = {"channels": [{
        "label": "nuclei",
        "color": "FF00FF",
        "window": {"start": 0, "end": 1, "min": 0, "max": 1},
    }]}

    # The label image must be declared by its parent group.
    root["labels"].attrs["labels"] = ["nuclei"]

    print("\nPATCHED layer multiscales:")
    print(json.dumps(layer.attrs["multiscales"], indent=1))
    print("\nlabels group attrs:", dict(root["labels"].attrs))

    # Confirm the arrays themselves still line up with the image.
    mask = layer[present[0]]
    image = root[present[0]]
    print(f"\nmask {mask.shape} {mask.dtype}   image {image.shape}")
    assert tuple(mask.shape) == tuple(image.shape[-3:]), "spatial shape mismatch"
    print("spatial shapes match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
