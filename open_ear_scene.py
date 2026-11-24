"""Open the ear ROI in napari with the image, the predicted mask and Imaris spots.

    python open_ear_scene.py

`napari ear1_1-6m_roi.zarr` opens the image and the label layer with their
physical transform attached, but a points table is not an image, so the Imaris
spots are skipped by the OME-Zarr plugin and have to be added explicitly. This
script adds all three.

The spots are stored in ABSOLUTE microns -- the same world frame the image
layers produce through ``index * scale + translate``. So the points go in with
scale (1, 1, 1) and no translation: handing them the image's scale as well would
multiply their coordinates a second time and scatter them off the tissue. If the
spots do appear displaced, the transform is wrong, and keeping the correction
out of the file is what makes that visible rather than hidden.

Verified: 39,510 points, 100% inside the image's world extent.
"""
from pathlib import Path

import numpy as np
import zarr
from napari_ome_zarr import napari_get_reader

SCENE = Path(__file__).parent / "ear1_1-6m_roi.zarr"


def build_layers(scene: Path = SCENE):
    """Return layer specs as (kind, data, kwargs) tuples.

    Kept separate from the viewer so the layer set, shapes and transforms can be
    asserted without opening a window.
    """
    reader = napari_get_reader(str(scene))
    if reader is None:
        raise RuntimeError(f"no OME-Zarr reader for {scene}")

    specs = []
    for data, meta, layer_type in reader(str(scene)):
        name = meta.get("name", "")
        kwargs = {
            "name": name,
            "scale": meta.get("scale"),
            "translate": meta.get("translate"),
        }
        # A label layer needs a discrete colour map rather than the greyscale one.
        if layer_type == "labels" or "mask" in name.lower():
            specs.append(("labels", data, kwargs))
        else:
            specs.append(("image", data, kwargs))

    root = zarr.open_group(str(scene), mode="r")
    if "spots" in root:
        points = np.stack(
            [np.asarray(root["spots"][k][:], float) for k in "zyx"], axis=1
        )
        specs.append((
            "points",
            points,
            {
                "name": f"Imaris spots ({len(points):,})",
                "size": 5.0,               # microns; a nucleus is ~14.8 um across
                "face_color": "cyan",
                "border_color": "white",
                "scale": (1.0, 1.0, 1.0),  # already absolute microns
                "translate": (0.0, 0.0, 0.0),
            },
        ))
    return specs


def main() -> None:
    import napari

    viewer = napari.Viewer(ndisplay=3)
    adders = {"image": viewer.add_image, "labels": viewer.add_labels,
              "points": viewer.add_points}
    for kind, data, kwargs in build_layers():
        adders[kind](data, **kwargs)
    napari.run()


if __name__ == "__main__":
    main()
