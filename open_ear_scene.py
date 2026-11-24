"""Open the ear ROI in napari with the image, the mask and both point sets.

    python open_ear_scene.py

Four layers: the marker channel, the predicted mask, the Imaris spots and the
model's own instance centroids. The OME-Zarr plugin loads the first two with their
physical transform attached, but a points table is not an image, so the points are
added explicitly.

All four share one world frame. The arrays carry ``index * scale + translate``; the
points are stored in ABSOLUTE microns already, so they go in with scale (1, 1, 1)
and no translation. Giving them the image's scale as well would multiply their
coordinates a second time and scatter them off the tissue. If a point layer does
appear displaced, the transform is wrong; keeping the correction out of the file is
what makes that visible instead of hidden.
"""
from pathlib import Path

import napari
import numpy as np
import zarr
from napari_ome_zarr import napari_get_reader

SCENE = Path(__file__).parent.parent / "data" / "ear1_1-6m_roi.zarr"


def build_layers(scene: Path = SCENE):
    """Return napari layer specs as (kind, data, kwargs) tuples.

    Separated from the viewer so it can be checked without opening a window:
    a headless test can assert the layer count, shapes and transforms.
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
        # A label layer needs a discrete colour map, not the image's greyscale.
        if layer_type == "labels" or "mask" in name.lower():
            specs.append(("labels", data, kwargs))
        else:
            specs.append(("image", data, kwargs))

    root = zarr.open_group(str(scene), mode="r")

    # Two point sources, kept as separate layers. They are independent detectors, and
    # merging them would hide exactly the agreement the scene exists to show.
    point_layers = (
        ("spots", "Imaris spots", "cyan"),
        ("centroids", "model centroids", "yellow"),
    )
    for group, label, colour in point_layers:
        if group not in root:
            continue
        points = np.stack(
            [np.asarray(root[group][k][:], float) for k in "zyx"], axis=1
        )
        specs.append((
            "points",
            points,
            {
                "name": f"{label} ({len(points):,})",
                "size": 4.0,             # microns; a nucleus is ~14.8 um across
                "face_color": colour,
                "border_color": "black",
                "opacity": 0.6,
                # Already absolute microns: giving these the image's scale too would
                # multiply their coordinates a second time.
                "scale": (1.0, 1.0, 1.0),
                "translate": (0.0, 0.0, 0.0),
            },
        ))
    return specs


def main() -> None:
    viewer = napari.Viewer(ndisplay=3)
    adders = {"image": viewer.add_image, "labels": viewer.add_labels,
              "points": viewer.add_points}
    for kind, data, kwargs in build_layers():
        adders[kind](data, **kwargs)
    napari.run()


if __name__ == "__main__":
    main()
