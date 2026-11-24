"""Open the ear ROI in napari: marker, predicted mask, and both point sets.

    python open_ear_scene.py

Eight layers: the five acquired channels, the predicted mask, the Imaris spots and
the model's own instance centroids. Only the marker channel is visible on open --
that is the channel the model ran on, marked ``[MODEL INPUT]`` in its name -- because
five channels added together read as a bright smear. The other four are one click
away in the layer list.

Two details this file has to get right, both of which were wrong once and looked
like a broken store rather than a broken viewer:

* The reader hands back ``contrast_limits`` and ``colormap`` from the store's `omero`
  block. They must be passed through. Rebuilding a layer with only name/scale/
  translate makes napari fall back to a min/max stretch, and since the marker spans
  0..22815 while only the lower end is real signal, the whole volume clips to white.
* A multi-channel image arrives as one ``(c, z, y, x)`` array with a *list* of names,
  so it needs ``channel_axis`` to split into one layer per channel. Without it,
  ``add_image`` raises TypeError on the list-valued name.

All layers share one world frame. The arrays carry ``index * scale + translate``; the
points are stored in ABSOLUTE microns already, so they go in with scale (1, 1, 1) and
no translation. Giving them the image's scale as well would multiply their coordinates
a second time and scatter them off the tissue.
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

        # Carry over EVERYTHING the reader parsed, not a hand-picked subset. The
        # OME-Zarr reader reads `contrast_limits` and `colormap` out of the store's
        # `omero` block; rebuilding the layer with only name/scale/translate throws
        # that away and napari then falls back to a min/max stretch. That is the same
        # saturation bug as the hardcoded 0..1 window the exporter had: the marker
        # spans 0..22815 but only the lower end is real signal, so the whole volume
        # clips to white.
        kwargs = {}
        for key in ("name", "scale", "translate", "contrast_limits", "colormap",
                    "metadata", "opacity", "visible"):
            if meta.get(key) is not None:
                kwargs[key] = meta[key]

        # A multi-channel image arrives as one (c, z, y, x) array with `name` a list
        # of per-channel names. napari needs `channel_axis` to split it into one layer
        # per channel; without it, add_image raises TypeError on the list-valued name.
        if meta.get("channel_axis") is not None:
            kwargs["channel_axis"] = meta["channel_axis"]
            kwargs.setdefault("blending", "additive")
            # Show the marker the model actually ran on; leave the others off so the
            # first frame is the segmentation's own input rather than five channels
            # of anatomy added together, which reads as a bright smear. They stay one
            # click away in the layer list.
            labels = meta.get("name") or []
            if isinstance(labels, list) and labels:
                kwargs["visible"] = ["MODEL INPUT" in str(x).upper() for x in labels]
                if not any(kwargs["visible"]):
                    kwargs["visible"][0] = True

        # Keep ONLY the finest level. This is what makes 3D usable.
        #
        # napari's rule for 2D is "coarsest level that still exceeds the canvas"
        # (layer_utils.compute_multiscale_level), so zoomed out it shows level 1 or 2
        # -- reasonable, and at fit-zoom on a 690x630 plane it already picks level 0.
        # But for 3D base.py does this unconditionally:
        #
        #     # set the data_level so that it is the lowest resolution in 3d view
        #     if self.multiscale is True:
        #         self._data_level = len(self.level_shapes) - 1
        #
        # So every 3D render uses the COARSEST level, and there is no way to override
        # it from outside: _update_draw resets it on each redraw. With 3 levels that
        # was level 2 = (54, 172, 157), 1.6% of the voxels, and upsampled back it
        # differs from native by 83% mean relative error -- the blurred look.
        #
        # Handing napari a single level makes multiscale False, so data_level is 0
        # in both 2D and 3D and every view is native. The arrays stay lazy (zarr
        # backed), so this costs nothing until a slice is read: 0.5 s / 188 MB for
        # one channel, and only the marker starts visible.
        finest = data[0] if isinstance(data, (list, tuple)) else data

        # A label layer takes a discrete label->colour mapping, not an image colormap,
        # so it keeps only the transform. Its `omero` window is 0..1, which is already
        # the right stretch for a binary mask.
        if layer_type == "labels" or "mask" in str(name).lower():
            label_kwargs = {k: v for k, v in kwargs.items()
                            if k in ("name", "scale", "translate", "opacity", "visible")}
            specs.append(("labels", finest, label_kwargs))
        else:
            specs.append(("image", finest, kwargs))

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
