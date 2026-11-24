"""Open the ear ROI in napari: marker, predicted mask, and both point sets.

    python open_ear_scene.py                 # 4 layers, the default
    python open_ear_scene.py --all-channels   # 8 layers, every acquired channel

The default is the four things the segmentation is actually about: the marker channel
the model ran on, the predicted mask, the Imaris spots and the model's own instance
centroids. That is the view for asking whether the two detectors agree.

``--all-channels`` adds the other four acquired channels, hidden, so the anatomy is
one click away without being in the way: five channels blended additively read as a
bright smear, which hides the nuclei the mask is drawn over.

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


def _take_first_level(data, keep, channel_axis):
    """Select `keep` channels from every level of a multiscale, channel-axis array.

    Indexing with a plain integer drops the axis rather than leaving a size-1
    dimension, which matters because a leftover channel axis would make a single
    channel a 4D array and shift the scale/translate napari pairs with it. Indexing
    a dask array also stays lazy (measured 3.5 ms), so the trim costs nothing until a
    slice is displayed.
    """
    index = [slice(None)] * data[0].ndim
    index[channel_axis] = keep[0] if len(keep) == 1 else list(keep)
    index = tuple(index)
    return [level[index] for level in data]


def build_layers(scene: Path = SCENE, all_channels: bool = False):
    """Return napari layer specs as (kind, data, kwargs) tuples.

    `all_channels=False` (the default) keeps only the marker channel, giving the four
    layers the segmentation is about: marker, mask, spots, centroids. `all_channels=True`
    adds the remaining acquired channels as hidden layers.

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
        channel_axis = meta.get("channel_axis")
        if channel_axis is not None:
            labels = list(meta.get("name") or [])
            # Pick the marker by name, never by position: the channel order is the
            # acquisition's, and the label is what marks which one the model saw.
            marked = [i for i, x in enumerate(labels) if "MODEL INPUT" in str(x).upper()]
            keep = list(range(len(labels))) if all_channels else marked[:1]

            if len(keep) != len(labels):
                # Trim the stack and every per-channel metadata list together. napari
                # matches these positionally, so slicing only the array would attach
                # the marker's window and colour to the wrong channel.
                data = _take_first_level(data, keep, channel_axis)
                for key in ("name", "contrast_limits", "colormap", "visible"):
                    value = kwargs.get(key)
                    if isinstance(value, list) and len(value) == len(labels):
                        kwargs[key] = [value[i] for i in keep]

            if isinstance(kwargs.get("name"), list) and len(kwargs["name"]) > 1:
                kwargs["channel_axis"] = channel_axis
                kwargs.setdefault("blending", "additive")
                # Show the marker; leave the rest off so the first frame is the
                # segmentation's own input rather than five channels of anatomy added
                # together. They stay one click away in the layer list.
                kwargs["visible"] = ["MODEL INPUT" in str(x).upper()
                                     for x in kwargs["name"]]
                if not any(kwargs["visible"]):
                    kwargs["visible"][0] = True
            else:
                # A single remaining channel needs no split, so drop the axis
                # machinery entirely. Every per-channel list must be unwrapped too
                # -- napari rejects a sequence-valued colormap without channel_axis.
                kwargs.pop("channel_axis", None)
                kwargs.pop("visible", None)
                for key in ("name", "contrast_limits", "colormap"):
                    value = kwargs.get(key)
                    if isinstance(value, list) and len(value) == 1:
                        kwargs[key] = value[0]

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
        # one channel.
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Open the ear ROI scene: marker, mask, Imaris spots, model "
                    "centroids (4 layers). Add --all-channels for the other four "
                    "acquired channels."
    )
    parser.add_argument("--all-channels", action="store_true",
                        help="also load the four non-marker channels, hidden; "
                             "default is marker + mask + spots + centroids")
    parser.add_argument("--2d", dest="two_d", action="store_true",
                        help="open in 2D instead of 3D")
    args = parser.parse_args()

    viewer = napari.Viewer(ndisplay=2 if args.two_d else 3)
    adders = {"image": viewer.add_image, "labels": viewer.add_labels,
              "points": viewer.add_points}
    for kind, data, kwargs in build_layers(all_channels=args.all_channels):
        adders[kind](data, **kwargs)
    napari.run()


if __name__ == "__main__":
    main()
