"""Open just the input channel and the predicted mask, with the channel normalised.

    python open_layers.py

Two layers, both in absolute specimen coordinates so they overlay with no offset:

  nuclei (input channel)   uint16 -> float32, percentile-normalised to 0-1 for display
  nuclei mask              binary 0/1, magenta

Normalisation is a 1-99.5 percentile stretch, not min-max. Light-sheet data has a
handful of very bright voxels (this channel reaches 22,815) that would push every
nucleus into near-black under min-max; the percentile stretch is what makes them
visible. It changes the display only -- the stored layer stays uint16 and every
measurement in the report was taken on the unnormalised values.
"""
from pathlib import Path

import numpy as np
import zarr

SCENE = Path(__file__).parent.parent / "data" / "ear1_1-6m_roi.zarr"


def percentile_normalise(volume: np.ndarray, low: float = 1.0,
                         high: float = 99.5) -> np.ndarray:
    """Stretch to 0-1 on the [low, high] percentile range, clipped.

    Percentiles are taken over the whole volume rather than per slice, so a slice
    with little tissue does not get boosted to match a dense one -- the point of a
    shared window is that brightness means the same thing everywhere.
    """
    lo, hi = np.percentile(volume, (low, high))
    if hi <= lo:
        return np.zeros(volume.shape, dtype=np.float32)
    return np.clip((volume.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def model_input_index(root) -> int:
    """Which channel of the store is the segmentation model's input.

    By name, not position: the layer is labelled ``[MODEL INPUT]`` in the store's
    `omero` block precisely so this does not depend on channel order, and a store
    where no channel is marked is an error rather than a guess.
    """
    channels = root.attrs.get("omero", {}).get("channels", [])
    marked = [i for i, c in enumerate(channels)
              if "MODEL INPUT" in str(c.get("label", "")).upper()]
    if not marked:
        raise RuntimeError(
            "no channel labelled as the model input; expected one omero channel "
            "with '[MODEL INPUT]' in its label"
        )
    return marked[0]


def build_layers(scene: Path = SCENE):
    """Return (kind, data, kwargs) specs. Separate from the viewer so it is testable."""
    root = zarr.open_group(str(scene), mode="r")
    entry = root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    scale = tuple(float(v) for v in entry[0]["scale"])
    translate = tuple(float(v) for v in entry[1]["translation"])
    # A channel axis adds a leading 1.0 scale and 0.0 translation; the spatial
    # transform is the trailing three either way.
    scale, translate = scale[-3:], translate[-3:]

    volume = np.asarray(root["0"][:])
    # Only the model's own input channel, not all five: this helper exists to judge
    # the segmentation against the marker it was actually computed from.
    if volume.ndim == 4:
        index = model_input_index(root)
        volume = volume[index]
        label = str(root.attrs["omero"]["channels"][index]["label"])
    else:
        label = "input channel"
    mask = np.asarray(root["labels"]["nuclei"]["0"][:]) > 0

    return [
        ("image", percentile_normalise(volume), {
            "name": f"{label} (normalised)",
            "colormap": "gray",
            "contrast_limits": (0.0, 1.0),
            "scale": scale,
            "translate": translate,
        }),
        ("labels", mask.astype(np.uint8), {
            "name": "nuclei mask",
            # napari 0.5 takes a label colour map via `colormap`, not `color`.
            # The None key sets the colour for all other label ids; without it
            # napari warns that unlisted ids would be transparent.
            "colormap": {None: "transparent", 1: "magenta"},
            "opacity": 0.75,
            "scale": scale,
            "translate": translate,
        }),
    ]


def main() -> None:
    import napari

    viewer = napari.Viewer(ndisplay=3)
    adders = {"image": viewer.add_image, "labels": viewer.add_labels}
    for kind, data, kwargs in build_layers():
        adders[kind](data, **kwargs)
    napari.run()


if __name__ == "__main__":
    main()
