"""Data loading, mirrored from the training package.

Mirrors ``numorph_nuclei_segmentation/data_loading/``.

``data_loader.py`` here is a verbatim copy of the trainer's resampling helpers.
It is imported by ``nuxnet_inference_package.preprocessing``, which adds the
inference-only pieces (single-volume preparation and sliding-window coverage).
"""

from nuxnet_inference_package.data_loading.data_loader import (
    DEFAULT_TARGET_VOXEL_SIZE_UM,
    component_centroids,
    gaussian_blur_for_downsampling,
    resample_volume_pair,
)

__all__ = [
    "DEFAULT_TARGET_VOXEL_SIZE_UM",
    "component_centroids",
    "gaussian_blur_for_downsampling",
    "resample_volume_pair",
]
