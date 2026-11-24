"""Data-loading primitives copied verbatim from ``nuxnet-training``.

A checkpoint reproduces its training behaviour only if inference feeds the network
what the trainer fed it. Re-implementing that pipeline invites drift, so this module
is a **verbatim copy** rather than a paraphrase:

    luiskuhn/nuxnet-training @ f716565
    numorph_nuclei_segmentation/data_loading/data_loader.py  (lines 124-228)

Mirrored path:

    nuxnet-training       numorph_nuclei_segmentation/data_loading/data_loader.py
    nuxnet-inference      nuxnet_inference_package/data_loading/data_loader.py

Vendored: ``_gaussian_blur_for_downsampling``, ``_component_centroids``,
``resample_volume_pair`` -- renamed without the leading underscore, because here they
are the public interface rather than private helpers.

Copying avoids a dependency on ``nidavellir_tools`` and ``pytorch_lightning``, which
the trainer needs for dataset download and training but inference does not.

Do not edit these functions. If upstream changes, re-copy, and
``tests/test_training_parity.py`` will confirm the copy still matches.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "DEFAULT_TARGET_VOXEL_SIZE_UM",
    "gaussian_blur_for_downsampling",
    "component_centroids",
    "resample_volume_pair",
]

#: Training default, copied from the same module.
DEFAULT_TARGET_VOXEL_SIZE_UM = (3.0, 1.0, 1.0)


def gaussian_blur_for_downsampling(
    image: torch.Tensor, scale: tuple[float, float, float]
) -> torch.Tensor:
    """Apply a separable Gaussian low-pass filter on axes that will shrink."""
    for axis, ratio in enumerate(scale):
        if ratio >= 1:
            continue
        sigma = max(0.5, 0.5 * (1.0 / ratio - 1.0))
        radius = max(1, int(np.ceil(3 * sigma)))
        coordinates = torch.arange(
            -radius, radius + 1, dtype=image.dtype, device=image.device
        )
        kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = kernel.numel()
        padding = [0, 0, 0]
        padding[axis] = radius
        channels = image.shape[1]
        image = F.conv3d(
            image,
            kernel.view(shape).repeat(channels, 1, 1, 1, 1),
            padding=padding,
            groups=channels,
        )
    return image


def component_centroids(mask: np.ndarray) -> list[np.ndarray]:
    """Return 26-connected foreground component centroids without an extra dependency."""
    remaining = set(map(tuple, np.argwhere(mask > 0)))
    centroids = []
    neighbours = [
        (a, b, c)
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for c in (-1, 0, 1)
        if (a, b, c) != (0, 0, 0)
    ]
    while remaining:
        stack, component = [remaining.pop()], []
        while stack:
            voxel = stack.pop()
            component.append(voxel)
            for delta in neighbours:
                candidate = tuple(voxel[i] + delta[i] for i in range(3))
                if candidate in remaining:
                    remaining.remove(candidate)
                    stack.append(candidate)
        centroids.append(np.asarray(component, dtype=np.float64).mean(axis=0))
    return centroids


def resample_volume_pair(
    image: np.ndarray,
    label: np.ndarray,
    source_voxel_size_um,
    target_voxel_size_um,
):
    """Place a CZYX/ZYX pair on the requested physical voxel-size grid."""
    source_size = np.asarray(source_voxel_size_um, dtype=np.float64)
    target_size = np.asarray(target_voxel_size_um, dtype=np.float64)
    if (
        source_size.shape != (3,)
        or target_size.shape != (3,)
        or np.any(source_size <= 0)
        or np.any(target_size <= 0)
    ):
        raise ValueError(
            "source and target voxel sizes must be three positive µm values in Z,Y,X order"
        )
    # Preserve the physical extent: a smaller requested voxel needs more voxels.
    resize_factors = source_size / target_size
    output_shape = tuple(
        max(1, int(round(length * factor)))
        for length, factor in zip(label.shape, resize_factors)
    )
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()[None]
    image_tensor = gaussian_blur_for_downsampling(image_tensor, tuple(resize_factors))
    resized_image = F.interpolate(
        image_tensor, size=output_shape, mode="trilinear", align_corners=False
    )[0]
    resized_label = F.interpolate(
        torch.from_numpy(np.ascontiguousarray(label)).float()[None, None],
        size=output_shape,
        mode="nearest-exact",
    )[0, 0].to(torch.int64)
    # Nearest-neighbour interpolation preserves every source voxel when no
    # axis shrinks. Connected-component recovery is only needed while
    # downsampling; running the Python flood fill for the common equal-spacing
    # or upsampling case needlessly scans the entire foreground mask and can
    # make a DataLoader appear to hang on large volumes.
    if np.any(resize_factors < 1.0):
        # Voxel-centre coordinates map as (p + .5) * source / target - .5.
        for centroid in component_centroids(label):
            transformed = np.rint((centroid + 0.5) * resize_factors - 0.5).astype(
                int
            )
            if np.all(transformed >= 0) and np.all(
                transformed < np.asarray(output_shape)
            ):
                resized_label[tuple(transformed)] = 1
    return resized_image.numpy(), resized_label.numpy()


