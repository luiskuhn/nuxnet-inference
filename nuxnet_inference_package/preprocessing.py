"""Prepare a volume for inference on the training grid.

Two layers, deliberately separated:

``data_loading.data_loader``
    Verbatim copies of the trainer's resampling helpers, at the path that mirrors
    ``nuxnet-training``. This is the definition of "what the trainer did" and must
    not be edited locally.

``apply_training_pipeline`` / ``sliding_window_predict`` (this module)
    Inference-only glue. Resampling a training *pair* differs from preparing a
    single inference volume, and a sliding window has no training counterpart (the
    trainer sampled one crop per step), so these cannot be copied.

If inference and training disagree, the fix belongs in the vendored layer, and
``tests/test_training_parity.py`` will say when upstream has moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from nuxnet_inference_package.data_loading.data_loader import (
    DEFAULT_TARGET_VOXEL_SIZE_UM,
    resample_volume_pair,
)

__all__ = [
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_TARGET_VOXEL_SIZE_UM",
    "SlidingWindowResult",
    "normalize_minmax",
    "apply_training_pipeline",
    "sliding_window_predict",
    "voxel_size_from_ome",
]

#: Trainer default ``--patch-size``. Keep in sync with the training repo.
DEFAULT_PATCH_SIZE: tuple[int, int, int] = (32, 128, 128)

#: The network downsamples twice, so every spatial axis must be a multiple of this.
REQUIRED_SIZE_MULTIPLE = 4


# ---------------------------------------------------------------------------
# input preparation
# ---------------------------------------------------------------------------
def voxel_size_from_ome(ome_metadata: str | None) -> np.ndarray | None:
    """Read ``PhysicalSizeZ/Y/X`` from OME-XML.

    Returns ``None`` when the file carries no physical size, which is normal for a
    plain TIFF. Callers must handle that explicitly rather than assume isotropic
    1 µm voxels, which is what an unguarded default amounts to.
    """
    if not ome_metadata:
        return None
    import re

    found: dict[str, float] = {}
    for axis in ("Z", "Y", "X"):
        match = re.search(rf'PhysicalSize{axis}="([-+0-9.eE]+)"', ome_metadata)
        if match is not None:
            found[axis] = float(match.group(1))
    if len(found) != 3:
        return None
    size = np.array([found["Z"], found["Y"], found["X"]], dtype=np.float64)
    return size if np.all(size > 0) else None


def normalize_minmax(
    volume: np.ndarray,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Min-max to 0..1, using the trainer's per-volume range unless one is given.

    Training normalized each volume by its own min/max, so with no ``value_range``
    this is exactly what the trainer did. Passing ``value_range`` pins the mapping
    to a fixed bracket instead.

    That matters when a large volume is processed in pieces. Each piece has its own
    min/max, so identical tissue lands on a different part of the 0..1 curve
    depending on which piece it fell into, and with the model's threshold fixed the
    same nucleus can be foreground in one chunk and background in the next.
    Measured on the ear ROI, normalizing two halves separately dropped Dice between
    the tiled and untiled result to 0.458. One range derived from the whole region
    and reused for every chunk is what makes tiling invisible in the output.
    """
    if value_range is None:
        low, high = float(volume.min()), float(volume.max())
    else:
        low, high = (float(v) for v in value_range)
    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    # Clip, because a pinned range may not cover this chunk's extremes: a chunk
    # holding one unusually bright object would otherwise push everything else
    # below the range the other chunks were mapped with.
    scaled = (volume.astype(np.float32) - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def apply_training_pipeline(
    volume: np.ndarray,
    source_voxel_size_um: Sequence[float] | None,
    target_voxel_size_um: Sequence[float] = DEFAULT_TARGET_VOXEL_SIZE_UM,
    normalize: bool = True,
    normalization_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Resample one volume to the training grid and normalize it.

    Delegates to the trainer's own ``resample_volume_pair`` so the interpolation and
    anti-aliasing are identical to training rather than an approximation of them. A
    dummy label is supplied because that function resamples an image/label pair; it
    is discarded here.

    Returns the prepared volume plus a report of what was done, including whether
    resampling was skipped and why.
    """
    details: dict[str, object] = {}
    working = np.ascontiguousarray(volume, dtype=np.float32)

    if source_voxel_size_um is None:
        details["resampled"] = False
        details["warning"] = (
            "input carries no PhysicalSize metadata; voxel spacing is unknown, so the "
            "volume is used at its stored sampling. A checkpoint trained with "
            "resampling will be out of distribution here."
        )
    else:
        source = np.asarray(source_voxel_size_um, dtype=np.float64)
        target = np.asarray(target_voxel_size_um, dtype=np.float64)
        details["source_voxel_size_um"] = [float(v) for v in source]
        details["target_voxel_size_um"] = [float(v) for v in target]
        if np.allclose(source, target):
            details["resampled"] = False
            details["note"] = "input is already on the target grid"
        else:
            dummy_label = np.zeros(working.shape, dtype=np.int64)
            # ``resample_volume_pair`` expects a CZYX image, as the trainer supplied
            # it, so add the channel axis and drop it again afterwards.
            resampled, _ = resample_volume_pair(
                working[None], dummy_label, source, target
            )
            working = np.ascontiguousarray(resampled[0], dtype=np.float32)
            details["resampled"] = True

    details["prepared_shape_zyx"] = [int(v) for v in working.shape]
    if normalize:
        working = normalize_minmax(working, value_range=normalization_range)
        details["normalized"] = (
            "minmax, per volume" if normalization_range is None
            else f"minmax, shared range {normalization_range[0]:g}.."
                 f"{normalization_range[1]:g}"
        )
    else:
        details["normalized"] = False
    return working, details


# ---------------------------------------------------------------------------
# sliding window (inference only; no training counterpart)
# ---------------------------------------------------------------------------
@dataclass
class SlidingWindowResult:
    """Probability map plus the coverage numbers needed to audit it."""

    probability: np.ndarray
    n_tiles: int
    patch_zyx: tuple[int, int, int]
    overlap: float


def _hann(n: int, floor: float = 1e-3) -> np.ndarray:
    """Separable taper, never exactly zero so no voxel is left unweighted."""
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    return np.maximum(np.hanning(n + 2)[1:-1].astype(np.float32), floor)


def _window_starts(length: int, patch: int, step: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, step))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def sliding_window_predict(
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size: Sequence[int] = DEFAULT_PATCH_SIZE,
    device: torch.device | str = "cpu",
    overlap: float = 0.5,
    batch_size: int = 4,
    classes: int = 2,
) -> SlidingWindowResult:
    """Foreground probability over a whole volume, blended across tiles.

    Hard tiling leaves seams at every boundary, which show up as missed or doubled
    nuclei, so overlapping tiles are blended with a separable Hann window.

    Every tile reaches the network at exactly the declared ``patch_size``,
    zero-padded where the volume is smaller, because that is what the trainer did
    (``_crop_or_pad`` always returns a full ``patch_size`` crop). Padding also keeps
    each spatial axis divisible by 4; without that the decoder's upsampled feature
    map is one voxel larger than its skip connection and ``torch.cat`` fails.

    Args:
        volume: normalized ``(Z, Y, X)`` array.
        patch_size: ``(Z, Y, X)`` tile size; the trainer's ``--patch-size``.
        overlap: fraction of a patch shared with its neighbour.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a (Z, Y, X) volume, got shape {volume.shape}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    depth, height, width = volume.shape
    # Non-integer factors are why every spatial axis has to be a multiple of
    # REQUIRED_SIZE_MULTIPLE; use ceiling division so this works for any positive patch.
    pz, py, px = (
        max(
            REQUIRED_SIZE_MULTIPLE,
            ((int(p) + REQUIRED_SIZE_MULTIPLE - 1) // REQUIRED_SIZE_MULTIPLE) * REQUIRED_SIZE_MULTIPLE,
        )
        for p in patch_size
    )
    step = tuple(max(1, int(round(p * (1 - overlap)))) for p in (pz, py, px))

    zs = _window_starts(depth, pz, step[0])
    ys = _window_starts(height, py, step[1])
    xs = _window_starts(width, px, step[2])

    weight = _hann(pz)[:, None, None] * _hann(py)[None, :, None] * _hann(px)[None, None, :]
    weight = weight.astype(np.float32)

    probability = np.zeros(volume.shape, dtype=np.float32)
    weight_sum = np.zeros(volume.shape, dtype=np.float32)

    device = torch.device(device)
    coords = [(z, y, x) for z in zs for y in ys for x in xs]

    model.eval()
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            chunk = coords[start : start + batch_size]
            tiles = []
            for z, y, x in chunk:
                tile = volume[z : z + pz, y : y + py, x : x + px]
                if tile.shape != (pz, py, px):
                    padded = np.zeros((pz, py, px), dtype=np.float32)
                    padded[: tile.shape[0], : tile.shape[1], : tile.shape[2]] = tile
                    tile = padded
                tiles.append(tile)

            tensor = torch.from_numpy(np.stack(tiles)[:, None]).float().to(device)
            logits = model(tensor)
            if classes == 2:
                foreground = torch.softmax(logits, dim=1)[:, 1]
            else:
                foreground = torch.softmax(logits, dim=1).amax(dim=1)
            foreground = foreground.cpu().numpy()

            for (z, y, x), tile in zip(chunk, foreground):
                stop_z = min(z + pz, depth)
                stop_y = min(y + py, height)
                stop_x = min(x + px, width)
                # Only the part of the tile that maps onto real voxels contributes.
                valid = (slice(z, stop_z), slice(y, stop_y), slice(x, stop_x))
                w = weight[: stop_z - z, : stop_y - y, : stop_x - x]
                probability[valid] += tile[: stop_z - z, : stop_y - y, : stop_x - x] * w
                weight_sum[valid] += w

    # Divide only where a voxel actually received weight. Clamping the denominator
    # (e.g. np.maximum(weight_sum, 1e-6)) looks equivalent but is not: at the extreme
    # volume corners the separable taper product reaches ~5e-8, so a clamp larger than
    # the true weight rescales those voxels instead of leaving them alone. Measured:
    # a constant 0.9 model came back as 0.04 in 104 corner voxels.
    probability = np.divide(
        probability,
        weight_sum,
        out=np.zeros_like(probability),
        where=weight_sum > 0,
    )
    return SlidingWindowResult(
        probability=probability,
        n_tiles=len(coords),
        patch_zyx=(pz, py, px),
        overlap=overlap,
    )
