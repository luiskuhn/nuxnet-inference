"""Tests for the inference-only glue in ``preprocessing.py``.

``preprocessing.py`` cannot be a verbatim copy, because resampling one inference volume
is not the same task as resampling a training pair, and a sliding window has no training
counterpart. So it needs its own tests, and they target the specific ways this code
failed while it was being written:

* OME physical size was read as ``[1,1,1]`` because it looked at the wrong attribute.
* Tiles were not padded to a multiple of 4, and ``torch.cat`` failed inside the decoder
  on a 53-voxel axis.
* ``probability /= np.maximum(weight_sum, 1e-6)`` rescaled corner voxels, because the
  separable taper product there is ~5e-8. A constant model came back as 0.04.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nuxnet_inference_package.preprocessing import (
    REQUIRED_SIZE_MULTIPLE,
    _hann,
    _window_starts,
    apply_training_pipeline,
    normalize_minmax,
    sliding_window_predict,
    voxel_size_from_ome,
)

OME = (
    '<OME><Image><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="float" '
    'SizeX="224" SizeY="224" SizeC="1" SizeZ="64" SizeT="1" '
    'PhysicalSizeX="0.75" PhysicalSizeXUnit="\u00b5m" '
    'PhysicalSizeY="0.75" PhysicalSizeYUnit="\u00b5m" '
    'PhysicalSizeZ="2.5" PhysicalSizeZUnit="\u00b5m"></Pixels></Image></OME>'
)


# ---------------------------------------------------------------------------
# OME spacing
# ---------------------------------------------------------------------------
def test_voxel_size_reads_zyx_order() -> None:
    """Z, Y, X order, not X, Y, Z -- getting this wrong transposes the resampling."""
    size = voxel_size_from_ome(OME)
    assert size is not None
    assert size.tolist() == [2.5, 0.75, 0.75]


def test_voxel_size_missing_returns_none() -> None:
    """A plain TIFF has no physical size; callers must not assume 1 um isotropic."""
    assert voxel_size_from_ome(None) is None
    assert voxel_size_from_ome("") is None
    assert voxel_size_from_ome("<OME><Image><Pixels SizeX='10'/></Image></OME>") is None


def test_voxel_size_rejects_nonpositive() -> None:
    bad = OME.replace('PhysicalSizeZ="2.5"', 'PhysicalSizeZ="0"')
    assert voxel_size_from_ome(bad) is None


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
def test_normalize_minmax_matches_trainer() -> None:
    """The trainer applies per-volume min-max after resampling; so must inference."""
    volume = np.array([[[0.0, 5.0], [10.0, 2.5]]], dtype=np.float32)
    out = normalize_minmax(volume)
    assert out.min() == 0.0 and out.max() == 1.0
    assert out[0, 0, 1] == pytest.approx(0.5)


def test_normalize_constant_volume_is_zeros() -> None:
    out = normalize_minmax(np.full((2, 3, 3), 7.0, dtype=np.float32))
    assert np.all(out == 0.0)


def test_shared_range_maps_identical_tissue_identically() -> None:
    """Two chunks containing the same tissue must normalize the same way.

    This is the bug that made tiled output differ from untiled: each chunk derived
    its own min/max, so a nucleus's brightness depended on its neighbours' extremes.
    """
    # The same 100-voxel value sits in two chunks whose brightest neighbour differs.
    # Per-chunk min-max maps it to 0.5 in one and to 0.2 in the other.
    chunk_a = np.array([[[0.0, 100.0, 200.0]]], dtype=np.float32)
    chunk_b = np.array([[[0.0, 100.0, 500.0]]], dtype=np.float32)

    per_chunk_a = normalize_minmax(chunk_a)
    per_chunk_b = normalize_minmax(chunk_b)
    assert per_chunk_a[0, 0, 1] == pytest.approx(0.5)
    assert per_chunk_b[0, 0, 1] == pytest.approx(0.2)
    assert not np.allclose(per_chunk_a, per_chunk_b), (
        "per-chunk min-max is supposed to differ -- that is the failure mode"
    )

    # With one shared range the same intensity maps to the same value in both.
    shared = (0.0, 500.0)
    out_a = normalize_minmax(chunk_a, value_range=shared)
    out_b = normalize_minmax(chunk_b, value_range=shared)
    assert out_a[0, 0, 1] == pytest.approx(0.2)
    assert out_b[0, 0, 1] == pytest.approx(0.2)


def test_shared_range_clips_values_outside_it() -> None:
    """A pinned range must not let one bright object rescale the chunk."""
    volume = np.array([[[0.0, 50.0, 500.0]]], dtype=np.float32)
    out = normalize_minmax(volume, value_range=(0.0, 100.0))
    assert out.min() == 0.0
    assert out[0, 0, 1] == pytest.approx(0.5)
    assert out[0, 0, 2] == 1.0, "values above the range clip to 1, not to 5.0"


def test_shared_range_reaches_pipeline() -> None:
    """The pinned range must actually be used by the training pipeline, not dropped."""
    volume = np.full((4, 8, 8), 10.0, dtype=np.float32)
    volume[:, :, :4] = 20.0
    _, details = apply_training_pipeline(
        volume,
        source_voxel_size_um=None,  # no resampling, so prepared == input
        normalize=True,
        normalization_range=(0.0, 40.0),
    )
    assert "shared range" in str(details["normalized"])
    assert "0..40" in str(details["normalized"])


# ---------------------------------------------------------------------------
# sliding window geometry
# ---------------------------------------------------------------------------
def test_window_starts_cover_whole_axis() -> None:
    """No voxel may be left uncovered, including when the last step overshoots."""
    for length, patch in ((64, 32), (53, 32), (168, 128), (10, 32), (128, 128)):
        starts = _window_starts(length, patch, max(1, patch // 2))
        assert starts[0] == 0
        assert starts[-1] == max(0, length - patch), f"{length}/{patch}: {starts}"
        for left, right in zip(starts, starts[1:]):
            assert right <= left + patch, f"gap between {left} and {right}"


def test_hann_taper_is_never_zero() -> None:
    """A zero weight would divide a voxel by zero at the edge of a tile."""
    taper = _hann(33)
    assert taper.min() > 0
    assert taper.max() == pytest.approx(1.0)
    assert taper[0] < taper[len(taper) // 2]


class _LogitModel(torch.nn.Module):
    """Stands in for UNet3D: returns constant logits, so softmax is constant too."""

    def __init__(self, background: float = 0.1, foreground: float = 0.9) -> None:
        super().__init__()
        self.background = background
        self.foreground = foreground

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = (x.shape[0], 1) + tuple(x.shape[2:])
        background = torch.full(shape, self.background, device=x.device)
        foreground = torch.full(shape, self.foreground, device=x.device)
        return torch.cat([background, foreground], dim=1)


@pytest.mark.parametrize("shape", [(53, 100, 100), (35, 128, 128), (12, 40, 40)])
def test_sliding_window_pads_non_multiple_of_four(shape) -> None:
    """The decoder needs every spatial axis divisible by 4; the window must pad.

    ``(1, 2, 53, 454, 454)`` raised ``torch.cat`` size mismatch before this was fixed.
    """
    result = sliding_window_predict(
        model=_LogitModel(),
        volume=np.zeros(shape, dtype=np.float32),
        patch_size=(32, 32, 32),
        overlap=0.5,
        batch_size=2,
    )
    assert result.probability.shape == shape
    assert result.patch_zyx == (32, 32, 32)
    for axis in result.patch_zyx:
        assert axis % REQUIRED_SIZE_MULTIPLE == 0


def test_blending_is_flat_for_a_constant_model() -> None:
    """A constant model must give a constant probability everywhere.

    This is the test that catches biased blending. It caught two real bugs: padded
    regions leaking in, and a denominator clamp that rescaled corner voxels from 0.9
    down to 0.04 because the taper product there is ~5e-8.
    """
    model = _LogitModel(0.1, 0.9)
    expected = float(torch.softmax(torch.tensor([0.1, 0.9]), dim=0)[1])
    shape = (35, 100, 100)
    result = sliding_window_predict(
        model=model,
        volume=np.zeros(shape, dtype=np.float32),
        patch_size=(32, 64, 64),
        overlap=0.5,
        batch_size=4,
    )
    assert result.probability.max() - result.probability.min() < 1e-5
    assert result.probability.mean() == pytest.approx(expected, abs=1e-5)

    # The corners are the specific place the clamp used to bite.
    for corner in (
        (0, 0, 0),
        (0, 0, -1),
        (0, -1, 0),
        (-1, 0, 0),
        (-1, -1, -1),
    ):
        assert result.probability[corner] == pytest.approx(expected, abs=1e-5), corner


def test_sliding_window_rejects_bad_input() -> None:
    model = _LogitModel()
    with pytest.raises(ValueError, match="Z, Y, X"):
        sliding_window_predict(model=model, volume=np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="overlap"):
        sliding_window_predict(
            model=model, volume=np.zeros((8, 8, 8), dtype=np.float32), overlap=1.0
        )


# ---------------------------------------------------------------------------
# mask output must keep physical calibration
# ---------------------------------------------------------------------------
def test_written_mask_preserves_voxel_size(tmp_path) -> None:
    """A mask without PhysicalSize misaligns anything that works in microns.

    The input OME-TIFF carries 2.5/0.75/0.75 um; the mask must carry the same, or a
    tool reporting distances treats it as 1 um isotropic and is wrong by 3.5x in XY
    and 5x in Z. This was a real defect: the mask was written with axes only.
    """
    import re

    import tifffile

    from nuxnet_inference_package.cli_pred import write_mask_ome_tiff

    # (4, 8, 8) rather than something smaller: tifffile 2022.4.8 refuses to write
    # very small OME cubes ("axes do not match stored shape"), which is a library
    # quirk and unrelated to what is being tested.
    prediction = np.zeros((4, 8, 8), dtype=np.uint8)
    prediction[1, 2:5, 2:5] = 1
    spacing = np.array([2.5, 0.75, 0.75])

    path = write_mask_ome_tiff(
        prediction, tmp_path / "mask", voxel_size_um=spacing
    )

    with tifffile.TiffFile(path) as tif:
        ome = tif.ome_metadata or ""
        written = np.squeeze(tif.series[0].asarray())
    found = {
        axis: float(re.search(rf'PhysicalSize{axis}="([^"]+)"', ome).group(1))
        for axis in ("Z", "Y", "X")
    }
    assert found == pytest.approx({"Z": 2.5, "Y": 0.75, "X": 0.75})
    assert np.array_equal(written, prediction)


def test_written_mask_without_spacing_is_still_valid(tmp_path) -> None:
    """A plain TIFF input has no calibration to carry; the mask must still be written."""
    import tifffile

    from nuxnet_inference_package.cli_pred import write_mask_ome_tiff

    prediction = np.ones((4, 8, 8), dtype=np.uint8)
    path = write_mask_ome_tiff(prediction, tmp_path / "mask", voxel_size_um=None)

    with tifffile.TiffFile(path) as tif:
        written = np.squeeze(tif.series[0].asarray())
    assert written.shape == prediction.shape
    assert np.array_equal(written, prediction)
