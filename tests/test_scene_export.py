"""A scene that opens as blank must fail, not pass.

These tests exist because a broken export shipped and looked finished. The store had
the right shape, the right dtype, the right multiscales metadata and the right
transforms; napari opened it and reported two layers with matching scale and
translation. Every voxel was zero.

The cause is a chunk-key mismatch with no error path. OME-NGFF 0.2 moved chunk keys
from flat (``0.0.0``) to nested (``0/0/0``), and ``ome_zarr`` -- which napari-ome-zarr
uses for every store it reads -- hardcodes ``dimension_separator="/"`` when opening one
(``FormatV02.init_store``). Write flat chunks, read nested keys, and every chunk lookup
misses. zarr answers a miss with ``fill_value`` rather than raising, so the read
succeeds and returns zeros.

Two consequences follow, and both are tested here:

* shape and dtype checks cannot detect this, so the guard has to assert on content;
* the separator has to be pinned at write time, because ``zarr.open_group(path)`` does
  not infer it back from the ``.zarray`` on read in the versions this package pins.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from zarr.storage import FSStore

zarr_format = pytest.mark.skipif(
    not hasattr(zarr, "open_array"), reason="zarr v2 API required"
)


def _write(out, separator: str, data: np.ndarray, chunks) -> None:
    store = FSStore(str(out), mode="w", dimension_separator=separator, auto_mkdir=True)
    group = zarr.open_group(store=store, mode="w")
    group.create_dataset(
        "0", data=data, chunks=chunks, overwrite=True, fill_value=0,
        dimension_separator=separator,
    )


def _read_like_ome_zarr(out) -> np.ndarray:
    """Read the way ``ome_zarr`` does: a store pinned to separator ``"/"``."""
    store = FSStore(str(out), mode="r", dimension_separator="/")
    return np.asarray(zarr.open_array(store=store, path="0", mode="r")[:])


@zarr_format
def test_flat_chunks_read_as_zeros_through_a_nested_store(tmp_path):
    """The failure mode, pinned so the reason for the fix stays visible.

    This asserts the *bug*: flat chunks plus a nested reader gives zeros with no
    exception. If a future zarr starts inferring the separator, this test fails and
    the comment in ``export_roi_scene.open_store`` can be relaxed.
    """
    data = np.zeros((32, 32, 32), dtype=np.uint16)
    data[8:16, 8:16, 8:16] = 1234

    out = tmp_path / "flat.zarr"
    _write(out, ".", data, (16, 16, 16))

    # Written correctly on its own terms...
    assert np.asarray(zarr.open_array(str(out / "0"), mode="r")[:]).max() == 1234

    # ...but invisible to the reader napari uses. Silent: no raise, just zeros.
    got = _read_like_ome_zarr(out)
    assert got.shape == data.shape
    assert got.dtype == data.dtype
    assert got.max() == 0


@zarr_format
def test_nested_chunks_are_readable_by_a_nested_store(tmp_path):
    """The fix: nested chunk keys round-trip through the reader napari uses."""
    data = np.zeros((32, 32, 32), dtype=np.uint16)
    data[8:16, 8:16, 8:16] = 1234

    out = tmp_path / "nested.zarr"
    _write(out, "/", data, (16, 16, 16))

    got = _read_like_ome_zarr(out)
    np.testing.assert_array_equal(got, data)


@zarr_format
def test_exported_store_layout_is_nested(tmp_path):
    """`open_store` must produce nested chunk keys and carry the separator."""
    from scripts.export_roi_scene import open_store

    out = tmp_path / "scene.zarr"
    group = open_store(out, mode="w")
    data = np.zeros((32, 32, 32), dtype=np.uint16)
    data[8:16, 8:16, 8:16] = 1234
    array = group.create_dataset(
        "0", data=data, chunks=(16, 16, 16), overwrite=True, fill_value=0,
        dimension_separator="/",
    )
    assert array.shape == data.shape

    # Nested layout means chunk directories, not dotted files.
    chunk_entries = [p.name for p in (out / "0").iterdir() if p.is_dir()]
    assert chunk_entries, "expected nested chunk directories, found flat files"

    np.testing.assert_array_equal(_read_like_ome_zarr(out), data)


def test_verify_scene_rejects_a_blank_store(tmp_path):
    """The guard itself: a store of zeros must raise rather than pass.

    ``verify_scene`` is the check that would have caught the shipped bug, so it is
    tested against exactly that input -- a store with correct NGFF metadata and
    correct shapes whose contents are all zero. It must reach the content check and
    fail there, not pass because the metadata looked right.
    """
    from scripts.export_roi_scene import open_store, write_group, verify_scene
    from numcodecs import Blosc

    out = tmp_path / "blank.zarr"
    group = open_store(out, mode="w")

    shape = (32, 32, 32)
    scale = [5.0, 3.5, 3.5]
    level_scales = [[s * (2 ** i) for s in scale] for i in range(1)]
    zeros = np.zeros(shape, dtype=np.uint16)
    compressor = Blosc(cname="zstd", clevel=5)

    # A structurally valid scene: image at the root, mask under labels/.
    write_group(group, None, [zeros], scale, [0.0, 0.0, 0.0], level_scales,
                "FFFFFF", "image", compressor)
    labels = group.require_group("labels")
    labels.attrs["labels"] = ["nuclei"]
    write_group(labels, "nuclei", [zeros.astype(np.uint8)], scale,
                [0.0, 0.0, 0.0], level_scales, "FF00FF", "mask", compressor,
                ngff_type="label")

    # The metadata is fine, so the reader returns both layers -- and the content
    # check is the only thing standing between this and a "successful" export.
    with pytest.raises(SystemExit, match="all zeros"):
        verify_scene(out, expected_mask_foreground=None)


def test_roundtrip_returns_to_the_source_grid():
    """The grid contract: resample to the training grid, predict, come back.

    The mask and every centroid it yields must be in the *source* voxel frame, so a
    chunk's result can be pasted into a store authored at the base LSFM resolution
    without a second resampling step. If this drifts, positions silently land at the
    training sampling and the layer no longer overlays the image.
    """
    torch = pytest.importorskip("torch")

    from nuxnet_inference_package.preprocessing import apply_training_pipeline
    from nuxnet_inference_package.cli_pred import extract_nuclei_instances

    source_um = np.array([5.005, 3.5455, 3.5455])   # the ear ROI
    target_um = np.array([3.0, 1.0, 1.0])           # the training grid

    shape = (40, 120, 120)
    volume = np.zeros(shape, np.float32)
    truth = [(10, 30, 30), (25, 80, 50), (33, 60, 95)]
    for z, y, x in truth:
        volume[z - 1:z + 2, y - 2:y + 3, x - 2:x + 3] = 1000.0

    working, details = apply_training_pipeline(
        volume, source_voxel_size_um=source_um, target_voxel_size_um=target_um,
        normalize=True,
    )
    assert details["resampled"] is True

    # Upsampling must follow the spacing ratio, not some fixed factor.
    expected = np.round(np.array(shape) * source_um / target_um).astype(int)
    np.testing.assert_allclose(working.shape, expected, atol=1)

    # Back-project exactly as cli_pred does.
    prediction = (working > 0.5).astype(np.float32)
    back = torch.from_numpy(prediction[None, None])
    back = torch.nn.functional.interpolate(
        back, size=volume.shape, mode="trilinear", align_corners=False
    )
    restored = (back[0, 0].numpy() >= 0.5).astype(np.uint8)

    assert restored.shape == volume.shape, "mask must be returned on the input grid"

    instances = extract_nuclei_instances(restored, nuclei_label=1, radius=2.0)
    assert len(instances) == len(truth)

    # And the centroids must be in source voxels, i.e. recover the planted positions.
    for z, y, x in truth:
        best = min(instances,
                   key=lambda i: (i["z"] - z) ** 2 + (i["y"] - y) ** 2 + (i["x"] - x) ** 2)
        distance = np.sqrt((best["z"] - z) ** 2 + (best["y"] - y) ** 2
                           + (best["x"] - x) ** 2)
        assert distance < 1.0, f"centroid {best} drifted from truth {(z, y, x)}"
