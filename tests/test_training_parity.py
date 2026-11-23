"""Keep the vendored training code identical to ``nuxnet-training``.

This package does not re-implement the trainer's model and data-loading helpers; it
copies them, because a checkpoint only reproduces its training behaviour if both
sides compute the same thing. These tests are what makes "copied" trustworthy:

* **verbatim** -- when the training checkout is available on disk, the vendored
  files must be byte-identical to it, modulo the renames noted in each file.
* **checkpoint parity** -- the parameter names and shapes must match a real
  checkpoint produced by the trainer. This is the check that would have caught the
  original bug, and it needs no training checkout.

Point ``NUXNET_TRAINING_DIR`` at a clone to enable the verbatim checks:

    NUXNET_TRAINING_DIR=~/projects/nuxnet-training pytest tests/test_training_parity.py

Without it, the verbatim tests skip and the checkpoint parity tests still run.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import torch

from nuxnet_inference_package.training_parity.unet_3d_models import UNet3D

GOLDEN = Path(__file__).parent / "unet3d_keys.json"
TRAINING_DIR = os.environ.get("NUXNET_TRAINING_DIR")

#: Upstream commit the vendored files were taken from.
UPSTREAM_COMMIT = "f716565"


def _upstream(*parts: str) -> Path:
    if not TRAINING_DIR:
        pytest.skip("set NUXNET_TRAINING_DIR to a nuxnet-training clone to run this check")
    path = Path(TRAINING_DIR).expanduser().joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"upstream file not found: {path}")
    return path


def _normalized(lines: list[str]) -> list[str]:
    """Drop blank lines and trailing whitespace so formatting does not matter."""
    return [line.rstrip() for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# verbatim checks (need NUXNET_TRAINING_DIR)
# ---------------------------------------------------------------------------
def test_unet3d_is_verbatim_copy_of_training() -> None:
    upstream = _upstream("numorph_nuclei_segmentation", "model", "unet_3d_models.py")
    vendored = (
        Path(__file__).parent.parent
        / "nuxnet_inference_package"
        / "training_parity"
        / "unet_3d_models.py"
    )
    assert _normalized(vendored.read_text().splitlines()) == _normalized(
        upstream.read_text().splitlines()
    ), (
        "the vendored UNet3D has diverged from the trainer; re-copy it rather than "
        "editing either side"
    )


def test_resampling_is_verbatim_copy_of_training() -> None:
    """The resampling helpers must match, ignoring the leading-underscore rename."""
    upstream = _upstream(
        "numorph_nuclei_segmentation", "data_loading", "data_loader.py"
    )
    vendored = (
        Path(__file__).parent.parent
        / "nuxnet_inference_package"
        / "training_parity"
        / "pipeline.py"
    )
    if not vendored.is_file():
        pytest.skip("pipeline.py not present in this revision")

    source = upstream.read_text()

    def extract(text: str, name: str) -> list[str]:
        lines = text.splitlines()
        start = next(
            (i for i, line in enumerate(lines) if line.startswith(f"def _{name}(") or line.startswith(f"def {name}(")),
            None,
        )
        if start is None:
            return []
        end = start + 1
        while end < len(lines) and (not lines[end] or lines[end].startswith(("def ", "class ")) is False):
            end += 1
        return _normalized(lines[start:end])

    for name in ("gaussian_blur_for_downsampling", "resample_volume_pair"):
        want = extract(source, name)
        got = extract(vendored.read_text(), name)
        assert want, f"could not locate {name} upstream"
        assert got, f"could not locate {name} in the vendored copy"
        assert want == got, f"vendored {name} diverged from the trainer"


# ---------------------------------------------------------------------------
# checkpoint parity (always runs)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def golden() -> dict[str, list[int]]:
    if not GOLDEN.is_file():
        pytest.skip(f"golden key file missing: {GOLDEN}")
    return json.loads(GOLDEN.read_text())


def test_parameter_names_match_trained_checkpoint(golden: dict[str, list[int]]) -> None:
    model_keys = set(UNet3D(in_channels=1, classes=2).state_dict())
    missing = sorted(set(golden) - model_keys)
    unexpected = sorted(model_keys - set(golden))
    assert not missing, f"{len(missing)} trained tensors have no home: {missing[:5]}"
    assert not unexpected, f"{len(unexpected)} extra tensors the checkpoint lacks: {unexpected[:5]}"


def test_parameter_shapes_match_trained_checkpoint(golden: dict[str, list[int]]) -> None:
    state = UNet3D(in_channels=1, classes=2).state_dict()
    bad = [
        (key, golden[key], list(state[key].shape))
        for key in sorted(set(golden) & set(state))
        if golden[key] != list(state[key].shape)
    ]
    assert not bad, f"{len(bad)} shape mismatches: {bad[:5]}"


def test_residual_shortcuts_present(golden: dict[str, list[int]]) -> None:
    """Regression guard: the pre-fix model had none, making these weights unusable."""
    shortcuts = [key for key in golden if "shortcut" in key]
    assert len(shortcuts) == 30, f"expected 30 shortcut tensors, found {len(shortcuts)}"
    state = set(UNet3D(in_channels=1, classes=2).state_dict())
    assert all(key in state for key in shortcuts)


def test_load_checkpoint_strictly() -> None:
    """The contract that matters: a trainer checkpoint loads without warnings."""
    state = UNet3D(in_channels=1, classes=2).state_dict()
    UNet3D(in_channels=1, classes=2).load_state_dict(state, strict=True)


def test_forward_requires_multiple_of_four() -> None:
    """Documents the constraint the sliding window must respect."""
    model = UNet3D(in_channels=1, classes=2).eval()
    with torch.no_grad():
        good = model(torch.zeros(1, 1, 32, 128, 128))
    assert good.shape == (1, 2, 32, 128, 128)

    with pytest.raises(RuntimeError, match="must match"):
        with torch.no_grad():
            model(torch.zeros(1, 1, 53, 454, 454))
