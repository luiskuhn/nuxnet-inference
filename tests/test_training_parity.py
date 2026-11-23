"""Keep the mirrored model identical to ``nuxnet-training``.

This package does not re-implement the trainer's model; it copies it, because a
checkpoint reproduces its training behaviour only if both sides compute the same
thing. These tests are what makes "copied" trustworthy.

Two kinds of check:

* **verbatim** -- when a training checkout is available, the mirrored file must be
  identical to it. Point ``NUXNET_TRAINING_DIR`` at a clone to enable it.
* **checkpoint parity** -- parameter names, shapes and residual shortcuts must match
  a real checkpoint produced by the trainer. This needs no training checkout, and it
  is the check that would have caught the original bug.

    NUXNET_TRAINING_DIR=~/projects/nuxnet-training pytest tests/test_training_parity.py

Without ``NUXNET_TRAINING_DIR`` the verbatim check skips and the rest still runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from nuxnet_inference_package.model.unet_3d_models import UNet3D

GOLDEN = Path(__file__).parent / "unet3d_keys.json"
PACKAGE = Path(__file__).parent.parent / "nuxnet_inference_package"
TRAINING_DIR = os.environ.get("NUXNET_TRAINING_DIR")


def _upstream(*parts: str) -> Path:
    if not TRAINING_DIR:
        pytest.skip("set NUXNET_TRAINING_DIR to a nuxnet-training clone to run this check")
    path = Path(TRAINING_DIR).expanduser().joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"upstream file not found: {path}")
    return path


def _normalized(text: str) -> list[str]:
    """Drop blank lines and trailing whitespace so formatting cannot matter."""
    return [line.rstrip() for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# verbatim check (needs NUXNET_TRAINING_DIR)
# ---------------------------------------------------------------------------
def test_model_is_verbatim_copy_of_trainer() -> None:
    """The mirrored model must be identical to the trainer's file, byte for byte."""
    upstream = _upstream("numorph_nuclei_segmentation", "model", "unet_3d_models.py")
    mirrored = PACKAGE / "model" / "unet_3d_models.py"
    assert _normalized(mirrored.read_text()) == _normalized(upstream.read_text()), (
        "the mirrored UNet3D has diverged from the trainer; re-copy the file rather "
        "than editing either side"
    )


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
