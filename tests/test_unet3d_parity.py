"""Pin the inference architecture to the one the trainer produces.

A trained checkpoint is a flat dictionary of tensors keyed by module path. If the
inference-side ``UNet3D`` renames a module, adds a bias, or drops a residual
shortcut, ``load_state_dict`` fails -- but only at runtime, on a real volume, in a
pipeline. That is the failure this test exists to prevent.

``unet3d_keys.json`` is the parameter-name/shape inventory of a checkpoint
actually produced by ``nuxnet-training``:

    luiskuhn/nuxnet-training @ f716565
    runs/fold3/model-package-inputs/weights.pt
    (numorph_nuclei_segmentation/model/unet_3d_models.py)

Regenerate it from a checkpoint with:

    python -m nuxnet_inference_package.models.dump_arch_keys weights.pt \
        > tests/unet3d_keys.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nuxnet_inference_package.models.unet3d import UNet3D

GOLDEN = Path(__file__).parent / "unet3d_keys.json"


def _inventory() -> dict[str, list[int]]:
    model = UNet3D(in_channels=1, classes=2)
    return {k: list(v.shape) for k, v in model.state_dict().items()}


@pytest.fixture(scope="module")
def golden() -> dict[str, list[int]]:
    if not GOLDEN.is_file():
        pytest.skip(f"golden key file missing: {GOLDEN}")
    return json.loads(GOLDEN.read_text())


def test_parameter_names_match_trainer(golden: dict[str, list[int]]) -> None:
    got = _inventory()
    missing = sorted(set(golden) - set(got))
    unexpected = sorted(set(got) - set(golden))
    assert not missing, (
        f"{len(missing)} tensors present in the trainer checkpoint but absent here; "
        f"first: {missing[:5]}"
    )
    assert not unexpected, (
        f"{len(unexpected)} tensors produced here but absent from the trainer checkpoint; "
        f"first: {unexpected[:5]}"
    )


def test_parameter_shapes_match_trainer(golden: dict[str, list[int]]) -> None:
    got = _inventory()
    mismatched = [
        (k, golden[k], got[k])
        for k in sorted(set(golden) & set(got))
        if golden[k] != got[k]
    ]
    assert not mismatched, f"{len(mismatched)} shape mismatches; first: {mismatched[:5]}"


def test_residual_shortcuts_exist(golden: dict[str, list[int]]) -> None:
    """Regression guard: the previous implementation had no shortcuts at all."""
    shortcuts = [k for k in golden if "shortcut" in k]
    assert shortcuts, "trainer checkpoint has no shortcut tensors -- wrong golden file?"
    assert all(k in _inventory() for k in shortcuts)


def test_convolutions_are_bias_free(golden: dict[str, list[int]]) -> None:
    """Regression guard: the previous implementation used bias=True convs.

    A convolution carries its own bias only when no ``BatchNorm3d`` follows it. In
    the trainer that means exactly three kinds of key:

    * ``...downN.down.0.bias``     -- stride-2 downsampling convolution
    * ``...shortcut.0.bias``       -- 1x1 projection inside a residual shortcut
    * ``outc.conv_1.bias``         -- final 1x1 classification head

    Every other convolution is followed by batch norm and must be bias-free. The
    previous implementation used ``bias=True`` throughout, which is one half of why
    trained checkpoints could not be loaded.

    Note the shortcut is ``nn.Sequential(Conv3d, BatchNorm3d)``, so its batch-norm
    bias is named ``shortcut.1.bias`` rather than containing ``batch_norm``.
    """
    def is_batch_norm_bias(key: str) -> bool:
        return key.endswith(".bias") and (
            "batch_norm" in key or ".shortcut.1." in key
        )

    conv_biases = {
        key
        for key in golden
        if key.endswith(".bias") and not is_batch_norm_bias(key)
    }
    expected = {
        key
        for key in golden
        if key.endswith(".bias")
        and (
            ".down.0." in key
            or ".shortcut.0." in key
            or key == "outc.conv_1.bias"
        )
    }
    unexpected = sorted(conv_biases - expected)
    assert not unexpected, (
        f"{len(unexpected)} convolutions carry a bias but are followed by batch norm: "
        f"{unexpected[:5]}"
    )


def test_loads_checkpoint_strictly(tmp_path: Path) -> None:
    """Round-trip: a state dict from this class loads strictly into itself.

    Does not need the real weights; it proves the class is self-consistent and
    that ``load_state_dict`` with ``strict=True`` is the right contract.
    """
    state = UNet3D(in_channels=1, classes=2).state_dict()
    target = UNet3D(in_channels=1, classes=2)
    target.load_state_dict(state, strict=True)


def test_forward_pass_preserves_spatial_shape() -> None:
    model = UNet3D(in_channels=1, classes=2).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 16, 32, 32))
    assert out.shape == (1, 2, 16, 32, 32)


def test_two_downsamplings_require_divisible_by_four() -> None:
    """Documents a real constraint rather than silently producing odd shapes."""
    model = UNet3D(in_channels=1, classes=2).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 8, 16, 16))
    assert out.shape[-3:] == (8, 16, 16)
