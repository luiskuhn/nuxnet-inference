"""Keep the mirrored dataloading helpers identical to ``nuxnet-training``.

``preprocessing.py`` builds on ``data_loading/data_loader.py``, which is a copy of the
trainer's resampling helpers rather than a re-implementation. If that copy drifts, the
resampling inference performs stops being the resampling training performed, and the
checkpoint is fed something it was never trained on. These tests are the guard.

Set ``NUXNET_TRAINING_DIR`` to a ``nuxnet-training`` clone to run the comparison:

    NUXNET_TRAINING_DIR=~/projects/nuxnet-training pytest tests/test_dataloading_parity.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parent.parent / "nuxnet_inference_package"
TRAINING_DIR = os.environ.get("NUXNET_TRAINING_DIR")

FUNCTIONS = ("gaussian_blur_for_downsampling", "component_centroids", "resample_volume_pair")


def _upstream() -> Path:
    if not TRAINING_DIR:
        pytest.skip("set NUXNET_TRAINING_DIR to a nuxnet-training clone to run this check")
    path = Path(TRAINING_DIR).expanduser().joinpath(
        "numorph_nuclei_segmentation", "data_loading", "data_loader.py"
    )
    if not path.is_file():
        pytest.skip(f"upstream file not found: {path}")
    return path


def _normalized(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def _extract(lines: list[str], function_name: str) -> list[str]:
    """Return one top-level function body: from its ``def`` to the next top-level one.

    The ``def`` line is normalized to an unnamed form so the documented rename (the
    trainer's private ``_name`` versus this package's public ``name``) does not register
    as divergence. Everything else must match exactly.
    """
    prefix = f"def {function_name}("
    start = next((i for i, line in enumerate(lines) if line.startswith(prefix)), None)
    if start is None:
        return []
    end = start + 1
    while end < len(lines) and not lines[end].startswith(("def ", "class ")):
        end += 1
    body = lines[start:end]
    body[0] = "def <normalized>"
    return body


def _denormalize(lines: list[str]) -> list[str]:
    """Undo the underscore so internal calls compare equal across the rename."""
    out = []
    for line in lines:
        line = line.replace("_gaussian_blur_for_downsampling(", "gaussian_blur_for_downsampling(")
        line = line.replace("_component_centroids(", "component_centroids(")
        out.append(line)
    return out


def test_helpers_are_verbatim() -> None:
    upstream_path = _upstream()
    mirrored_path = PACKAGE / "data_loading" / "data_loader.py"
    if not mirrored_path.is_file():
        pytest.skip("mirrored data_loader.py not present in this revision")

    source = _normalized(upstream_path.read_text())
    copy = _normalized(mirrored_path.read_text())

    for name in FUNCTIONS:
        # The trainer keeps these private except resample_volume_pair, which is public.
        upstream_body = _extract(source, f"_{name}") or _extract(source, name)
        mirrored_body = _extract(copy, name)
        assert upstream_body, f"could not locate {name} (public or private) in the trainer"
        assert mirrored_body, f"could not locate {name} in the mirrored copy"
        assert _denormalize(upstream_body) == _denormalize(mirrored_body), (
            f"mirrored {name} diverged from the trainer; re-copy rather than edit"
        )
