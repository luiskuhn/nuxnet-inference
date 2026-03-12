from __future__ import annotations

import os
import random

import click
import numpy as np
import torch
from rich import print


def _configure_cublas_workspace() -> None:
    """Set CuBLAS workspace config expected for deterministic CUDA behavior."""
    expected = ":4096:8"
    fallback = ":16:8"
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current in {None, ""}:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected
        return

    if current not in {expected, fallback}:
        print(
            "[bold yellow]Warning:[/bold yellow] CUBLAS_WORKSPACE_CONFIG is already set to "
            f"'{current}'. Deterministic CUDA behavior typically expects '{expected}' or '{fallback}'."
        )


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Configure RNG seeds and deterministic runtime options.

    Parameters
    ----------
    seed:
        Seed value for Python random, NumPy, and Torch.
    deterministic:
        If True, enable deterministic execution as far as possible in Torch/CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        if deterministic:
            _configure_cublas_workspace()
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(False)
    except RuntimeError as exc:
        raise click.ClickException(
            "Deterministic mode could not be fully enabled in this environment. "
            "Try '--no-deterministic' or use operators/hardware with deterministic implementations. "
            f"Details: {exc}"
        ) from exc
