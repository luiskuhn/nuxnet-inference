"""Dump the parameter-name/shape inventory of a ``UNet3D`` checkpoint.

The inventory is what ``tests/test_unet3d_parity.py`` compares against, so this
module is the documented way to regenerate the golden file after a deliberate
architecture change on the training side:

    python -m nuxnet_inference_package.models.dump_arch_keys path/to/weights.pt \
        > tests/unet3d_keys.json

Pass ``--from-model`` instead of a path to dump the inventory of the architecture
defined in this package, which is useful for a quick visual diff:

    python -m nuxnet_inference_package.models.dump_arch_keys --from-model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from nuxnet_inference_package.models.unet3d import UNet3D


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint, tolerating the wrappers training frameworks add."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only
        state = torch.load(path, map_location="cpu")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise SystemExit(f"checkpoint {path} did not contain a state dict")
    return state


def inventory(state: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(tensor.shape) for key, tensor in sorted(state.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", nargs="?", help="path to a .pt/.ckpt checkpoint")
    parser.add_argument(
        "--from-model",
        action="store_true",
        help="dump this package's UNet3D instead of a checkpoint",
    )
    parser.add_argument(
        "--in-channels", type=int, default=1, help="with --from-model: input channels"
    )
    parser.add_argument(
        "--classes", type=int, default=2, help="with --from-model: output classes"
    )
    args = parser.parse_args(argv)

    if args.from_model:
        if args.checkpoint:
            parser.error("pass either a checkpoint path or --from-model, not both")
        state = UNet3D(in_channels=args.in_channels, classes=args.classes).state_dict()
        source = f"UNet3D(in_channels={args.in_channels}, classes={args.classes})"
    else:
        if not args.checkpoint:
            parser.error("a checkpoint path is required unless --from-model is given")
        path = Path(args.checkpoint)
        if not path.is_file():
            raise SystemExit(f"no such checkpoint: {path}")
        state = load_state_dict(path)
        source = str(path)

    payload = inventory(state)
    json.dump(payload, sys.stdout, indent=1)
    sys.stdout.write("\n")
    print(f"{len(payload)} tensors from {source}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
