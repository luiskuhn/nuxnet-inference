from __future__ import annotations

from pathlib import Path
from typing import Iterable

import click
import numpy as np
import torch
from rich import print, traceback
from torch import nn

from nuxnet_inference_package.models.unet3d import UNet3D


class DummyNuxNet3D(nn.Module):
    """Minimal 3D segmentation network for CPU smoke testing."""

    def __init__(self, in_channels: int = 1, classes: int = 3, base_channels: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(base_channels, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def parse_shape(shape: str) -> tuple[int, int, int]:
    dims = tuple(int(v.strip()) for v in shape.split(","))
    if len(dims) != 3:
        raise click.ClickException("--shape must be D,H,W")
    return dims


def build_model(arch: str, in_channels: int, classes: int, dropout_rate: float) -> nn.Module:
    if arch == "dummy":
        return DummyNuxNet3D(in_channels=in_channels, classes=classes)
    if arch == "unet3d":
        return UNet3D(in_channels=in_channels, classes=classes, dropout=dropout_rate)
    raise click.ClickException(f"Unsupported architecture: {arch}")


def initialize_model(
    model_path: str | None,
    arch: str,
    in_channels: int,
    classes: int,
    dropout_rate: float,
    device: torch.device,
) -> nn.Module:
    model = build_model(arch=arch, in_channels=in_channels, classes=classes, dropout_rate=dropout_rate)
    if model_path:
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state))
        print(f"[bold green]Loaded model checkpoint: {model_path}")
    else:
        print("[bold yellow]No checkpoint provided; using random (untrained) model weights.")

    model.to(device)
    model.eval()
    return model


def read_input_or_dummy(file_path: Path | None, shape: str) -> np.ndarray:
    if file_path is None:
        volume = np.random.rand(*parse_shape(shape)).astype(np.float32)
        print(f"[bold yellow]No input file provided; generated dummy 3D input with shape {volume.shape}.")
    else:
        print(f"[bold yellow]Input: {file_path}")
        volume = np.load(file_path).astype(np.float32)

    if volume.ndim != 3:
        raise click.ClickException(f"Expected 3D volume with shape (D,H,W), got shape {volume.shape}")
    return volume


def predict_volume(volume: np.ndarray, model: nn.Module, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(volume[None, None, ...]).float().to(device)
    with torch.no_grad():
        logits = model(tensor)
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def iter_inputs(input_path: str | None) -> Iterable[Path | None]:
    if not input_path:
        yield None
        return

    path = Path(input_path)
    if path.is_dir():
        yield from sorted(path.glob("*.npy"))
    else:
        yield path


def output_prefix_for(input_file: Path | None, input_root: str | None, output: str) -> Path:
    output_path = Path(output)
    if input_file is None or not input_root:
        return output_path

    input_root_path = Path(input_root)
    if input_root_path.is_dir():
        relative = input_file.relative_to(input_root_path).with_suffix("")
        return output_path / relative

    return output_path


def write_results(prediction: np.ndarray, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_prefix, prediction)
    print(f"[bold green]Output: {output_prefix}.npy")


@click.command()
@click.option("-i", "--input", required=False, type=str, help="Path to a .npy file or directory of .npy files.")
@click.option("-m", "--model", required=False, type=str, help="Optional path to a .pt checkpoint (state_dict).")
@click.option("-o", "--output", default="predictions", required=True, type=str, help="Output path prefix or folder.")
@click.option("--shape", default="16,64,64", show_default=True, type=str, help="Dummy input shape as D,H,W when --input is omitted.")
@click.option("--arch", default="unet3d", show_default=True, type=click.Choice(["unet3d", "dummy"]), help="Inference model architecture.")
@click.option("--classes", default=3, show_default=True, type=int, help="Number of output classes.")
@click.option("--in-channels", default=1, show_default=True, type=int, help="Number of input channels expected by model.")
@click.option("--dropout-rate", default=0.25, show_default=True, type=float, help="Dropout used by UNet3D blocks.")
@click.option("--seed", default=42, show_default=True, type=int, help="RNG seed for dummy input and model init.")
@click.option("-c/-nc", "--cuda/--no-cuda", type=bool, default=False, help="Run on CUDA if available.")
def main(
    input: str | None,
    model: str | None,
    output: str,
    shape: str,
    arch: str,
    classes: int,
    in_channels: int,
    dropout_rate: float,
    seed: int,
    cuda: bool,
) -> None:
    """Run 3D segmentation inference (nuxnet-pred)."""
    print("[bold blue]nuxnet-pred")

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    model_3d = initialize_model(
        model_path=model,
        arch=arch,
        in_channels=in_channels,
        classes=classes,
        dropout_rate=dropout_rate,
        device=device,
    )

    for input_file in iter_inputs(input):
        output_prefix = output_prefix_for(input_file=input_file, input_root=input, output=output)
        volume = read_input_or_dummy(input_file, shape)
        prediction = predict_volume(volume, model_3d, device)
        write_results(prediction, output_prefix)


if __name__ == "__main__":
    traceback.install()
    main()  # pragma: no cover
