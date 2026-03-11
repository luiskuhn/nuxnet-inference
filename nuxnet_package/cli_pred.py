import glob
import os
import pathlib
import sys

import click
import numpy as np
import torch
from rich import print, traceback
from torch import nn


class DummyNuxNet3D(nn.Module):
    """Minimal 3D segmentation network for CPU smoke testing."""

    def __init__(self, in_channels: int = 1, classes: int = 3, base_channels: int = 8):
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


@click.command()
@click.option('-i', '--input', required=False, type=str, help='Path to a .npy file or a directory of .npy files. If omitted, a random 3D volume is generated.')
@click.option('-m', '--model', required=False, type=str, help='Optional path to a .pt checkpoint (state_dict) for DummyNuxNet3D. If omitted, random weights are used.')
@click.option('-o', '--output', default='predictions', required=True, type=str, help='Path prefix or output folder.')
@click.option('--shape', default='16,64,64', show_default=True, type=str, help='Dummy input shape as D,H,W when --input is omitted.')
@click.option('--classes', default=3, show_default=True, type=int, help='Number of output classes.')
@click.option('--seed', default=42, show_default=True, type=int, help='RNG seed used for dummy input and random model init.')
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to run on CUDA if available.')
def main(input: str, model: str, output: str, shape: str, classes: int, seed: int, cuda: bool):
    """Command-line interface for 3D segmentation inference (nuxnet-pred)."""
    print('[bold blue]nuxnet-pred (3D scaffold mode)')
    print('[bold blue]Run [green]nuxnet-pred --help [blue]for an overview of all commands\n')

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    model_3d = initialize_model(model, classes=classes, device=device)

    if input and os.path.isdir(input):
        for file_path in sorted(glob.glob(os.path.join(input, '*.npy'))):
            out_path = file_path.replace(input, output).replace('.npy', '')
            run_single_prediction(file_path=file_path, model=model_3d, output_prefix=out_path, shape=shape, classes=classes, device=device)
    else:
        run_single_prediction(file_path=input, model=model_3d, output_prefix=output, shape=shape, classes=classes, device=device)


def initialize_model(model_path, classes: int, device: torch.device) -> nn.Module:
    model_3d = DummyNuxNet3D(classes=classes)
    if model_path:
        state = torch.load(model_path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            model_3d.load_state_dict(state['state_dict'])
        else:
            model_3d.load_state_dict(state)
        print(f'[bold green]Loaded model checkpoint: {model_path}')
    else:
        print('[bold yellow]No checkpoint provided; using randomly initialized (untrained) model weights.')
    model_3d.to(device)
    model_3d.eval()
    return model_3d


def run_single_prediction(file_path, model: nn.Module, output_prefix: str, shape: str, classes: int, device: torch.device) -> None:
    volume = read_input_or_dummy(file_path, shape, classes)
    prediction = predict_volume(volume, model, device)
    write_results(prediction, output_prefix)
    print(f'[bold green]Output: {output_prefix}.npy')


def read_input_or_dummy(file_path, shape: str, classes: int) -> np.ndarray:
    if file_path:
        print(f'[bold yellow]Input: {file_path}')
        volume = np.load(file_path)
    else:
        dims = tuple(int(v.strip()) for v in shape.split(','))
        if len(dims) != 3:
            raise click.ClickException('--shape must be D,H,W')
        volume = np.random.rand(*dims).astype(np.float32)
        print(f'[bold yellow]No input file provided; generated dummy 3D input with shape {volume.shape}.')

    if volume.ndim != 3:
        raise click.ClickException(f'Expected 3D volume with shape (D,H,W), got shape {volume.shape}')

    return volume.astype(np.float32)


def predict_volume(volume: np.ndarray, model: nn.Module, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(volume[None, None, ...]).float().to(device)
    with torch.no_grad():
        logits = model(tensor)
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return prediction


def write_results(predictions: np.ndarray, path_to_write_to: str) -> None:
    os.makedirs(pathlib.Path(path_to_write_to).parent.absolute(), exist_ok=True)
    np.save(path_to_write_to, predictions)


# -----------------------------------------------------------------------------
# Legacy code intentionally kept as comments for later restoration.
# The old workflow loaded TIFF input and fetched a pretrained checkpoint from
# Zenodo. Keep this as reference once production model I/O is re-enabled.
#
# def read_data_to_predict(path_to_data_to_predict: str):
#     return tiff.imread(path_to_data_to_predict)
#
# def download(filepath) -> None:
#     if _check_exists(filepath):
#         return
#     mirrors = ['https://zenodo.org/record/']
#     resources = [
#         (
#             'mark1-PHDFM-u2net-model.ckpt',
#             '6937290/files/mark1-PHDFM-u2net-model.ckpt',
#             '5dd5d425afb4b17444cb31b1343f23dc',
#         ),
#     ]
#     ...
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    traceback.install()
    sys.exit(main())  # pragma: no cover
