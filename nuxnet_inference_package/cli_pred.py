from __future__ import annotations

from pathlib import Path
from typing import Iterable

import click
import networkx as nx
import numpy as np
import torch
from rich import print, traceback
from scipy.spatial import KDTree
from skimage import exposure, measure
from tifffile import TiffFile, imwrite
from torch import nn

from nuxnet_inference_package.mlf_core import configure_reproducibility
from nuxnet_inference_package.models.unet3d import UNet3D


def extract_nuclei_instances(
    prediction: np.ndarray,
    nuclei_label: int,
    radius: float = 2.0,
) -> list[dict[str, float | int]]:
    """Cluster nuclei voxels via KDTree-neighborhood graph and return centroid/size per cluster."""
    if prediction.ndim != 3:
        raise click.ClickException(f"Expected prediction with shape (D,H,W), got shape {prediction.shape}")

    nuclei_mask = prediction == nuclei_label
    labeled_mask = measure.label(nuclei_mask.astype(np.uint8), background=0, connectivity=1)
    foreground_coords = np.argwhere(labeled_mask > 0)

    if foreground_coords.size == 0:
        return []

    tree = KDTree(foreground_coords)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(foreground_coords)))

    for node_index, coord in enumerate(foreground_coords):
        neighbors = tree.query_ball_point(coord, r=radius)
        for neighbor_index in neighbors:
            if neighbor_index == node_index:
                continue
            graph.add_edge(node_index, neighbor_index)

    components = sorted(nx.strongly_connected_components(graph), key=lambda comp: min(comp))
    nuclei_instances = []

    for nuclei_id, component in enumerate(components, start=1):
        component_indices = np.fromiter(component, dtype=int)
        component_coords = foreground_coords[component_indices]
        centroid_z, centroid_y, centroid_x = component_coords.mean(axis=0)
        nuclei_instances.append(
            {
                "id": nuclei_id,
                "z": float(centroid_z),
                "y": float(centroid_y),
                "x": float(centroid_x),
                "size_voxels": int(component_coords.shape[0]),
            }
        )

    return nuclei_instances


def write_instances_tsv(instances: list[dict[str, float | int]], output_prefix: Path) -> Path:
    output_path = output_prefix.with_suffix(".nuclei.tsv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("id\tz\ty\tx\tsize_voxels\n")
        for instance in instances:
            file.write(
                f"{instance['id']}\t{instance['z']:.3f}\t{instance['y']:.3f}\t{instance['x']:.3f}\t{instance['size_voxels']}\n"
            )

    print(f"[bold green]Nuclei table: {output_path}")
    return output_path


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
        raise click.ClickException("Shape must be formatted as D,H,W")
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


def _volume_from_ome_tiff(file_path: Path) -> np.ndarray:
    with TiffFile(file_path) as tif:
        series = tif.series[0]
        volume = series.asarray()
        axes = series.axes

    if not set("ZYX").issubset(set(axes)):
        raise click.ClickException(
            f"Expected OME-TIFF axes to include ZYX, got axes '{axes}' for shape {volume.shape}."
        )

    axis_index = {axis: idx for idx, axis in enumerate(axes)}
    order = [axis_index["Z"], axis_index["Y"], axis_index["X"]] + [
        idx for idx, axis in enumerate(axes) if axis not in {"Z", "Y", "X"}
    ]
    volume = np.transpose(volume, order)

    if volume.ndim > 3:
        extra_shape = volume.shape[3:]
        if any(size != 1 for size in extra_shape):
            raise click.ClickException(
                "Expected singleton dimensions for non-ZYX axes in OME-TIFF input, "
                f"got shape {volume.shape} after reordering axes from '{axes}'."
            )
        volume = np.squeeze(volume, axis=tuple(range(3, volume.ndim)))

    return volume.astype(np.float32)


def read_input_or_dummy(file_path: Path | None, shape: str) -> np.ndarray:
    if file_path is None:
        volume = np.random.rand(*parse_shape(shape)).astype(np.float32)
        print(f"[bold yellow]Generated random 3D input with shape {volume.shape}.")
    else:
        print(f"[bold yellow]Input: {file_path}")
        volume = _volume_from_ome_tiff(file_path)

    if volume.ndim != 3:
        raise click.ClickException(f"Expected 3D volume with shape (D,H,W), got shape {volume.shape}")
    return volume


def predict_volume(
    volume: np.ndarray,
    model: nn.Module,
    device: torch.device,
    normalize_input: bool,
) -> np.ndarray:
    if normalize_input:
        model_input = exposure.rescale_intensity(
            volume,
            in_range=(float(volume.min()), float(volume.max())),
            out_range=(0.0, 1.0),
        ).astype(np.float32)
    else:
        model_input = volume.astype(np.float32)

    tensor = torch.from_numpy(model_input[None, None, ...]).float().to(device)
    with torch.no_grad():
        logits = model(tensor)
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def iter_inputs(input_path: str | None) -> Iterable[Path | None]:
    if not input_path:
        yield None
        return

    path = Path(input_path)
    if path.is_dir():
        patterns = ("*.ome.tiff", "*.ome.tif", "*.tiff", "*.tif")
        files = []
        for pattern in patterns:
            files.extend(path.glob(pattern))
        yield from sorted(set(files))
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


def write_mask_ome_tiff(prediction: np.ndarray, output_prefix: Path) -> Path:
    """Write segmentation mask as OME-TIFF with ZYX axis metadata."""
    output_path = output_prefix.with_suffix(".ome.tiff")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(output_path, prediction, ome=True, metadata={"axes": "ZYX"})
    print(f"[bold green]Output: {output_path}")
    return output_path


def run_inference(
    input_path: str | None,
    model_path: str | None,
    output: str,
    input_shape: str,
    arch: str,
    classes: int,
    in_channels: int,
    dropout_rate: float,
    seed: int,
    cuda: bool,
    deterministic: bool,
    normalize_input: bool,
    postprocess_instances: bool,
    nuclei_label: int,
    neighbor_radius: float,
) -> None:
    print("[bold blue]nuxnet-pred")
    configure_reproducibility(seed=seed, deterministic=deterministic)
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    print(f"[bold cyan]Deterministic mode:[/bold cyan] {'ON' if deterministic else 'OFF'}")
    print(f"[bold cyan]Device:[/bold cyan] {device}")

    model_3d = initialize_model(
        model_path=model_path,
        arch=arch,
        in_channels=in_channels,
        classes=classes,
        dropout_rate=dropout_rate,
        device=device,
    )

    for input_file in iter_inputs(input_path):
        output_prefix = output_prefix_for(input_file=input_file, input_root=input_path, output=output)
        volume = read_input_or_dummy(input_file, input_shape)
        prediction = predict_volume(volume, model_3d, device, normalize_input=normalize_input)
        write_mask_ome_tiff(prediction, output_prefix)
        if postprocess_instances:
            instances = extract_nuclei_instances(
                prediction=prediction,
                nuclei_label=nuclei_label,
                radius=neighbor_radius,
            )
            write_instances_tsv(instances=instances, output_prefix=output_prefix)


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """nuxnet-pred CLI for 3D segmentation inference and smoke testing."""
    if ctx.invoked_subcommand is None:
        # Backward-compatible default behavior.
        ctx.invoke(
            predict,
            input_path=None,
            model_path=None,
            output="predictions",
            input_shape="16,64,64",
            arch="unet3d",
            classes=3,
            in_channels=1,
            dropout_rate=0.25,
            seed=42,
            cuda=False,
            normalize_input=True,
            postprocess_instances=False,
            nuclei_label=1,
            neighbor_radius=2.0,
            deterministic=False,
        )


@main.command("predict")
@click.option(
    "--input",
    "input_path",
    required=False,
    type=str,
    help="Path to an OME-TIFF volume (.ome.tiff/.ome.tif) or a folder containing OME-TIFF files.",
)
@click.option("--model", "model_path", required=False, type=str, help="Optional .pt checkpoint to load. If omitted, model is randomly initialized.")
@click.option("--output", default="predictions", show_default=True, required=True, type=str, help="Output prefix or output folder for .ome.tiff mask files.")
@click.option("--input-shape", default="16,64,64", show_default=True, type=str, help="Shape D,H,W used only when --input is omitted.")
@click.option("--arch", default="unet3d", show_default=True, type=click.Choice(["unet3d", "dummy"]), help="Network architecture used for inference.")
@click.option("--classes", default=3, show_default=True, type=int, help="Number of output segmentation classes.")
@click.option("--in-channels", default=1, show_default=True, type=int, help="Number of input channels expected by the model.")
@click.option("--dropout-rate", default=0.25, show_default=True, type=float, help="Dropout rate used by UNet3D blocks.")
@click.option("--seed", default=42, show_default=True, type=int, help="Random seed for dummy input generation and model init.")
@click.option("--cuda/--no-cuda", default=False, help="Use CUDA when available.")
@click.option(
    "--deterministic/--no-deterministic",
    default=False,
    show_default=True,
    help="Enable mlf-core-style deterministic execution settings (may reduce performance).",
)
@click.option(
    "--normalize-input/--no-normalize-input",
    default=True,
    show_default=True,
    help="Enable/disable skimage.exposure.rescale_intensity normalization before model inference.",
)
@click.option(
    "--postprocess-instances/--no-postprocess-instances",
    default=False,
    show_default=True,
    help="Run KDTree+NetworkX graph clustering (strongly connected components) and export nuclei TSV.",
)
@click.option(
    "--nuclei-label",
    default=1,
    show_default=True,
    type=int,
    help="Class label in the predicted mask interpreted as nuclei for instance extraction.",
)
@click.option(
    "--neighbor-radius",
    default=2.0,
    show_default=True,
    type=float,
    help="Radius used to create dense voxel neighborhoods with KDTree for graph clustering.",
)
def predict(
    input_path: str | None,
    model_path: str | None,
    output: str,
    input_shape: str,
    arch: str,
    classes: int,
    in_channels: int,
    dropout_rate: float,
    seed: int,
    cuda: bool,
    deterministic: bool,
    normalize_input: bool,
    postprocess_instances: bool,
    nuclei_label: int,
    neighbor_radius: float,
) -> None:
    """Run inference from provided input volumes or generated random input."""
    run_inference(
        input_path=input_path,
        model_path=model_path,
        output=output,
        input_shape=input_shape,
        arch=arch,
        classes=classes,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        seed=seed,
        cuda=cuda,
        deterministic=deterministic,
        normalize_input=normalize_input,
        postprocess_instances=postprocess_instances,
        nuclei_label=nuclei_label,
        neighbor_radius=neighbor_radius,
    )


@main.command("smoke-test")
@click.option("--output", default="smoke_test_mask", show_default=True, type=str, help="Output prefix for generated .ome.tiff mask.")
@click.option("--input-shape", default="16,64,64", show_default=True, type=str, help="Random 3D input shape formatted as D,H,W.")
@click.option("--arch", default="unet3d", show_default=True, type=click.Choice(["unet3d", "dummy"]), help="Architecture used for the untrained forward pass.")
@click.option("--classes", default=3, show_default=True, type=int, help="Number of output classes.")
@click.option("--in-channels", default=1, show_default=True, type=int, help="Number of model input channels.")
@click.option("--dropout-rate", default=0.25, show_default=True, type=float, help="Dropout rate used by UNet3D blocks.")
@click.option("--seed", default=42, show_default=True, type=int, help="Seed for random input/model initialization.")
@click.option("--cuda/--no-cuda", default=False, help="Use CUDA when available.")
@click.option(
    "--deterministic/--no-deterministic",
    default=False,
    show_default=True,
    help="Enable mlf-core-style deterministic execution settings (may reduce performance).",
)
@click.option(
    "--normalize-input/--no-normalize-input",
    default=True,
    show_default=True,
    help="Enable/disable skimage.exposure.rescale_intensity normalization before model inference.",
)
def smoke_test(
    output: str,
    input_shape: str,
    arch: str,
    classes: int,
    in_channels: int,
    dropout_rate: float,
    seed: int,
    cuda: bool,
    deterministic: bool,
    normalize_input: bool,
) -> None:
    """Generate random 3D input, run untrained model, and write OME-TIFF mask."""
    run_inference(
        input_path=None,
        model_path=None,
        output=output,
        input_shape=input_shape,
        arch=arch,
        classes=classes,
        in_channels=in_channels,
        dropout_rate=dropout_rate,
        seed=seed,
        cuda=cuda,
        deterministic=deterministic,
        postprocess_instances=False,
        nuclei_label=1,
        neighbor_radius=2.0,
        normalize_input=normalize_input,
    )


if __name__ == "__main__":
    traceback.install()
    main()  # pragma: no cover
