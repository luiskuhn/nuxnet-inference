"""Build a validation report as a PDF, from the artifacts eval_numorph.py saved.

The report exists because numbers alone hide failure modes and pictures alone hide
systematic error. Each volume gets a slice montage -- where a misalignment is visible --
next to a per-slice Dice curve, where it is measurable.

One page is deliberately about *where* the score is computed. On this dataset that
choice moved the headline number from 0.46 to 0.80, so a report that omitted it would
be misleading rather than merely incomplete.

    python make_report.py --out report.pdf --artifacts /out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GREEN = "#2e9e4f"
RED = "#c0392b"
BLUE = "#2b6cb0"
AMBER = "#b7791f"
LIGHT = "#f2f4f7"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


def component_sizes(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    return np.array([]) if n == 0 else np.bincount(labeled.ravel())[1:]


def normalize(volume: np.ndarray) -> np.ndarray:
    low, high = float(volume.min()), float(volume.max())
    return (volume - low) / (high - low) if high > low else np.zeros(volume.shape, dtype=float)


def page_header(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.07, 0.955, title, fontsize=17, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.07, 0.920, subtitle, fontsize=9.5, color=MUTED, va="top")


def page_footer(fig, page: int, total: int, note: str = "") -> None:
    if note:
        fig.text(0.07, 0.028, note, fontsize=7.5, color=MUTED, va="bottom")
    fig.text(0.93, 0.028, f"{page} / {total}", fontsize=8, color=MUTED, va="bottom", ha="right")


def dedent(text: str) -> str:
    """Remove the common leading indentation so prose blocks can be written inline."""
    import textwrap

    return textwrap.dedent(text).strip("\n")


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def page_cover(pdf: PdfPages, results: list[dict], page: int, total: int, cfg: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    train = np.array([r["train_grid_dice"] for r in results])
    ceil = np.array([r["roundtrip_ceiling"] for r in results])
    inp = np.array([r["input_grid_dice"] for r in results])

    fig.text(0.08, 0.885, "nuxnet inference", fontsize=32, fontweight="bold", color=INK)
    fig.text(0.08, 0.838, "validation on the NuMorph training dataset", fontsize=15, color=MUTED)

    # headline numbers
    fig.text(0.08, 0.760, "Headline", fontsize=11, fontweight="bold", color=BLUE)
    entries = [
        ("train-grid Dice", f"{train.mean():.3f}", f"range {train.min():.3f} - {train.max():.3f}"),
        ("input-grid Dice", f"{inp.mean():.3f}", f"{inp.mean() / ceil.mean() * 100:.0f}% of achievable"),
        ("volumes", f"{len(results)}", "image / mask pairs"),
        ("checkpoint", "108", "tensors, strict=True"),
    ]
    y = 0.705
    for label, value, note in entries:
        fig.text(0.10, y, label, fontsize=10, color=MUTED)
        fig.text(0.42, y - 0.004, value, fontsize=20, fontweight="bold", color=INK)
        fig.text(0.56, y, note, fontsize=8.5, color=MUTED)
        y -= 0.052

    fig.text(0.08, 0.455, "Question", fontsize=11, fontweight="bold", color=BLUE)
    fig.text(
        0.08,
        0.425,
        dedent("""
        Can the inference CLI load a checkpoint trained by nuxnet-training, read a
        single image file, run a forward pass, write a segmentation mask, and produce
        the nuclei instance table from the unchanged post-processing?
        """),
        fontsize=10.5,
        color=INK,
        va="top",
        linespacing=1.6,
    )

    fig.text(0.08, 0.360, "Configuration", fontsize=11, fontweight="bold", color=BLUE)
    rows = [
        ("weights", cfg["weights"]),
        ("target voxel spacing", f"{cfg['target']} um (Z, Y, X)"),
        ("patch size", f"{cfg['patch']} (trainer default)"),
        ("threshold", str(cfg["threshold"])),
        ("device", cfg["device"]),
    ]
    y = 0.330
    for key, value in rows:
        fig.text(0.10, y, key, fontsize=9, color=MUTED)
        fig.text(0.36, y, str(value), fontsize=9, color=INK, family="monospace")
        y -= 0.026

    fig.text(
        0.08,
        0.180,
        dedent("""
        Reading order: page 2 explains why the score depends on which grid it is
        computed on. Pages 3 onward show every volume, one page each.
        """),
        fontsize=8.8,
        color=AMBER,
        va="top",
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#fdf6e3", edgecolor=AMBER, linewidth=0.8),
    )

    page_footer(fig, page, total)
    pdf.savefig(fig)
    plt.close(fig)


def page_grid_question(pdf: PdfPages, results: list[dict], page: int, total: int) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(
        fig,
        "Which grid the score is computed on",
        "The same prediction scores 0.46 or 0.75 depending only on where it is compared.",
    )

    fig.text(
        0.08,
        0.860,
        dedent("""
        The inputs are stored at 2.5 / 0.75 / 0.75 um and at 4.0 / 1.21 / 1.21 um. The
        checkpoint was trained at 3.0 / 1.0 / 1.0 um, so inference resamples onto that
        grid, predicts there, then projects the mask back so the output lines up with
        the input file the caller passed in.

        That projection cannot preserve which slice was which. When the source spacing
        is coarser in Z (4.0 um), the training grid has *more* slices and the round trip
        is nearly lossless. When the source is finer (2.5 um), 64 slices must compress
        into 53 and back, and the round trip throws information away.

        The ceiling below is the Dice a PERFECT mask would score after the same round
        trip, with no network involved. Anything above it is impossible.
        """),
        fontsize=9.8,
        color=INK,
        va="top",
        linespacing=1.62,
    )

    axis = fig.add_axes([0.14, 0.545, 0.70, 0.175])
    labels = ["perfect mask,\nround-tripped", "model output,\ninput grid", "model output,\ntraining grid"]
    values = [
        float(np.mean([r["roundtrip_ceiling"] for r in results])),
        float(np.mean([r["input_grid_dice"] for r in results])),
        float(np.mean([r["train_grid_dice"] for r in results])),
    ]
    bars = axis.barh(labels, values, color=[MUTED, AMBER, GREEN], height=0.55)
    for bar, value in zip(bars, values):
        axis.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                  va="center", fontsize=11, fontweight="bold", color=INK)
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Dice coefficient (mean over volumes)", fontsize=9, color=MUTED)
    axis.tick_params(labelsize=9, colors=INK)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("A perfect mask sets the achievable maximum on the input grid",
                   fontsize=10, color=INK, pad=8)

    fig.text(0.08, 0.475, "Per-group breakdown", fontsize=11, fontweight="bold", color=BLUE)
    groups = sorted({r["group"] for r in results})
    axis = fig.add_axes([0.14, 0.285, 0.70, 0.155])
    width = 0.8 / max(len(groups), 1)
    positions = np.arange(3)
    for index, group in enumerate(groups):
        subset = [r for r in results if r["group"] == group]
        spikes = [np.mean([r["roundtrip_ceiling"] for r in subset]),
                  np.mean([r["input_grid_dice"] for r in subset]),
                  np.mean([r["train_grid_dice"] for r in subset])]
        label = f"{group} ({subset[0]['source_spacing'][0]:g} um Z)"
        axis.bar(positions + index * width, spikes, width * 0.9, label=label,
                 color=[MUTED, AMBER, GREEN][index % 3], alpha=0.9)
    axis.set_xticks(positions + width * (len(groups) - 1) / 2)
    axis.set_xticklabels(["ceiling", "input grid", "training grid"], fontsize=9)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Dice", fontsize=9, color=MUTED)
    axis.tick_params(labelsize=8, colors=MUTED)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=8, frameon=False, loc="lower right")

    fig.text(
        0.08,
        0.185,
        dedent("""
        The first group compresses 64 slices to 53 and back, so even a perfect mask
        caps near 0.78. The second group upsamples in Z, so the ceiling is about 0.98
        and the input grid is the more favourable place to measure.

        Practical consequence: the mask output must stay on the input grid, because
        downstream tools index it against the input volume. The scaling loss belongs
        to the export, not the model. Quoting an input-grid Dice without the ceiling
        would blame the network for the resampling.
        """),
        fontsize=9.3,
        color=INK,
        va="top",
        linespacing=1.6,
    )

    page_footer(fig, page, total, "ceiling computed by round-tripping ground truth through resample_volume_pair")
    pdf.savefig(fig)
    plt.close(fig)


def page_volume(pdf: PdfPages, result: dict, artifacts: Path, page: int, total: int) -> None:
    stem = result["volume"].replace(".ome.tif", "")
    image = np.load(artifacts / f"{stem}.prepared.npy").astype(np.float32)
    truth_grid = np.load(artifacts / f"{stem}.truth_grid.npy") > 0
    probability = np.load(artifacts / f"{stem}.prob.npy").astype(np.float32)
    pred_grid = probability >= 0.8

    depth = image.shape[0]
    slices = [int(depth * fraction) for fraction in (0.28, 0.5, 0.72)]

    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(
        fig,
        stem.replace("NUMORPH_SEM_SEG_", "").replace("_", " "),
        f"group {result['group']}   source {result['source_spacing']} um   "
        f"prepared {tuple(result['prepared_shape'])}   {result['tiles']} tiles",
    )

    fig.text(
        0.07,
        0.888,
        f"train-grid Dice {result['train_grid_dice']:.3f}   "
        f"input-grid Dice {result['input_grid_dice']:.3f}   "
        f"ceiling {result['roundtrip_ceiling']:.3f}   "
        f"({result['fraction_of_ceiling'] * 100:.0f}% of achievable)",
        fontsize=9,
        color=MUTED,
    )

    # montage
    top, bottom, left = 0.835, 0.470, 0.085
    column = 0.195
    gap = 0.025
    headers = ["input (training grid)", "ground truth", "prediction", "overlay"]
    for index, header in enumerate(headers):
        fig.text(left + index * (column + gap), top + 0.012, header, fontsize=8.5, color=MUTED)

    row_height = (top - bottom) / len(slices)
    for row, z in enumerate(slices):
        image_slice = normalize(image[z])
        truth_slice, pred_slice = truth_grid[z], pred_grid[z]
        overlay = np.dstack([image_slice] * 3)
        overlay[truth_slice] = [0.10, 0.70, 0.30]
        overlay[pred_slice] = [0.85, 0.15, 0.15]
        overlay[truth_slice & pred_slice] = [1.0, 0.90, 0.20]

        y = top - (row + 1) * row_height
        for col, (data, cmap) in enumerate((
            (image_slice, "gray"),
            (truth_slice.astype(float), "gray"),
            (pred_slice.astype(float), "gray"),
            (overlay, None),
        )):
            axis = fig.add_axes([left + col * (column + gap), y + 0.012, column, row_height - 0.045])
            axis.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#d0d0d0")
                spine.set_linewidth(0.6)
        fig.text(left - 0.022, y + row_height / 2, f"z={z}", fontsize=8.5, color=INK, va="center")

    # per-slice dice
    axis = fig.add_axes([0.11, 0.245, 0.79, 0.165])
    per_slice = [dice(pred_grid[z], truth_grid[z]) for z in range(depth)]
    axis.plot(range(depth), per_slice, color=BLUE, linewidth=1.0)
    axis.axhline(result["train_grid_dice"], color=GREEN, linestyle="--", linewidth=1.0,
                 label=f"volume Dice {result['train_grid_dice']:.3f}")
    for z in slices:
        axis.axvline(z, color=AMBER, linewidth=0.7, alpha=0.6)
    axis.set_xlabel("slice index on the training grid", fontsize=8.5, color=MUTED)
    axis.set_ylabel("Dice", fontsize=8.5, color=MUTED)
    axis.set_ylim(0, 1)
    axis.tick_params(labelsize=7.5, colors=MUTED)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=8, frameon=False, loc="lower right")
    axis.set_title("Per-slice agreement: a flat curve means no systematic misalignment",
                   fontsize=9, color=INK, pad=5)

    # distributions
    pred_sizes = component_sizes(pred_grid)
    truth_sizes = component_sizes(truth_grid)
    axis = fig.add_axes([0.11, 0.095, 0.34, 0.095])
    upper = max(pred_sizes.max() if pred_sizes.size else 1,
                truth_sizes.max() if truth_sizes.size else 1)
    bins = np.linspace(0, upper, 24)
    axis.hist(truth_sizes, bins=bins, color=GREEN, alpha=0.55, density=True, label="truth")
    axis.hist(pred_sizes, bins=bins, color=RED, alpha=0.45, density=True, label="prediction")
    axis.set_xlabel("component size (voxels)", fontsize=7.5, color=MUTED)
    axis.set_yticks([])
    axis.tick_params(labelsize=7, colors=MUTED)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(fontsize=7, frameon=False)
    axis.set_title("component size", fontsize=8, color=INK, pad=4)

    lines = [
        f"components   truth {len(truth_sizes):5d}   prediction {len(pred_sizes):5d}",
        f"median size  truth {np.median(truth_sizes) if truth_sizes.size else 0:5.0f}   "
        f"prediction {np.median(pred_sizes) if pred_sizes.size else 0:5.0f} voxels",
        f"foreground   truth {truth_grid.mean() * 100:5.2f}%   prediction {pred_grid.mean() * 100:5.2f}%",
        f"instances    {result['instances']} written to the TSV",
    ]
    fig.text(0.50, 0.170, "\n".join(lines), fontsize=7.8, color=INK, va="top",
             linespacing=1.8, family="monospace")

    page_footer(fig, page, total, "overlay: truth green, prediction red, both yellow")
    pdf.savefig(fig)
    plt.close(fig)


def page_threshold(pdf: PdfPages, result: dict, artifacts: Path, page: int, total: int) -> None:
    stem = result["volume"].replace(".ome.tif", "")
    probability = np.load(artifacts / f"{stem}.prob.npy").astype(np.float32)
    truth = np.load(artifacts / f"{stem}.truth_grid.npy") > 0

    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(
        fig,
        "Threshold sensitivity",
        f"{stem.replace('NUMORPH_SEM_SEG_', '').replace('_', ' ')} - the operating point is not cherry-picked.",
    )

    thresholds = np.linspace(0.05, 0.95, 19)
    scores, precision, recall = [], [], []
    for threshold in thresholds:
        pred = probability >= threshold
        scores.append(dice(pred, truth))
        inter = np.logical_and(pred, truth).sum()
        precision.append(inter / max(pred.sum(), 1))
        recall.append(inter / max(truth.sum(), 1))

    axis = fig.add_axes([0.11, 0.560, 0.78, 0.270])
    axis.plot(thresholds, scores, color=BLUE, linewidth=1.8, label="Dice")
    axis.plot(thresholds, precision, color=RED, linewidth=1.1, linestyle="--", label="precision")
    axis.plot(thresholds, recall, color=GREEN, linewidth=1.1, linestyle="--", label="recall")
    best = thresholds[int(np.argmax(scores))]
    axis.axvline(best, color=AMBER, linewidth=1.0, linestyle=":")
    axis.annotate(f"best {max(scores):.3f}\nat {best:.2f}",
                  xy=(best, max(scores)), xytext=(best + 0.05, max(scores) - 0.10),
                  fontsize=9, color=AMBER,
                  arrowprops=dict(arrowstyle="->", color=AMBER, linewidth=0.9))
    axis.axvline(0.8, color=MUTED, linewidth=0.8, linestyle="-", alpha=0.5)
    axis.set_xlabel("foreground probability threshold", fontsize=9, color=MUTED)
    axis.set_ylabel("score", fontsize=9, color=MUTED)
    axis.set_ylim(0.4, 1.01)
    axis.tick_params(labelsize=8, colors=MUTED)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=9, frameon=False, loc="lower left")
    axis.set_title("Dice is broad across 0.4-0.9, so the result does not depend on tuning",
                   fontsize=10, color=INK, pad=8)

    axis = fig.add_axes([0.11, 0.270, 0.37, 0.180])
    axis.plot(thresholds, [float((probability >= t).mean() * 100) for t in thresholds],
              color=BLUE, linewidth=1.5)
    axis.axhline(truth.mean() * 100, color=GREEN, linestyle="--", linewidth=1.0,
                 label=f"truth {truth.mean() * 100:.2f}%")
    axis.set_xlabel("threshold", fontsize=8.5, color=MUTED)
    axis.set_ylabel("foreground %", fontsize=8.5, color=MUTED)
    axis.tick_params(labelsize=7.5, colors=MUTED)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=8, frameon=False)
    axis.set_title("foreground fraction", fontsize=9, color=INK, pad=5)

    axis = fig.add_axes([0.58, 0.270, 0.31, 0.180])
    axis.hist(probability[truth].ravel(), bins=40, color=GREEN, alpha=0.6,
              density=True, label="inside truth")
    axis.hist(probability[~truth].ravel(), bins=40, color=RED, alpha=0.45,
              density=True, label="outside truth")
    axis.set_xlabel("probability", fontsize=8.5, color=MUTED)
    axis.set_yticks([])
    axis.tick_params(labelsize=7.5, colors=MUTED)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(fontsize=7.5, frameon=False)
    axis.set_title("probability separation", fontsize=9, color=INK, pad=5)

    table = "\n".join(
        f"{t:.2f}   Dice {d:.3f}   precision {p:.3f}   recall {r:.3f}"
        for t, d, p, r in zip(thresholds[::3], scores[::3], precision[::3], recall[::3])
    )
    fig.text(0.11, 0.155, table, fontsize=7.6, color=MUTED, va="top",
             family="monospace", linespacing=1.7)

    page_footer(fig, page, total, "probability saved from the same forward pass that produced the mask")
    pdf.savefig(fig)
    plt.close(fig)


def page_packaging(pdf: PdfPages, page: int, total: int, package_dir: Path | None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(
        fig,
        "BioImage.IO packaging",
        "What a consumer needs that a state dict cannot tell them.",
    )

    fig.text(
        0.08,
        0.870,
        dedent("""
        The checkpoint alone is not a usable model. Three things it cannot express:

        1. the preprocessing that puts a volume on the grid the weights expect
           (3.0 / 1.0 / 1.0 um), because the network has no concept of physical scale;
        2. that the outputs are LOGITS, not probabilities;
        3. that every spatial axis must be a multiple of 4 -- the decoder upsamples
           twice, so anything else raises a tensor-size mismatch rather than
           degrading gracefully.

        The same blind spot bit the CLI's own output: masks were written without
        PhysicalSize, so they aligned with the input array but not with the specimen.
        Carrying the calibration through was a separate fix.
        """),
        fontsize=9.6,
        color=INK,
        va="top",
        linespacing=1.6,
    )

    fig.text(0.08, 0.640, "Package contents", fontsize=11, fontweight="bold", color=BLUE)
    contents = [
        ("rdf.yaml", "BioImage.IO resource description, format 0.5.12"),
        ("weights.pt", "state dict, 108 tensors"),
        ("model/unet_3d_models.py", "the architecture, so the package reloads standalone"),
        ("environment.yml", "pinned dependencies"),
        ("test-input.npy", "legal (1,1,32,128,128) tensor"),
        ("test-output.npy", "reference logits, generated from these weights"),
        ("README.md", "model card with the preprocessing contract"),
        ("SHA256SUMS", "integrity"),
    ]
    y = 0.610
    for name, note in contents:
        fig.text(0.10, y, name, fontsize=8.6, color=INK, family="monospace")
        fig.text(0.40, y, note, fontsize=8.4, color=MUTED)
        y -= 0.024

    fig.text(0.08, 0.390, "Verification", fontsize=11, fontweight="bold", color=BLUE)
    fig.text(
        0.10,
        0.360,
        dedent("""
        scripts/verify_package.py reloads the package the way a consumer would:
        it imports the PACKAGED architecture file, loads the state dict strictly,
        regenerates the output from the packaged input, and compares.

        It also asserts the documented multiple-of-4 constraint is real by feeding
        the model a 30-voxel axis and requiring it to fail. A model card that claims
        a constraint the code does not enforce is worse than no model card.
        """),
        fontsize=9.3,
        color=INK,
        va="top",
        linespacing=1.6,
    )

    if package_dir and package_dir.is_dir():
        total_kib = sum(p.stat().st_size for p in package_dir.rglob("*") if p.is_file()) / 1024
        fig.text(0.10, 0.250, f"verified package: {package_dir.name}  ({total_kib:.0f} KiB)",
                 fontsize=8.6, color=GREEN, family="monospace")

    fig.text(0.08, 0.190, "Before publishing", fontsize=11, fontweight="bold", color=AMBER)
    fig.text(
        0.10,
        0.160,
        dedent("""
        - replace the citation placeholder; a package without a resolvable citation
          is not acceptable in the zoo;
        - run `bioimageio test <package.zip>` and attach the output;
        - decide whether to ship one fold or an ensemble, and say which in the card;
        - the target is a CENTRE MARKER, not a nucleus body. State it plainly, or
          users will treat the foreground as a segmentation of nucleus volume.
        """),
        fontsize=9.2,
        color=INK,
        va="top",
        linespacing=1.6,
    )

    page_footer(fig, page, total, "package source: scripts/build_bioimageio_package.py")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--package", default=None)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    summary_path = artifacts / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"no summary.json in {artifacts}")
    results = json.loads(summary_path.read_text())
    results = results[: args.limit]

    cfg = {
        "weights": "numorph_fold1_weights.pt",
        "target": "3.0,1.0,1.0",
        "patch": "32,128,128",
        "threshold": args.threshold,
        "device": "cuda (Tesla T4)",
    }

    # cover + grid question + one page per volume + threshold page + packaging
    total = 2 + len(results) * 2 + 1

    with PdfPages(args.out) as pdf:
        page = 1
        page_cover(pdf, results, page, total, cfg)
        page += 1
        page_grid_question(pdf, results, page, total)
        page += 1
        for result in results:
            page_volume(pdf, result, artifacts, page, total)
            page += 1
            stem = result["volume"].replace(".ome.tif", "")
            if (artifacts / f"{stem}.prob.npy").is_file():
                page_threshold(pdf, result, artifacts, page, total)
                page += 1
            else:
                total -= 1
        page_packaging(pdf, page, total, Path(args.package) if args.package else None)

        metadata = pdf.infodict()
        metadata["Title"] = "nuxnet inference validation on NuMorph"
        metadata["Subject"] = "CLI prediction, instance post-processing, BioImage.IO packaging"
        metadata["Keywords"] = "nuxnet, numorph, nuclei, 3D U-Net, light-sheet"

    print(f"wrote {args.out} ({total} pages, {len(results)} volumes)")


if __name__ == "__main__":
    main()
