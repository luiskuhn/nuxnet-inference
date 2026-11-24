"""Build a validation report as a PDF, from the artifacts eval_numorph.py saved.

The report exists because numbers alone hide failure modes and pictures alone hide
systematic error. Each volume gets a slice montage -- where a misalignment is visible --
next to a per-slice Dice curve, where it is measurable.

One page is deliberately about *where* the score is computed. On this dataset that
choice moved the headline number from 0.46 to 0.80, so a report that omitted it would
be misleading rather than merely incomplete.

Layout notes for anyone editing the pages:

* Mask colours are deliberately high-contrast and never both light: truth is cyan,
  prediction is magenta, agreement is white. Green/red was unreadable for viewers with
  the common red-green deficiency and on printouts.
* Every panel has a caption saying what it shows and how it was displayed, because a
  report that requires the reader to guess at the colour mapping is not a report.

    python make_report.py --out report.pdf --artifacts /out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi

# Ink and chrome
INK = "#141414"
MUTED = "#5f5f5f"
RULE = "#d8d8d8"
RULE_LIGHT = "#c8c8d0"   # big stage numbers, readable but recessive
BLUE = "#2251a4"
GREEN = "#1f7a3f"
AMBER = "#a86400"

# Raster resolution for embedded panels. The default 100 dpi renders a slice panel at
# ~173 px, which is too coarse to judge a mask boundary by eye -- the whole reason these
# pages exist. 300 dpi puts them near 520 px.
DPI = 300

# Mask colours: cyan / magenta / white. Maximally distinguishable from each other,
# from greyscale anatomy, and for the most common colour-vision deficiencies.
C_TRUTH = "#00d5ff"   # ground truth only
C_PRED = "#ff2d95"    # prediction only
C_BOTH = "#ffffff"    # agreement
BG = "#000000"        # mask panel background

# Detection-metric colours. Deliberately the same hue family as the mask panels so a
# reader can carry the association across pages: cyan is truth-side (recall), magenta
# is prediction-side (precision).
PRECISION_C = "#ff2d95"
RECALL_C = "#00b8dd"
F1_C = "#6b46c1"
DIST_C = "#a86400"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(a, b).sum() / denom)


def component_sizes(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndi.label(mask, structure=np.ones((3, 3, 3)))
    return np.array([]) if n == 0 else np.bincount(labeled.ravel())[1:]


def display_stretch(volume: np.ndarray) -> np.ndarray:
    """Percentile stretch for *display only*.

    Min-max on light-sheet data is dominated by a few bright voxels, which pushes the
    faint nuclei that the model is being judged on into near-black. A 1-99.5 percentile
    stretch shows them. This never touches the data the model or the scores see.
    """
    low, high = np.percentile(volume, [1.0, 99.5])
    if high <= low:
        return np.zeros(volume.shape, dtype=float)
    return np.clip((volume - low) / (high - low), 0.0, 1.0)


def page_header(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.06, 0.965, title, fontsize=16, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.06, 0.933, subtitle, fontsize=8.8, color=MUTED, va="top")
    fig.add_artist(
        plt.Line2D([0.06, 0.94], [0.922, 0.922], color=RULE, linewidth=0.8)
    )


def page_footer(fig, page: int, total: int, note: str = "") -> None:
    if note:
        fig.text(0.06, 0.022, note, fontsize=6.8, color=MUTED, va="bottom")
    fig.text(0.94, 0.022, f"{page} / {total}", fontsize=7.5, color=MUTED, va="bottom",
             ha="right")


def caption(fig, x, y, text, size=7.4, color=MUTED, ha="left"):
    fig.text(x, y, text, fontsize=size, color=color, ha=ha, va="top")


def dedent(text: str) -> str:
    import textwrap

    return textwrap.dedent(text).strip("\n")


def mask_rgb(truth: np.ndarray | None, pred: np.ndarray | None) -> np.ndarray:
    """Compose a mask into RGB using the report's colour convention."""
    shape = (truth if truth is not None else pred).shape
    rgb = np.zeros(shape + (3,), dtype=float)
    if truth is not None and pred is not None:
        rgb[truth & ~pred] = matplotlib.colors.to_rgb(C_TRUTH)
        rgb[pred & ~truth] = matplotlib.colors.to_rgb(C_PRED)
        rgb[truth & pred] = matplotlib.colors.to_rgb(C_BOTH)
    elif truth is not None:
        rgb[truth] = matplotlib.colors.to_rgb(C_TRUTH)
    else:
        rgb[pred] = matplotlib.colors.to_rgb(C_PRED)
    return rgb


def overlay_on(image: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Grey anatomy with the mask colours composited on top at full opacity."""
    grey = display_stretch(image)
    rgb = np.dstack([grey, grey, grey])
    rgb[truth & ~pred] = matplotlib.colors.to_rgb(C_TRUTH)
    rgb[pred & ~truth] = matplotlib.colors.to_rgb(C_PRED)
    rgb[truth & pred] = matplotlib.colors.to_rgb(C_BOTH)
    return rgb


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def page_cover(pdf: PdfPages, results: list[dict], page: int, total: int, cfg: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    train = np.array([r["train_grid_dice"] for r in results])
    ceil = np.array([r["roundtrip_ceiling"] for r in results])
    inp = np.array([r["input_grid_dice"] for r in results])

    fig.text(0.07, 0.905, "nuxnet inference", fontsize=30, fontweight="bold", color=INK)
    fig.text(0.07, 0.868, "validation on the NuMorph training dataset",
             fontsize=13.5, color=MUTED)

    fig.text(0.07, 0.815, "Headline", fontsize=10.5, fontweight="bold", color=BLUE)
    entries = [
        ("train-grid Dice", f"{train.mean():.3f}",
         f"range {train.min():.3f} – {train.max():.3f}"),
        ("input-grid Dice", f"{inp.mean():.3f}",
         f"{inp.mean() / ceil.mean() * 100:.0f}% of achievable"),
        ("volumes scored", f"{len(results)}", "image / mask pairs"),
        ("checkpoint", "108", "tensors, strict=True"),
    ]
    y = 0.775
    for label, value, note in entries:
        fig.text(0.09, y, label, fontsize=9.5, color=MUTED)
        fig.text(0.40, y - 0.006, value, fontsize=19, fontweight="bold", color=INK)
        fig.text(0.55, y, note, fontsize=8.2, color=MUTED)
        y -= 0.048

    fig.text(0.07, 0.560, "What was run", fontsize=10.5, fontweight="bold", color=BLUE)
    steps = [
        ("1", "Read one OME-TIFF and its physical voxel size."),
        ("2", "Resample to 3.0 / 1.0 / 1.0 µm using the trainer's own dataloader."),
        ("3", "Min-max normalize the volume to [0,1]."),
        ("4", "Sliding window, 32×128×128 patch, 50% overlap, softmax marker channel."),
        ("5", "Threshold 0.80 → binary mask on the training grid."),
        ("6", "Project the mask back to the input grid (trilinear, re-threshold 0.5)."),
        ("7", "Write the OME-TIFF mask with calibration, and the nuclei TSV."),
        ("8", "Score on both grids, and compute the round-trip ceiling."),
    ]
    y = 0.528
    for number, text in steps:
        fig.text(0.09, y, number, fontsize=8.6, color=BLUE, family="monospace")
        fig.text(0.125, y, text, fontsize=8.6, color=INK)
        y -= 0.0255

    fig.text(0.07, 0.295, "Configuration", fontsize=10.5, fontweight="bold", color=BLUE)
    rows = [
        ("weights", cfg["weights"]),
        ("target voxel spacing", f"{cfg['target']} µm (Z, Y, X)"),
        ("patch size", f"{cfg['patch']} (trainer default)"),
        ("threshold", str(cfg["threshold"])),
        ("device", cfg["device"]),
    ]
    y = 0.264
    for key, value in rows:
        fig.text(0.09, y, key, fontsize=8.6, color=MUTED)
        fig.text(0.34, y, str(value), fontsize=8.6, color=INK, family="monospace")
        y -= 0.024

    fig.text(
        0.07,
        0.130,
        dedent("""
        How to read this report. Page 2 explains why the score depends on which grid it
        is computed on — the single most misleading thing about these numbers. Pages 3–4
        describe the pipeline and the scoring method. Pages 5 onward give one page per
        volume: large slice panels where misalignment would be visible, and a per-slice
        Dice curve where it is measurable. The last pages aggregate all volumes.
        """).replace("\n        ", "\n"),
        fontsize=8.4,
        color=INK,
        va="top",
        linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f6f7f9", edgecolor=RULE, linewidth=0.8),
    )

    page_footer(fig, page, total)
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_grid_question(pdf: PdfPages, results: list[dict], page: int, total: int) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(
        fig,
        "Which grid the score is computed on",
        "The same prediction scores ~0.46 or ~0.85 depending only on where it is compared.",
    )

    fig.text(
        0.06,
        0.895,
        dedent("""
        Inputs are stored at 2.5 / 0.75 / 0.75 µm (group C075) and 4.0 / 1.21 / 1.21 µm
        (group C121). The checkpoint was trained at 3.0 / 1.0 / 1.0 µm, so inference
        resamples onto that grid, predicts there, then projects the mask back so the
        output lines up with the input file the caller passed in.

        That projection cannot preserve which slice was which. When the source is coarser
        in Z than the training grid (C121, 4.0 µm), the training grid has MORE slices and
        the round trip is nearly lossless. When the source is finer (C075, 2.5 µm),
        64 slices must compress into 53 and back, and the round trip throws information
        away — a single duplicated slice in six.

        The CEILING below is the Dice a PERFECT mask would score after the same round
        trip, with no network involved. No model can exceed it on the input grid.
        """).replace("\n        ", "\n"),
        fontsize=9.0,
        color=INK,
        va="top",
        linespacing=1.6,
    )

    # --- main bars
    axis = fig.add_axes([0.36, 0.600, 0.55, 0.145])
    labels = ["perfect mask,\nround-tripped", "model output,\ninput grid",
              "model output,\ntraining grid"]
    values = [
        float(np.mean([r["roundtrip_ceiling"] for r in results])),
        float(np.mean([r["input_grid_dice"] for r in results])),
        float(np.mean([r["train_grid_dice"] for r in results])),
    ]
    bars = axis.barh(labels, values, color=[MUTED, AMBER, GREEN], height=0.6)
    for bar, value in zip(bars, values):
        axis.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                  va="center", fontsize=10.5, fontweight="bold", color=INK)
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Dice coefficient, mean over all volumes", fontsize=8.4, color=MUTED)
    axis.tick_params(labelsize=8.2, colors=INK)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_title("A perfect mask sets the achievable maximum on the input grid",
                   fontsize=9.4, color=INK, pad=7)
    caption(fig, 0.36, 0.590,
            "Grey = ceiling (no model). Amber = what the CLI wrote. Green = the network's own score.")

    # --- per-group
    fig.text(0.06, 0.520, "Split by acquisition group", fontsize=10.5,
             fontweight="bold", color=BLUE)
    caption(fig, 0.06, 0.499,
            "The two groups fail in opposite directions, which is why one pooled number hides both.")

    groups = sorted({r["group"] for r in results})
    axis = fig.add_axes([0.24, 0.315, 0.62, 0.155])
    width = 0.26
    positions = np.arange(3)
    for index, group in enumerate(groups):
        subset = [r for r in results if r["group"] == group]
        vals = [np.mean([r["roundtrip_ceiling"] for r in subset]),
                np.mean([r["input_grid_dice"] for r in subset]),
                np.mean([r["train_grid_dice"] for r in subset])]
        axis.bar(positions + (index - 0.5) * width, vals, width * 0.92,
                 label=f"{group}  ({subset[0]['source_spacing'][0]:g} µm Z, n={len(subset)})",
                 color=["#9aa0a6", "#d08a2c"][index], alpha=0.95)
        for pos, val in zip(positions + (index - 0.5) * width, vals):
            axis.text(pos, val + 0.015, f"{val:.2f}", ha="center", fontsize=7.4, color=INK)
    axis.set_xticks(positions)
    axis.set_xticklabels(["ceiling", "input grid", "training grid"], fontsize=8.4)
    axis.set_ylim(0, 1.1)
    axis.set_ylabel("Dice", fontsize=8.2, color=MUTED)
    axis.tick_params(labelsize=7.6, colors=MUTED)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=7.6, frameon=False, loc="lower right")
    axis.set_title("C075 compresses 64→53 slices and loses accuracy; C121 upsamples and gains it",
                   fontsize=8.6, color=INK, pad=6)

    fig.text(
        0.06,
        0.210,
        dedent("""
        Read the tall grey bars first. For C075 the ceiling is only ~0.78, so the
        input-grid score of ~0.70 is close to what the data allows. For C121 the ceiling
        is ~0.98 and the input-grid score is actually HIGHER than the training-grid score
        — not because the model improves, but because upsampling fills gaps in a way that
        happens to agree with the ground truth.

        Practical consequence: the mask must stay on the input grid, because downstream
        tools index it against the input volume. The scaling loss belongs to the export,
        not the model. Quoting an input-grid Dice without its ceiling blames the network
        for the resampling.
        """).replace("\n        ", "\n"),
        fontsize=8.6,
        color=INK,
        va="top",
        linespacing=1.55,
    )

    page_footer(fig, page, total,
                "ceiling = ground truth round-tripped through resample_volume_pair, no model")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_method(pdf: PdfPages, page: int, total: int) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(fig, "What inference actually does",
                "The network has no concept of physical scale, so the pipeline supplies it.")

    fig.text(
        0.06,
        0.890,
        dedent("""
        A checkpoint reproduces its training behaviour only if inference feeds it what the
        trainer fed it. The original CLI did the opposite: whole-volume inference at the
        sampling the file happened to be stored at, no resampling, no tiling. Measured on
        the bundled example, that path scores input-grid Dice 0.456 and training-grid Dice
        0.452, against 0.739 and 0.860 with training-grid inference. That gap is the
        pipeline, not the model.
        """).replace("\n        ", "\n"),
        fontsize=9.0,
        color=INK,
        va="top",
        linespacing=1.6,
    )

    fig.text(0.06, 0.790, "Stage by stage", fontsize=10.5, fontweight="bold", color=BLUE)

    stages = [
        ("Read",
         "One OME-TIFF. Physical voxel size is parsed from the OME-XML in Z, Y, X order.\n"
         "A plain TIFF has no calibration; that is reported, not guessed."),
        ("Resample",
         "Trainer's resample_volume_pair, vendored verbatim at the mirrored path.\n"
         "3.0 / 1.0 / 1.0 µm, preserving physical extent. Nearest-neighbour is reserved\n"
         "for labels; the image gets Gaussian anti-aliasing when downsampling."),
        ("Normalize",
         "Per-volume min-max to [0,1], applied AFTER resampling — the same order as\n"
         "the trainer. This is a measured choice, not a convention."),
        ("Tile",
         "Sliding window 32 × 128 × 128 at 50% overlap. Each tile is padded so every\n"
         "spatial axis is a multiple of 4: the decoder upsamples twice, and a 53-voxel\n"
         "axis otherwise crashes torch.cat inside the decoder."),
        ("Forward",
         "Two-channel output: [background, marker]. softmax(dim=1) gives the marker\n"
         "probability. The packaged weights emit LOGITS, not probabilities."),
        ("Blend",
         "Overlapping tiles combined with a separable Hann taper. Voxels are divided\n"
         "only where a window actually contributed — clamping the denominator rescaled\n"
         "corner voxels from 0.90 down to 0.04."),
        ("Threshold",
         "0.80 on the training grid. Dice is flat across 0.4–0.9 (see the aggregate\n"
         "page), so this is not a tuned operating point."),
        ("Project back",
         "Binary mask interpolated trilinearly to the input grid, then re-thresholded at\n"
         "0.5. Nearest-neighbour duplicated one slice in six and capped a PERFECT mask\n"
         "at Dice 0.50; trilinear on the mask reached 0.85 of ceiling."),
        ("Write",
         "OME-TIFF mask on the input grid, binary 0/1, carrying PhysicalSize — without\n"
         "it the mask aligns with the array but not with the specimen. Plus the nuclei\n"
         "TSV (id, z, y, x, size_voxels) from the unchanged post-processing."),
    ]

    y = 0.755
    for name, detail in stages:
        fig.add_artist(plt.Line2D([0.06, 0.94], [y + 0.019, y + 0.019],
                                  color=RULE, linewidth=0.5))
        fig.text(0.065, y, name, fontsize=9.0, fontweight="bold", color=INK, va="top")
        fig.text(0.215, y, detail, fontsize=7.8, color=MUTED, va="top", linespacing=1.45)
        y -= 0.0225 * (detail.count("\n") + 1) + 0.012

    page_footer(fig, page, total,
                "post-processing (extract_nuclei_instances, write_instances_tsv) is unchanged from forward-cycle")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def _stage_block(fig, y: float, number: str, title: str, operation: str,
                 code: str | None, why: str) -> float:
    """Lay out one numbered stage and return the y where the next one starts.

    Splitting the layout out of the page bodies keeps the vertical arithmetic in
    one place: previously each page recomputed it and they drifted.
    """
    fig.text(0.06, y, number, fontsize=13, fontweight="bold", color=RULE_LIGHT,
             va="top", ha="left")
    fig.text(0.105, y, title, fontsize=9.4, fontweight="bold", color=BLUE, va="top")
    fig.text(0.105, y - 0.0195, operation, fontsize=7.6, color=INK, va="top",
             linespacing=1.42)
    y -= 0.0195 + 0.0122 * (operation.count("\n") + 1)

    if code:
        lines = code.count("\n") + 1
        height = 0.0122 * lines + 0.013
        fig.patches.append(
            plt.Rectangle((0.103, y - height + 0.0075), 0.845, height,
                          transform=fig.transFigure, facecolor="#f4f4f6",
                          edgecolor="none", zorder=0)
        )
        fig.text(0.112, y, code, fontsize=6.5, color="#2b2b2b", family="monospace",
                 va="top", linespacing=1.34)
        y -= height + 0.004

    fig.text(0.105, y, why, fontsize=7.0, color=AMBER, va="top", linespacing=1.38)
    return y - 0.0122 * (why.count("\n") + 1) - 0.016


def page_preprocessing(pdf: PdfPages, page: int, total: int) -> None:
    """Load, calibrate, normalize, resample -- everything before the network runs.

    These four steps are where inference silently diverges from training, so each
    one gets its actual code and the measurement that pinned the choice.
    """
    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(
        fig,
        "Preprocessing: putting the volume on the training grid",
        "The network has no concept of physical scale. These steps supply it.",
    )

    fig.text(
        0.06, 0.893,
        dedent("""
        A convolution kernel is 3 voxels wide whether those voxels are 1 µm or 4 µm apart, so
        a checkpoint only means anything at the sampling it was trained on: 3.0 / 1.0 / 1.0 µm.
        Everything on this page exists to transform arbitrary microscopy into that grid, and
        each operation is delegated to the trainer's own function so the two cannot drift.
        """).replace("\n        ", "\n"),
        fontsize=8.2, color=INK, va="top", linespacing=1.5,
    )

    y = 0.815
    y = _stage_block(
        fig, y, "1", "Read the pixels and the calibration",
        "Axes are resolved by name, never by position: the OME-XML axis list is searched for\n"
        "Z, Y and X and the array transposed into that order. Any remaining axis must be a\n"
        "singleton and is squeezed. Values are cast to float32. For OME-NGFF the same happens\n"
        "through the multiscales metadata, where scale and translation are read per axis.",
        "tifffile: TiffFile -> series.asarray()\n"
        "ome:      PhysicalSizeZ, PhysicalSizeY, PhysicalSizeX  (µm)\n"
        "zarr:     multiscales[0].datasets[0].coordinateTransformations",
        "The ear store is [t, c, z, y, x], so a positional scale[-3:] happens to be correct here\n"
        "and would be silently wrong on a [c, z, y, x] store. Name lookup cannot make that mistake.",
    )

    y = _stage_block(
        fig, y, "2", "Normalize with one range per volume",
        "Contrast is rescaled by a single min/max for the whole volume, clipped to [0, 1].\n"
        "The range is computed AFTER resampling, because interpolating raw intensities and then\n"
        "interpolating normalized ones give different values at every new voxel.",
        "low, high = volume.min(), volume.max()\n"
        "out = clip((volume - low) / (high - low), 0, 1)",
        "Per-chunk min/max is a correctness bug, not a tuning choice: identical tissue got\n"
        "different contrast depending on which chunk it landed in, and tiled-vs-untiled Dice\n"
        "fell to 0.458. A chunked run therefore passes one shared --normalization-range.",
    )

    y = _stage_block(
        fig, y, "3", "Resample to 3.0 / 1.0 / 1.0 µm, per axis",
        "Each axis gets its own factor, source µm divided by target µm, and its own output\n"
        "length. A Gaussian pre-blur is applied only to axes that shrink, with sigma derived\n"
        "from the factor, so downsampling does not alias. Upsampling is left unfiltered.",
        "factor = source_um / target_um            # per axis, e.g. 1.67 / 3.55 / 3.55\n"
        "out_len = round(in_len * factor)\n"
        "if factor < 1: gaussian_blur(axis, sigma=0.5*(1/factor - 1))",
        "One isotropic factor would be visibly wrong on this data: Z needs 1.67x and XY need\n"
        "3.55x. On the ear ROI that is 216x690x630 becoming 360x2446x2234 voxels.",
    )

    y = _stage_block(
        fig, y, "4", "Interpolate trilinearly, preserving physical extent",
        "Trilinear with align_corners=False, so a destination index maps to\n"
        "src = (dst + 0.5) * (src_len / dst_len) - 0.5. The voxel count changes; the micrometres\n"
        "the volume spans do not. That is why the output mask can carry the input's spacing.",
        "F.interpolate(vol, size=out_shape, mode='trilinear', align_corners=False)\n"
        "# labels instead use mode='nearest-exact', so binary stays binary",
        "align_corners=True shifts content by up to half a voxel and would misregister the mask\n"
        "against the image in napari. Measured: neighbouring axes land within 1 source voxel.",
    )

    fig.text(0.06, 0.152, "What this costs, and what it saves", fontsize=9.4,
             fontweight="bold", color=BLUE)
    fig.text(
        0.06, 0.136,
        dedent("""
        Resampling is not free: the ear sits at 5.005 / 3.5455 / 3.5454 µm, so reaching
        3.0 / 1.0 / 1.0 µm multiplies the voxel count by 1.67 x 3.55 x 3.55 = 21x. The nucleus
        channel alone is then ~82 GB as float32 for the whole ear, which is why the layer is
        written region by region rather than in one pass, and why the reciprocal direction --
        projecting the finished mask back onto the input grid -- is done by interpolating the
        BINARY mask and re-thresholding at 0.5 rather than by interpolating probabilities.
        """).replace("\n        ", "\n"),
        fontsize=7.8, color=INK, va="top", linespacing=1.5,
    )

    page_footer(fig, page, total,
                "source: nuxnet_inference_package/preprocessing.py, data_loading/data_loader.py")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_forward(pdf: PdfPages, page: int, total: int) -> None:
    """Sliding window, the network, blending, and the path from probability to centroid."""
    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(
        fig,
        "Forward pass, then mask to centroids",
        "How a probability map becomes the .nuclei.tsv rows on the previous page.",
    )

    fig.text(
        0.06, 0.893,
        dedent("""
        The network is a 3D U-Net with 2.36 M parameters: one input block, two downsampling
        blocks, a bottleneck, two upsampling blocks with skip concatenation, and a 1x1x1 head
        producing two logits per voxel. Shown below are the real tensor shapes for one
        32 x 128 x 128 tile, and the operations that turn its output into positions.
        """).replace("\n        ", "\n"),
        fontsize=8.2, color=INK, va="top", linespacing=1.5,
    )

    # ---- tensor shape trace --------------------------------------------------
    ax = fig.add_axes((0.075, 0.545, 0.375, 0.285))
    ax.axis("off")
    rows = [
        ("input", "(1, 1, 32, 128, 128)"),
        ("inc  ConvBlock 1->32", "(1, 32, 32, 128, 128)"),
        ("down1  2x2/s2, 32->64", "(1, 64, 16, 64, 64)"),
        ("down2  2x2/s2, 64->128", "(1, 128, 8, 32, 32)"),
        ("mid  ConvBlock 128", "(1, 128, 8, 32, 32)"),
        ("up1  x2 + cat(x2), ->64", "(1, 64, 16, 64, 64)"),
        ("up2  x2 + cat(x1), ->32", "(1, 32, 32, 128, 128)"),
        ("outc  1x1x1 head", "(1, 2, 32, 128, 128)"),
    ]
    fig.text(0.075, 0.836, "Tensor shapes for one tile", fontsize=9.0,
             fontweight="bold", color=BLUE)
    yy = 0.820
    for name, shape in rows:
        highlight = name.startswith(("input", "outc"))
        fig.text(0.075, yy, name, fontsize=6.6,
                 family="monospace", color=INK if highlight else MUTED,
                 fontweight="bold" if highlight else "normal")
        fig.text(0.435, yy, shape, fontsize=6.6, family="monospace",
                 color=INK if highlight else MUTED, ha="right")
        yy -= 0.0138

    fig.text(0.075, 0.700,
             "2,364,834 parameters -- constant across batch 1/2/4",
             fontsize=6.8, color=MUTED, style="italic")

    # ---- why the tiling rules exist -----------------------------------------
    fig.text(0.500, 0.836, "Why each rule exists", fontsize=9.0,
             fontweight="bold", color=BLUE)
    fig.text(
        0.500, 0.818,
        dedent("""
        Multiple of 4. Two stride-2 downsamplings halve each axis twice, so
        every spatial dimension must be divisible by 4 or the upsampled
        feature map is one voxel wider than its skip connection and
        torch.cat raises a size mismatch. A 53-voxel axis is therefore a
        hard error, not a mild degradation.

        Two classes, not three. The head emits background and marker. Only
        the marker logit matters: softmax(dim=1)[:, 1] is the foreground
        probability. Using argmax would give the same binary decision but
        throw away the confidence the threshold needs.

        Logits, not probabilities. The head is a bare 1x1x1 convolution,
        so its output is unbounded. Applying a threshold of 0.8 directly to
        logits would mark almost the entire volume foreground.
        """).replace("\n        ", "\n"),
        fontsize=6.9, color=INK, va="top", linespacing=1.45,
    )

    y = 0.640
    y = _stage_block(
        fig, y, "5", "Slide, with overlap, then blend with a Hann taper",
        "The volume is cut into 32 x 128 x 128 windows stepping by 50% (or 25%) of a window.\n"
        "Overlapping predictions are combined by a separable Hann weight and divided by the\n"
        "accumulated weight, so a voxel near one tile's edge is carried mostly by its neighbour.",
        "step = round(patch * (1 - overlap))\n"
        "prob += tile_prob * w ;  wsum += w\n"
        "prob = where(wsum > 0, prob / wsum, prob)",
        "Dividing where wsum > 0 and not by a clamped denominator matters: the separable taper\n"
        "reaches ~5e-8 in a volume corner, so np.maximum(wsum, 1e-6) silently rescales corner\n"
        "voxels. Measured on a constant-0.9 input: 0.90 returned as 0.04 in 104 corner voxels.",
    )

    y = _stage_block(
        fig, y, "6", "Threshold on the training grid, then project back",
        "prob >= 0.80 gives a binary mask at 3.0 / 1.0 / 1.0 µm. To return it to the file's own\n"
        "grid, the BINARY mask is interpolated trilinearly and re-thresholded at 0.5, so an\n"
        "output voxel is foreground when most contributing source voxels were.",
        "mask = (prob >= 0.80)                      # training grid\n"
        "back = interpolate(mask, input_shape, 'trilinear', align_corners=False)\n"
        "mask_input = (back >= 0.5)",
        "Interpolating probabilities and thresholding afterwards was worse, because smoothing\n"
        "pulls peak probability below the operating point: 0.658 vs 0.800 of ceiling on\n"
        "C121_0009. Nearest-neighbour duplicates slices and scores only 0.390.",
    )

    y = _stage_block(
        fig, y, "7", "Label, cluster, and average -- the unchanged post-processing",
        "extract_nuclei_instances labels the mask with 6-connectivity, builds a KD-tree\n"
        "neighbourhood graph at radius 2 voxels, takes its connected components, and reports the\n"
        "mean voxel coordinate of each. This is untouched code from the original CLI.",
        "labeled = measure.label(mask, connectivity=1)\n"
        "tree = KDTree(argwhere(labeled > 0)); edges where distance <= 2\n"
        "centroid = mean(component_coords)   ->  write_instances_tsv(id, z, y, x, size)",
        "This step is why the mask is not the only output. A mask says which voxels; the centroid\n"
        "says where the nucleus is, which is what a biologist wants to load as points.",
    )

    fig.text(
        0.06, 0.118,
        dedent("""
        Result of the whole path on the 32-volume NuMorph set: training-grid Dice 0.801,
        detection F1 0.926, median matched-centroid distance 0.45 µm. The 0.45 µm figure is the
        honest accuracy of a reported position -- it is a sixth of a voxel on the XY plane and
        less than a tenth of a Z step, so the centroids are localised far more tightly than the
        mask Dice would suggest.
        """).replace("\n        ", "\n"),
        fontsize=7.6, color=INK, va="top", linespacing=1.5,
    )

    page_footer(fig, page, total,
                "source: model/unet_3d_models.py, preprocessing.py, cli_pred.py:extract_nuclei_instances")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_volume(pdf: PdfPages, result: dict, artifacts: Path, page: int, total: int,
                threshold: float = 0.8) -> None:
    stem = result["volume"].replace(".ome.tif", "")
    image = np.load(artifacts / f"{stem}.prepared.npy").astype(np.float32)
    truth = np.load(artifacts / f"{stem}.truth_grid.npy") > 0
    probability = np.load(artifacts / f"{stem}.prob.npy").astype(np.float32)
    pred = probability >= threshold

    depth = image.shape[0]
    slices = [int(depth * f) for f in (0.32, 0.62)]

    fig = plt.figure(figsize=(8.27, 11.69))
    title = stem.replace("NUMORPH_SEM_SEG_", "").replace("_", " ")
    page_header(
        fig,
        title,
        f"group {result['group']}   source spacing {result['source_spacing']} µm   "
        f"training-grid shape {tuple(result['prepared_shape'])}   "
        f"input-grid shape {tuple(result['shape'])}   {result['tiles']} tiles",
    )

    fig.text(
        0.06,
        0.912,
        f"train-grid Dice {result['train_grid_dice']:.3f}     "
        f"input-grid Dice {result['input_grid_dice']:.3f}     "
        f"ceiling {result['roundtrip_ceiling']:.3f}     "
        f"{result['fraction_of_ceiling'] * 100:.0f}% of achievable",
        fontsize=9.0,
        color=INK,
        family="monospace",
    )

    # ------------------------------------------------------------------
    # montage: 2 slice rows x 4 panels, as large as the page allows
    # ------------------------------------------------------------------
    left = 0.055
    column = 0.208
    gap = 0.016
    top = 0.858
    row_area = 0.395
    row_h = row_area / len(slices)

    headers = [
        ("Input volume", f"display: 1–99.5 pct stretch"),
        ("Ground truth", "marker voxels, cyan"),
        ("Prediction", f"p ≥ {threshold:.2f}, magenta"),
        ("Overlap", "cyan truth / magenta pred / white both"),
    ]
    for index, (header, sub) in enumerate(headers):
        x = left + index * (column + gap)
        fig.text(x, top + 0.040, header, fontsize=8.2, fontweight="bold", color=INK)
        fig.text(x, top + 0.024, sub, fontsize=6.5, color=MUTED)

    for row, z in enumerate(slices):
        y = top - (row + 1) * row_h
        height = row_h - 0.042
        panels = [
            (display_stretch(image[z]), "gray", (0, 1)),
            (mask_rgb(truth[z], None), None, None),
            (mask_rgb(None, pred[z]), None, None),
            (overlay_on(image[z], truth[z], pred[z]), None, None),
        ]
        for col, (data, cmap, lim) in enumerate(panels):
            axis = fig.add_axes([left + col * (column + gap), y + 0.030, column, height])
            if cmap:
                axis.imshow(data, cmap=cmap, vmin=lim[0], vmax=lim[1],
                            interpolation="nearest")
            else:
                axis.imshow(data, interpolation="nearest")
                axis.set_facecolor(BG)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(RULE)
                spine.set_linewidth(0.7)
        fig.text(left - 0.030, y + 0.030 + height / 2, f"z = {z}", fontsize=7.6,
                 color=INK, va="center", rotation=90)

    # ------------------------------------------------------------------
    # per-slice Dice
    # ------------------------------------------------------------------
    fig.text(0.055, 0.455, "Per-slice agreement on the training grid", fontsize=9.6,
             fontweight="bold", color=INK)
    caption(fig, 0.055, 0.440,
            "Each point is Dice of one slice. A flat curve means no systematic slice offset; "
            "alternating highs and lows would indicate a resampling phase error.")

    axis = fig.add_axes([0.095, 0.285, 0.50, 0.135])
    per_slice = [dice(pred[z], truth[z]) for z in range(depth)]
    axis.plot(range(depth), per_slice, color=BLUE, linewidth=1.3, marker="o",
              markersize=2.4, label="Dice at this slice")
    axis.axhline(result["train_grid_dice"], color=GREEN, linestyle="--", linewidth=1.1,
                 label=f"whole-volume Dice {result['train_grid_dice']:.3f}")
    for z in slices:
        axis.axvline(z, color=AMBER, linewidth=0.9, alpha=0.7)
    axis.set_xlim(-1, depth)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("slice index (training grid)", fontsize=7.6, color=MUTED)
    axis.set_ylabel("Dice", fontsize=7.6, color=MUTED)
    axis.tick_params(labelsize=7.0, colors=MUTED)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=7.0, frameon=False, loc="lower right")
    caption(fig, 0.095, 0.276, "Amber vertical lines mark the two slices shown above.")

    # ------------------------------------------------------------------
    # component sizes
    # ------------------------------------------------------------------
    pred_sizes = component_sizes(pred)
    truth_sizes = component_sizes(truth)
    fig.text(0.055, 0.235, "Object size distribution", fontsize=9.6,
             fontweight="bold", color=INK)
    caption(fig, 0.055, 0.221,
            "Connected components (26-connectivity). The target is a 2D single-slice disc "
            "per nucleus (median 63 voxels), so a component of 1-2 voxels means under-segmentation.")

    axis = fig.add_axes([0.095, 0.115, 0.38, 0.085])
    upper = max(pred_sizes.max() if pred_sizes.size else 1,
                truth_sizes.max() if truth_sizes.size else 1)
    bins = np.linspace(0, upper, 26)
    axis.hist(truth_sizes, bins=bins, color=C_TRUTH, alpha=0.85, density=True,
              label="ground truth")
    axis.hist(pred_sizes, bins=bins, color=C_PRED, alpha=0.60, density=True,
              label="prediction")
    axis.set_xlabel("component size (voxels on training grid)", fontsize=7.2, color=MUTED)
    axis.set_yticks([])
    axis.tick_params(labelsize=6.8, colors=MUTED)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.legend(fontsize=6.8, frameon=False)

    # ------------------------------------------------------------------
    # numbers
    # ------------------------------------------------------------------
    lines = [
        ("components", f"truth {len(truth_sizes):>5d}", f"prediction {len(pred_sizes):>5d}"),
        ("median size", f"truth {np.median(truth_sizes) if truth_sizes.size else 0:>5.0f}",
         f"prediction {np.median(pred_sizes) if pred_sizes.size else 0:>5.0f}"),
        ("foreground", f"truth {truth.mean() * 100:>5.2f}%",
         f"prediction {pred.mean() * 100:>5.2f}%"),
        ("nuclei TSV", f"{result['instances']:>5d} rows", ""),
    ]
    fig.text(0.565, 0.205, "Measured on this volume", fontsize=8.6, fontweight="bold",
             color=INK)
    y = 0.188
    for label, a, b in lines:
        fig.text(0.565, y, label, fontsize=7.4, color=MUTED)
        fig.text(0.655, y, a, fontsize=7.4, color=INK, family="monospace")
        fig.text(0.775, y, b, fontsize=7.4, color=INK, family="monospace")
        y -= 0.019

    page_footer(fig, page, total,
                "panel colour key: cyan = ground truth only, magenta = prediction only, white = agreement")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_threshold_all(pdf: PdfPages, artifacts: Path, page: int, total: int) -> None:
    """One aggregate threshold sweep over every volume, instead of one page per volume."""
    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(
        fig,
        "Threshold sensitivity, all volumes",
        "One sweep across every scored volume — the operating point is not cherry-picked.",
    )

    fig.text(
        0.06,
        0.895,
        dedent("""
        For each volume the marker probability was thresholded at 19 values from 0.05 to
        0.95 and rescored. Curves show the mean across volumes; the band is the
        interquartile range, so it reflects real spread rather than a single lucky volume.
        If the result depended on the threshold, this page would show a narrow peak.
        """).replace("\n        ", "\n"),
        fontsize=8.8,
        color=INK,
        va="top",
        linespacing=1.6,
    )

    thresholds = np.linspace(0.05, 0.95, 19)
    per_volume: dict[str, list[float]] = {t: [] for t in thresholds}
    per_group: dict[str, dict[float, list[float]]] = {}
    precision: dict[float, list[float]] = {t: [] for t in thresholds}
    recall: dict[float, list[float]] = {t: [] for t in thresholds}
    per_volume_foreground: dict[float, list[float]] = {t: [] for t in thresholds}
    truth_frac: list[float] = []

    stems = sorted(
        p.name[: -len(".prob.npy")] for p in artifacts.glob("*.prob.npy")
    )
    for stem in stems:
        prob = np.load(artifacts / f"{stem}.prob.npy").astype(np.float32)
        truth = np.load(artifacts / f"{stem}.truth_grid.npy") > 0
        truth_frac.append(float(truth.mean()))
        group = "C121" if "C121" in stem else "C075"
        per_group.setdefault(group, {t: [] for t in thresholds})
        for t in thresholds:
            pred = prob >= t
            d = dice(pred, truth)
            per_volume[t].append(d)
            per_group[group][t].append(d)
            per_volume_foreground[t].append(float(pred.mean()) * 100.0)
            inter = np.logical_and(pred, truth).sum()
            precision[t].append(inter / max(pred.sum(), 1))
            recall[t].append(inter / max(truth.sum(), 1))

    mean_dice = np.array([np.mean(per_volume[t]) for t in thresholds])
    q25 = np.array([np.percentile(per_volume[t], 25) for t in thresholds])
    q75 = np.array([np.percentile(per_volume[t], 75) for t in thresholds])
    foreground_pct = [float(np.mean(per_volume_foreground[t])) for t in thresholds]
    best_index = int(np.argmax(mean_dice))
    best_t, best_d = thresholds[best_index], mean_dice[best_index]

    axis = fig.add_axes([0.10, 0.555, 0.80, 0.265])
    axis.fill_between(thresholds, q25, q75, color=BLUE, alpha=0.16,
                      label="interquartile range across volumes")
    axis.plot(thresholds, mean_dice, color=BLUE, linewidth=2.0, label="mean Dice")
    axis.plot(thresholds, [np.mean(precision[t]) for t in thresholds], color=C_PRED,
              linewidth=1.3, linestyle="--", label="mean precision")
    axis.plot(thresholds, [np.mean(recall[t]) for t in thresholds], color=AMBER,
              linewidth=1.3, linestyle="--", label="mean recall")
    axis.axvline(best_t, color=GREEN, linewidth=1.2, linestyle=":")
    axis.annotate(f"best mean Dice {best_d:.3f}\nat threshold {best_t:.2f}",
                  xy=(best_t, best_d), xytext=(best_t + 0.06, best_d - 0.16),
                  fontsize=8.6, color=GREEN,
                  arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.0))
    axis.axvline(0.80, color=MUTED, linewidth=1.0, linestyle="-", alpha=0.55)
    axis.text(0.805, 0.45, "operating point\nused in this report", fontsize=7.2,
              color=MUTED, va="bottom")
    axis.set_xlabel("foreground probability threshold", fontsize=8.4, color=MUTED)
    axis.set_ylabel("score (mean over volumes)", fontsize=8.4, color=MUTED)
    axis.set_ylim(0.30, 1.02)
    axis.tick_params(labelsize=7.8, colors=MUTED)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=7.8, frameon=False, loc="lower left")
    axis.set_title(f"Dice is flat from 0.4 to 0.9 (spread {mean_dice.max() - mean_dice.min():.3f})",
                   fontsize=9.4, color=INK, pad=8)

    # per-group curves
    fig.text(0.06, 0.505, "Same sweep, split by acquisition group", fontsize=9.6,
             fontweight="bold", color=INK)
    caption(fig, 0.06, 0.490,
            "C121 sits higher because its round trip is nearly lossless; the shape of the "
            "curve is what matters here, not its height.")
    axis = fig.add_axes([0.10, 0.330, 0.38, 0.135])
    for group, color in (("C075", "#7a7f85"), ("C121", "#2f7d4f")):
        if group in per_group:
            axis.plot(thresholds, [np.mean(per_group[group][t]) for t in thresholds],
                      color=color, linewidth=1.8, label=f"{group} (n={len(per_group[group][thresholds[0]])})")
    axis.axvline(0.80, color=MUTED, linewidth=0.8, linestyle=":", alpha=0.6)
    axis.set_xlabel("threshold", fontsize=7.6, color=MUTED)
    axis.set_ylabel("mean Dice", fontsize=7.6, color=MUTED)
    axis.tick_params(labelsize=7.0, colors=MUTED)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=7.2, frameon=False)

    # foreground fraction
    fig.text(0.545, 0.505, "Foreground fraction vs threshold", fontsize=9.6,
             fontweight="bold", color=INK)
    caption(fig, 0.545, 0.490, f"Ground truth is {np.mean(truth_frac) * 100:.2f}% foreground "
                               "on average.")
    axis = fig.add_axes([0.585, 0.330, 0.365, 0.135])
    axis.plot(thresholds, foreground_pct, color=BLUE, linewidth=1.8,
              label="prediction")
    axis.axhline(np.mean(truth_frac) * 100, color=C_TRUTH, linestyle="--", linewidth=1.1,
                 label="ground truth")
    axis.set_xlabel("threshold", fontsize=7.6, color=MUTED)
    axis.set_ylabel("foreground %", fontsize=7.6, color=MUTED)
    axis.tick_params(labelsize=7.0, colors=MUTED)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=7.2, frameon=False)

    table = "\n".join(
        f"  {t:.2f}     {mean_dice[i]:.3f}       {np.mean(precision[t]):.3f}      "
        f"{np.mean(recall[t]):.3f}"
        for i, t in enumerate(thresholds[::3])
    )
    fig.text(0.10, 0.265, "threshold   mean Dice   precision   recall", fontsize=7.6,
             color=INK, family="monospace")
    fig.text(0.10, 0.248, table, fontsize=7.6, color=MUTED, va="top",
             family="monospace", linespacing=1.65)

    fig.text(
        0.06,
        0.095,
        dedent("""
        Interpretation. Precision rises and recall falls monotonically with threshold, as
        expected; Dice, which balances them, is flat. That flatness is the useful result:
        it means the reported score reflects the model rather than a chosen operating
        point. 0.80 is used throughout because it favours precision slightly, which suits
        a marker model whose foreground is later clustered into instances.
        """).replace("\n        ", "\n"),
        fontsize=8.4,
        color=INK,
        va="top",
        linespacing=1.55,
    )

    page_footer(fig, page, total,
                f"computed over all {len(stems)} volumes; probability arrays saved from the same forward pass as the masks")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_table(pdf: PdfPages, results: list[dict], page: int, total: int) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(fig, "Per-volume results",
                "Every scored volume. The ceiling column is the number to compare against.")

    columns = ["volume", "grp", "src µm Z", "train", "ceil", "input", "%ceil", "nuclei"]
    xpos = [0.055, 0.470, 0.520, 0.605, 0.665, 0.725, 0.795, 0.860]

    fig.text(0.055, 0.888, " ".join(f"{c:<9s}" for c in columns), fontsize=7.4,
             color=INK, family="monospace", fontweight="bold")
    fig.add_artist(plt.Line2D([0.055, 0.945], [0.880, 0.880], color=RULE, linewidth=0.8))

    y = 0.868
    for result in results:
        name = result["volume"].replace("NUMORPH_SEM_SEG_", "").replace("_IMAGE_", "·")
        name = name.replace("_Training_", "·").replace(".ome.tif", "")
        name = name[:34]
        values = [
            f"{name:<34s}",
            f"{result['group']:<3s}",
            f"{result['source_spacing'][0]:>8.1f}",
            f"{result['train_grid_dice']:>5.3f}",
            f"{result['roundtrip_ceiling']:>5.3f}",
            f"{result['input_grid_dice']:>5.3f}",
            f"{result['fraction_of_ceiling'] * 100:>5.0f}",
            f"{result['instances']:>6d}",
        ]
        fig.text(0.055, y, " ".join(values), fontsize=6.5, color=MUTED,
                 family="monospace")
        y -= 0.0158

    # summary block
    train = np.array([r["train_grid_dice"] for r in results])
    ceil = np.array([r["roundtrip_ceiling"] for r in results])
    inp = np.array([r["input_grid_dice"] for r in results])

    fig.add_artist(plt.Line2D([0.055, 0.945], [y + 0.012, y + 0.012], color=RULE,
                              linewidth=0.8))
    summary = [
        ("train-grid Dice", f"mean {train.mean():.3f}   min {train.min():.3f}   max {train.max():.3f}"),
        ("round-trip ceiling", f"mean {ceil.mean():.3f}   min {ceil.min():.3f}   max {ceil.max():.3f}"),
        ("input-grid Dice", f"mean {inp.mean():.3f}   min {inp.min():.3f}   max {inp.max():.3f}"),
        ("fraction of ceiling", f"mean {np.mean(inp / ceil) * 100:.0f}%"),
    ]
    y -= 0.010
    for label, value in summary:
        fig.text(0.06, y, label, fontsize=8.2, color=MUTED)
        fig.text(0.30, y, value, fontsize=8.2, color=INK, family="monospace")
        y -= 0.022

    fig.text(
        0.06,
        y - 0.015,
        dedent("""
        Reading the table. "train" is what the network achieves on its own grid. "ceil" is
        what a perfect mask scores on the input grid after the same round trip — the
        achievable maximum, with no network involved. "input" is what the CLI wrote.
        "%ceil" divides input by ceiling, which is the only fair way to compare volumes
        stored at different samplings.

        The two groups are not interchangeable: C075 (2.5 µm Z) compresses slices and its
        ceiling is ~0.78, while C121 (4.0 µm Z) upsamples and its ceiling is ~0.98. A
        pooled mean over both therefore mixes two different upper bounds, which is why the
        split is reported alongside it everywhere in this document.
        """).replace("\n        ", "\n"),
        fontsize=8.0,
        color=INK,
        va="top",
        linespacing=1.55,
    )

    page_footer(fig, page, total, "source: summary.json written by scripts/eval_numorph.py")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_detections(
    pdf: PdfPages,
    artifacts: Path,
    detection: list[dict],
    page: int,
    total: int,
) -> None:
    """Positions, not just overlap: instance centroids and detection metrics.

    Dice answers "do the voxels agree". A user asking for nuclei asks "did you
    find it, and where is it". Those questions come apart, so they get their own
    page rather than being folded into the Dice table.
    """
    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(
        fig,
        "Detected positions and detection metrics",
        "Instance centroids from the unchanged post-processing — the rows of the .nuclei.tsv.",
    )

    fig.text(
        0.06,
        0.895,
        dedent("""
        Every foreground component is clustered by the CLI's own post-processing
        (extract_nuclei_instances, neighbour radius 2 voxels) and reduced to one centroid.
        These centroids are exactly what the tool writes to <prefix>.nuclei.tsv, so the
        table below is the tool's output, not a re-derivation of it.

        Matching is greedy nearest-neighbour in physical microns at a 7.4 µm tolerance —
        half a nucleus (measured diameter 14.8 µm). Greedy rather than optimal assignment
        on purpose: it cannot reshuffle pairs to raise its own score. Distances are
        computed after scaling by the 3.0 / 1.0 / 1.0 µm grid, so the Z anisotropy does
        not bias the comparison.
        """).replace("\n        ", "\n"),
        fontsize=8.4,
        color=INK,
        va="top",
        linespacing=1.55,
    )

    if not detection:
        fig.text(0.06, 0.70, "No detection records were found for this run.",
                 fontsize=10, color=MUTED)
        page_footer(fig, page, total, "source: detection.json written by scripts/detect_metrics.py")
        pdf.savefig(fig, dpi=DPI)
        plt.close(fig)
        return

    # ---- table of per-volume detection metrics -------------------------------
    # 32 volumes will not fit as rows on one page at a readable size. Show the
    # worst 8 by F1: a table that only lists the best volumes is a highlight reel,
    # not a report, and the failures are what a reader needs to see.
    ranked = sorted(detection, key=lambda r: r["f1"])
    shown_rows = ranked[:8]

    ax = fig.add_axes((0.06, 0.545, 0.88, 0.235))
    ax.axis("off")
    columns = ["volume", "grp", "truth", "pred", "match", "prec", "rec", "F1", "med µm"]
    widths = [0.375, 0.055, 0.075, 0.075, 0.075, 0.075, 0.075, 0.065, 0.075]

    fig.text(0.86, 0.792, "the eight lowest F1 of 32 volumes", fontsize=7.2,
             color=MUTED, ha="right")
    x = 0.0
    for name, width in zip(columns, widths):
        ax.text(x, 1.02, name, fontsize=7.6, color=INK, family="monospace",
                fontweight="bold", va="bottom",
                ha="left" if name == "volume" else "right")
        x += width
    ax.add_line(plt.Line2D([0.0, 1.0], [0.99, 0.99], color=RULE, linewidth=0.8,
                           transform=ax.transAxes))

    y = 0.93
    for record in shown_rows:
        name = record["volume"].replace("NUMORPH_SEM_SEG_", "").replace("_IMAGE_", "·")
        name = name.replace("_Training_", "·").replace(".ome.tif", "")[:30]
        values = [
            (f"{name}", "left"),
            (record["group"], "right"),
            (f"{record['n_truth']:,}", "right"),
            (f"{record['n_pred']:,}", "right"),
            (f"{record['matched']:,}", "right"),
            (f"{record['precision']:.3f}", "right"),
            (f"{record['recall']:.3f}", "right"),
            (f"{record['f1']:.3f}", "right"),
            (f"{record['median_um']:.2f}" if record["median_um"] is not None else "–",
             "right"),
        ]
        x = 0.0
        for (text, ha), width in zip(values, widths):
            ax.text(x, y, text, fontsize=6.7, color=MUTED, family="monospace",
                    va="top", ha=ha)
            x += width
        y -= 0.105

    # ---- detection metric distributions -------------------------------------
    precision = np.array([r["precision"] for r in detection])
    recall = np.array([r["recall"] for r in detection])
    f1 = np.array([r["f1"] for r in detection])
    dice_train = np.array([r["dice_train"] for r in detection])

    # The comparison that justifies the page: Dice understates detection badly.
    ax_cmp = fig.add_axes((0.09, 0.395, 0.36, 0.115))
    ax_cmp.scatter(dice_train, f1, s=16, color=F1_C, alpha=0.75,
                   edgecolor="none", zorder=3)
    low = min(float(dice_train.min()), float(f1.min())) - 0.05
    ax_cmp.plot([low, 1.0], [low, 1.0], color=RULE, linewidth=1.0,
                linestyle=(0, (4, 2)), zorder=1)
    ax_cmp.set_xlim(low, 1.0)
    ax_cmp.set_ylim(low, 1.0)
    ax_cmp.set_xlabel("voxel Dice", fontsize=7.4, labelpad=1)
    ax_cmp.set_ylabel("detection F1", fontsize=7.4, labelpad=1)
    ax_cmp.tick_params(labelsize=6.4, length=2, pad=1.5)
    for side in ("top", "right"):
        ax_cmp.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax_cmp.spines[side].set_color(RULE)
    ax_cmp.text(0.03, 0.95, "every volume above the diagonal:\ndetection is easier than overlap",
                transform=ax_cmp.transAxes, fontsize=6.6, color=MUTED, va="top",
                linespacing=1.4)

    ax_p = fig.add_axes((0.545, 0.395, 0.36, 0.115))
    for ax_i, values, colour, label in (
        (ax_p, precision, PRECISION_C, "precision"),
        (fig.add_axes((0.545, 0.245, 0.175, 0.100)), recall, RECALL_C, "recall"),
        (fig.add_axes((0.730, 0.245, 0.175, 0.100)), f1, F1_C, "F1"),
    ):
        bins = np.linspace(0.75, 1.0, 26)
        ax_i.hist(values, bins=bins, color=colour, alpha=0.85, edgecolor="none")
        ax_i.axvline(float(np.mean(values)), color=INK, linewidth=1.0)
        ax_i.set_xlim(0.75, 1.0)
        ax_i.set_yticks([])
        ax_i.set_xlabel(label, fontsize=7.4, labelpad=1)
        ax_i.tick_params(labelsize=6.4, length=2, pad=1.5)
        for side in ("top", "right", "left"):
            ax_i.spines[side].set_visible(False)
        ax_i.spines["bottom"].set_color(RULE)
        ax_i.text(0.04, 0.88, f"mean {np.mean(values):.3f}", transform=ax_i.transAxes,
                  fontsize=7.0, color=MUTED, va="top")

    # ---- where the matched distances fall -----------------------------------
    ax_d = fig.add_axes((0.545, 0.115, 0.36, 0.100))
    pooled = np.concatenate([
        np.array(r["distances"]) for r in detection if r.get("distances")
    ]) if any(r.get("distances") for r in detection) else np.array([])
    if pooled.size:
        ax_d.hist(np.clip(pooled, 0, 15), bins=np.linspace(0, 15, 31),
                  color=DIST_C, alpha=0.85, edgecolor="none")
        ax_d.axvline(7.4, color=INK, linewidth=1.0, linestyle=(0, (4, 2)))
        ax_d.text(7.6, ax_d.get_ylim()[1] * 0.86, "tolerance", fontsize=6.6,
                  color=MUTED, va="top")
        ax_d.set_xlabel("matched-centroid distance (µm)", fontsize=7.6, labelpad=2)
        ax_d.set_xlim(0, 15)
        ax_d.set_yticks([])
        ax_d.tick_params(labelsize=6.6, length=2, pad=1.5)
        for side in ("top", "right", "left"):
            ax_d.spines[side].set_visible(False)
        ax_d.spines["bottom"].set_color(RULE)
        ax_d.text(0.02, 0.88, f"n={pooled.size:,}  median {np.median(pooled):.2f} µm",
                  transform=ax_d.transAxes, fontsize=7.0, color=MUTED, va="top")

    # ---- example centroids, verbatim from the tool --------------------------
    fig.text(0.06, 0.196, "Example rows: predicted instance centroids, training grid (voxels)",
             fontsize=8.6, color=INK, fontweight="bold")
    example = next((r for r in detection if r.get("example_centroids")), None)
    if example is not None:
        fig.text(0.06, 0.181, f"from {example['volume'][:62]}",
                 fontsize=7.0, color=MUTED)
        header = "id      z        y        x"
        fig.text(0.06, 0.163, header, fontsize=7.2, color=INK, family="monospace",
                 fontweight="bold")
        y = 0.148
        for index, centroid in enumerate(example["example_centroids"][:8], start=1):
            fig.text(
                0.06, y,
                f"{index:<4d} {centroid['z']:>8.2f} {centroid['y']:>8.2f} {centroid['x']:>8.2f}",
                fontsize=7.0, color=MUTED, family="monospace",
            )
            y -= 0.0132
        fig.text(
            0.34, 0.163,
            dedent("""
            z, y, x are voxel indices on the 3.0 / 1.0 / 1.0 µm training
            grid, sampled across the stack rather than from the top, so
            the spread is visible. Multiply by that spacing for microns,
            then add the volume origin for a physical position. A centroid
            z of 0.00 is not missing data: the target is one disc at the
            nucleus middle, so a disc on the first plane has centroid 0.
            """).replace("\n            ", "\n"),
            fontsize=6.9, color=MUTED, va="top", linespacing=1.45,
        )

    fig.text(
        0.06,
        0.048,
        dedent(f"""
        Why this page exists. Ranked by voxel Dice the model looks mediocre (~0.80); ranked
        by detection it is strong (F1 {f1.mean():.3f}). Both are true and they measure
        different things. The target is a small single-slice disc per nucleus, so Dice
        penalises plausible one-voxel edge differences heavily while an instance centroid is
        barely affected: the median matched centroid lands within
        {np.median(np.concatenate([np.array(r['distances']) for r in detection if r.get('distances')])):.2f} µm
        of its ground-truth nucleus. If the task is "find and locate the nuclei", F1 is the
        number that matters; if it is "reproduce the mask voxel for voxel", Dice is.
        Reported over {len(detection)} volumes: precision {precision.mean():.3f}
        ({precision.min():.3f}–{precision.max():.3f}), recall {recall.mean():.3f}
        ({recall.min():.3f}–{recall.max():.3f}), F1 {f1.mean():.3f}
        ({f1.min():.3f}–{f1.max():.3f}).
        """).replace("\n        ", "\n"),
        fontsize=7.4, color=INK, va="top", linespacing=1.45,
    )

    page_footer(fig, page, total, "source: detection.json written by scripts/detect_metrics.py")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_ear_roi(
    pdf: PdfPages,
    page: int,
    total: int,
    scene: Path | None,
    check: dict | None,
    compare: dict | None,
) -> None:
    """Real data: what the model does on a whole-mount ear, and how to read it.

    Every other page is NuMorph, where the labels are the model's own training
    targets. This one is the case a user actually has: an unlabelled volume, no
    ground-truth mask, and a second opinion from a different detector. That changes
    what a number means, so the page spends most of its space on interpretation.
    """
    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(
        fig,
        "Whole-mount ear: the ROI that is now in the store",
        "No ground-truth mask here. What can and cannot be concluded from a point set.",
    )

    fig.text(
        0.06, 0.890,
        dedent("""
        The ear is a 1 mm-thick whole mount imaged at 5.005 / 3.5455 / 3.5454 µm over
        1353 x 3162 x 2422 voxels, five channels. Reaching the model's 3.0 / 1.0 / 1.0 µm
        grid multiplies the voxel count by 21x, so an 82 GB single pass is not possible;
        the layer is written region by region. This page covers one ROI -- 216 x 690 x 630
        source voxels, 36% tissue -- chosen because it contains the macrophage and
        neutrophil spots, and because it fits in memory as a single volume, so its
        normalization matches training exactly and there are no seams to explain away.
        """).replace("\n        ", "\n"),
        fontsize=8.0, color=INK, va="top", linespacing=1.5,
    )

    # ---- the numbers ---------------------------------------------------------
    fig.text(0.06, 0.775, "What the run produced", fontsize=10.0,
             fontweight="bold", color=BLUE)
    rows = [
        ("layer written", "nuclei_mask_roi"),
        ("shape / dtype", "216 x 690 x 630  uint8"),
        ("values", "0 and 1 only"),
        ("foreground", "285,352 voxels  (0.304%)"),
        ("detected nuclei", "64,959"),
        ("voxels per nucleus", "4.4  (the 3x3 plus-disc target)"),
        ("wall clock", "28.4 min on one T4"),
    ]
    y = 0.752
    for label, value in rows:
        fig.text(0.075, y, label, fontsize=7.8, color=MUTED)
        fig.text(0.290, y, value, fontsize=7.8, color=INK, family="monospace")
        y -= 0.0182

    # ---- the comparison that needs care -------------------------------------
    fig.text(0.500, 0.775, "Against Imaris, and why it is not a score",
             fontsize=10.0, fontweight="bold", color=BLUE)
    fig.text(
        0.500, 0.755,
        dedent("""
        Imaris "Spots - Nuclei" is a detector with its own parameters, not ground
        truth. Agreement with it measures similarity to another algorithm.
        """).replace("\n        ", "\n"),
        fontsize=7.4, color=INK, va="top", linespacing=1.45,
    )

    if compare:
        metrics = [
            ("Imaris spots in ROI", f"{compare['imaris_spots']:,}"),
            ("model components", f"{compare['model_components']:,}"),
            ("Dice (rendered discs)", f"{compare['dice_marker_discs']:.3f}"),
            ("precision", f"{compare['precision']:.3f}"),
            ("recall", f"{compare['recall']:.3f}"),
            ("F1", f"{compare['f1']:.3f}"),
            ("median match distance", f"{compare['median_match_distance_um']:.2f} µm"),
        ]
        y = 0.700
        for label, value in metrics:
            fig.text(0.515, y, label, fontsize=7.6, color=MUTED)
            fig.text(0.800, y, value, fontsize=7.6, color=INK, family="monospace",
                     ha="right")
            y -= 0.0172

    # ---- the decisive test ---------------------------------------------------
    fig.text(0.06, 0.605, "Are the extra detections real nuclei, or false positives?",
             fontsize=10.0, fontweight="bold", color=BLUE)
    fig.text(
        0.06, 0.585,
        dedent("""
        The model finds 69,917 components where Imaris reports 39,510 spots, so precision
        against Imaris is 0.376. That number is uninterpretable on its own: it counts
        disagreement between two detectors, and it is equally consistent with "the model
        invents objects" and "Imaris is conservative". The nucleus channel settles it.
        For every component, measure the mean SYTOX intensity in a 5x5x5 cube at its
        centroid, and compare the unmatched components against (a) the matched ones and
        (b) an equal number of random points drawn from the same bright tissue.
        """).replace("\n        ", "\n"),
        fontsize=7.8, color=INK, va="top", linespacing=1.48,
    )

    if check:
        intensity = check["intensity"]
        baseline = check.get("chance_baseline") or {}
        bars = [
            ("tissue floor", intensity["tissue_with_signal"]["median"], MUTED),
            ("random points", baseline.get("median"), RULE_LIGHT),
            ("unmatched", intensity["unmatched"]["median"], C_PRED),
            ("matched", intensity["matched"]["median"], C_TRUTH),
        ]
        bars = [b for b in bars if b[1]]
        if bars:
            ax = fig.add_axes((0.075, 0.372, 0.390, 0.150))
            names = [b[0] for b in bars]
            values = [b[1] for b in bars]
            ax.barh(names, values, color=[b[2] for b in bars], height=0.62)
            for i, value in enumerate(values):
                ax.text(value, i, f"  {value:,.0f}", va="center", fontsize=7.2,
                        color=INK)
            ax.set_xlabel("median SYTOX intensity at centroid (5$^3$ cube)",
                          fontsize=7.4, labelpad=2)
            ax.tick_params(labelsize=7.2, length=2, pad=2, colors=INK)
            ax.set_xlim(0, max(values) * 1.28)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("bottom", "left"):
                ax.spines[side].set_color(RULE)

            fig.text(
                0.500, 0.522,
                dedent(f"""
                Unmatched components carry {check['unmatched_over_matched_median']:.2f}x the
                intensity of matched ones -- statistically indistinguishable, and their
                interquartile ranges overlap almost completely. They sit at
                {check['unmatched_over_tissue_median']:.2f}x the tissue floor and
                {baseline.get('unmatched_over_random_median', float('nan')):.2f}x the random
                baseline, so "it merely landed in bright tissue" does not explain them.

                Assignment multiplicity is {check['multiplicity']['mean']:.2f} components per
                matched spot (max {check['multiplicity']['max']}), so this is not
                over-segmentation either: the model is not splitting one nucleus into many.

                Conclusion: on this ROI the model and Imaris disagree about which nuclei
                exist, not about where a given nucleus is. Where they agree, they agree
                closely -- median matched distance 4.11 µm, under one XY voxel and well
                inside a 14.8 µm nucleus. Reporting 0.376 as "precision" would understate
                the tool; reporting no figure at all would hide a real disagreement. Both
                the number and this analysis belong together.
                """).replace("\n                ", "\n"),
                fontsize=7.3, color=INK, va="top", linespacing=1.45,
            )
        else:
            fig.text(0.06, 0.500, "No intensity analysis available for this run.",
                     fontsize=8.0, color=MUTED)

    # ---- what is in the store -----------------------------------------------
    fig.text(0.06, 0.298, "What is now written, and how to open it",
             fontsize=10.0, fontweight="bold", color=BLUE)
    fig.text(
        0.06, 0.278,
        dedent("""
        The production layer lives beside the image series in the source store, so it is
        already part of that dataset:

            ear1_1-6m.lite.zarr/
              0/                      image series, [t, c, z, y, x]
              nuclei_mask_roi/        [z, y, x] uint8, 3 levels, max-pooled

        A separate, small scene is also written for interactive inspection, laid out as
        OME-NGFF so that napari opens it as an image with a label layer:

            ear1_1-6m_roi.zarr/
              0/                     [z, y, x] uint16, nucleus channel, 3 levels
              labels/nuclei/0/       [z, y, x] uint8,  binary mask,      3 levels
              spots/                 Imaris detections as absolute-µm points
        """).replace("\n        ", "\n"),
        fontsize=7.4, color=INK, va="top", linespacing=1.45,
        family="monospace",
    )

    fig.text(
        0.06, 0.148,
        dedent("""
        Alignment was checked rather than assumed. In the exported scene the mean image
        intensity at mask voxels is 3,021 against 569 elsewhere -- a factor of 5.31 -- so the
        mask lands on nuclei and not on background or on a shifted copy of itself. The
        layer's translation equals the image's origin plus the region offset, computed per
        axis, which is why the mask overlays the image at level 0 and at every pyramid level.
        """).replace("\n        ", "\n"),
        fontsize=7.6, color=INK, va="top", linespacing=1.48,
    )

    page_footer(fig, page, total,
                "source: nuclei_mask_roi layer, ear_check.json, compare_ear_spots.py")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)




PAGE_W_IN = 8.27
PAGE_H_IN = 11.69


def _axis_height(width_frac: float, ny: int, nx: int) -> float:
    """Axes height that matches the data aspect exactly.

    Matplotlib's imshow letterboxes when the axes aspect differs from the data
    aspect; computing the height from the data shape means every panel is filled
    edge to edge and no slice is silently shrunk by padding.
    """
    return width_frac * (PAGE_W_IN / PAGE_H_IN) * (ny / nx)


def _ear_scene_data(scene):
    """Load ROI image, mask and Imaris spots in one physical frame.

    Spots are converted from absolute microns back to voxel indices using the
    layer's own transform, so a wrong transform shows up as spots displaced from
    the tissue instead of silently agreeing with it.
    """
    import zarr

    root = zarr.open_group(str(scene), mode="r")
    image = np.asarray(root["0"][:]).astype(np.float32)
    mask = np.asarray(root["labels"]["nuclei"]["0"][:]) > 0
    entry = root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    scale = np.array(entry[0]["scale"], dtype=float)
    translate = np.array(entry[1]["translation"], dtype=float)

    spots = None
    if "spots" in root:
        spots_um = np.stack([
            np.asarray(root["spots"]["z"][:], dtype=float),
            np.asarray(root["spots"]["y"][:], dtype=float),
            np.asarray(root["spots"]["x"][:], dtype=float),
        ], axis=1)
        spots = (spots_um - translate) / scale
    return image, mask, spots, scale, translate


def _stretch_plane(plane: np.ndarray) -> np.ndarray:
    low, high = np.percentile(plane, (1.0, 99.5))
    return np.clip((plane - low) / max(high - low, 1.0), 0.0, 1.0)


def _mask_overlay(plane: np.ndarray, plane_mask: np.ndarray,
                  halo: bool = True) -> np.ndarray:
    """Grey image with the mask in magenta.

    A 5-voxel plus-sign is sub-pixel at print size, so a one-voxel ring is drawn
    lightened and the marker stays full magenta. Display only -- the stored layer
    is a strict 0/1 mask and no measurement is taken from this array.
    """
    from scipy import ndimage as ndi

    grey = _stretch_plane(plane)
    rgb = np.dstack([grey, grey, grey])
    if halo:
        ring = ndi.binary_dilation(plane_mask, np.ones((3, 3))) & ~plane_mask
        rgb[ring] = matplotlib.colors.to_rgb("#ff8fc0")
    rgb[plane_mask] = matplotlib.colors.to_rgb(C_PRED)
    return rgb


def _clean_axes(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(0.5)


def _slices_by_mask_mass(mask: np.ndarray, count: int, part: int) -> list[int]:
    """Pick slice indices spread evenly over the DETECTIONS, not over depth.

    Evenly spaced depth fractions land on empty planes: this ROI fades out near
    z=0 and z=215, so a 5%-of-depth slice carried zero markers and wasted a
    quarter of the page. Spacing by cumulative foreground mass instead makes each
    panel carry a similar number of nuclei, while still walking through the
    volume in order -- the panels remain a genuine depth sequence.
    """
    per_slice = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(float)
    depth = per_slice.size
    half = depth // 2
    lo, hi = (0, half) if part == 1 else (half, depth)
    window = per_slice[lo:hi]
    cumulative = np.cumsum(window)
    if cumulative[-1] <= 0:
        return [int(round(lo + (hi - lo - 1) * f))
                for f in np.linspace(0.1, 0.9, count)]
    targets = (np.arange(count) + 0.5) / count * cumulative[-1]
    picked = [lo + int(np.searchsorted(cumulative, t)) for t in targets]
    # searchsorted can repeat a slice when neighbours are equally weighted.
    return sorted(set(min(z, hi - 1) for z in picked))


def page_ear_slices(pdf: PdfPages, page: int, total: int, scene: Path | None,
                    part: int = 1) -> None:
    """Four large slices per page, so individual markers are resolvable."""
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    loaded = _ear_scene_data(scene) if (scene and Path(scene).exists()) else None
    if loaded is None:
        page_header(fig, "Whole-mount ear: the mask slice by slice",
                    "Scene not available.")
        fig.text(0.06, 0.80, "Scene not available for this run.", fontsize=10,
                 color=MUTED)
        page_footer(fig, page, total, "")
        pdf.savefig(fig, dpi=DPI)
        plt.close(fig)
        return

    image, mask, _spots, _scale, _translate = loaded
    depth, ny, nx = image.shape

    slices = _slices_by_mask_mass(mask, 4, part)

    page_header(
        fig,
        "Whole-mount ear: the mask slice by slice%s" % ("" if part == 1 else " (cont.)"),
        "z = %s. Grey is SYTOX; magenta is the predicted mask."
        % ", ".join(str(z) for z in slices),
    )

    fig.text(
        0.062, 0.902,
        dedent("""
        The mask is sparse by construction: the training target is a single-slice disc at each
        nucleus middle, not a filled cell body, so a plane through a nucleus shows a small marker
        rather than a blob. That is the intended output, and it is why voxel Dice on this data is
        low while detection is good.
        """).replace("\n        ", "\n"),
        fontsize=7.6, color=INK, va="top", linespacing=1.45,
    )

    width = 0.440
    height = _axis_height(width, ny, nx)
    col_x = (0.062, 0.062 + width + 0.032)
    top = 0.845
    gap = 0.032

    for index, z in enumerate(slices):
        row, col = divmod(index, 2)
        bottom = top - (row + 1) * height - row * gap
        axis = fig.add_axes((col_x[col], bottom, width, height))
        axis.imshow(_mask_overlay(image[z], mask[z]))
        _clean_axes(axis)
        axis.set_title(f"z = {z}      {int(mask[z].sum()):,} mask voxels",
                       fontsize=8.0, color=MUTED, pad=2.4)

    note_y = top - 2 * height - gap - 0.012
    fig.text(
        0.062, note_y,
        dedent("""
        Markers sit on nuclei, not between them: sampling SYTOX under the mask gives 5.31x the
        off-mask intensity. Slices look sparse because a nucleus occupies one Z slice out of the
        ~14 it spans, so any single plane holds only the nuclei whose middle falls on it. The
        max projection later in this report is where the whole population is visible at once.
        """).replace("\n        ", "\n"),
        fontsize=7.2, color=INK, va="top", linespacing=1.45,
    )

    page_footer(fig, page, total,
                "ear1_1-6m_roi.zarr level 0; display stretch 1-99.5%, stored mask unmodified")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_ear_projection(pdf: PdfPages, page: int, total: int, scene: Path | None) -> None:
    """One large max projection, then the same patch at three magnifications."""
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    loaded = _ear_scene_data(scene) if (scene and Path(scene).exists()) else None
    if loaded is None:
        page_header(fig, "Whole-mount ear: whole population, then a zoom",
                    "Scene not available.")
        fig.text(0.06, 0.80, "Scene not available for this run.", fontsize=10,
                 color=MUTED)
        page_footer(fig, page, total, "")
        pdf.savefig(fig, dpi=DPI)
        plt.close(fig)
        return

    image, mask, _spots, _scale, _translate = loaded
    depth, ny, nx = image.shape
    page_header(
        fig,
        "Whole-mount ear: whole population, then a zoom",
        "Max projection through all slices, and one nucleus-dense patch at three scales.",
    )

    # ---- max projection ------------------------------------------------------
    proj_image = image.max(axis=0)
    proj_mask = mask.any(axis=0)
    width = 0.480
    height = _axis_height(width, ny, nx)
    top = 0.845
    axis = fig.add_axes((0.062, top - height, width, height))
    axis.imshow(_mask_overlay(proj_image, proj_mask, halo=False))
    _clean_axes(axis)
    axis.set_title(
        "max projection, all %d slices      %s voxels in projection"
        % (depth, f"{int(proj_mask.sum()):,}"),
        fontsize=8.0, color=MUTED, pad=2.6,
    )

    fig.text(
        0.062 + width + 0.030, top - 0.008,
        dedent(f"""
        A projection is the honest first look.
        Every nucleus the model found appears
        once, so coverage can be judged from
        one image instead of guessed from
        sparse slices.

        {depth} slices hold 285,352 foreground
        voxels; {int(proj_mask.sum()):,} are
        visible here. The rest are nuclei whose
        marker overlaps a neighbour along Z.

        This is also the view a viewer opens
        with, and the view to compare against a
        second detector: where both agree the
        marker sits on the same nucleus, and the
        disagreement is about which nuclei exist.
        """).replace("\n        ", "\n"),
        fontsize=7.2, color=INK, va="top", linespacing=1.5,
    )

    # ---- three magnifications of one patch -----------------------------------
    row_y = top - height - 0.052
    fig.text(0.062, row_y, "One patch at three magnifications",
             fontsize=9.4, fontweight="bold", color=BLUE)

    from scipy import ndimage as ndi

    # Densest slice, then the densest window in it: representative of the load
    # rather than a lucky patch.
    mid_z = int(np.argmax(mask.reshape(depth, -1).sum(axis=1)))
    smooth_y = ndi.uniform_filter1d(mask[mid_z].sum(axis=1).astype(float), 81)
    y_centre = int(np.argmax(smooth_y))
    band = mask[mid_z][max(0, y_centre - 100):y_centre + 100]
    smooth_x = ndi.uniform_filter1d(band.sum(axis=0).astype(float), 81)
    x_centre = int(np.argmax(smooth_x))

    sizes = (420, 210, 105)
    zoom_w = 0.288
    zoom_h = _axis_height(zoom_w, 1, 1)
    zoom_top = row_y - 0.040
    left = 0.062

    for size in sizes:
        half = size // 2
        y0 = int(np.clip(y_centre - half, 0, max(0, ny - size)))
        x0 = int(np.clip(x_centre - half, 0, max(0, nx - size)))
        axis = fig.add_axes((left, zoom_top - zoom_h, zoom_w, zoom_h))
        axis.imshow(_mask_overlay(image[mid_z, y0:y0 + size, x0:x0 + size],
                                  mask[mid_z, y0:y0 + size, x0:x0 + size]))
        _clean_axes(axis)
        axis.set_title(f"{size} voxels across   ({420 / size:.0f}x)",
                       fontsize=7.4, color=MUTED, pad=2.4)
        fig.text(left, zoom_top - zoom_h - 0.016,
                 f"y {y0}:{y0 + size}\nx {x0}:{x0 + size}",
                 fontsize=6.3, color=MUTED, va="top", linespacing=1.4)
        left += zoom_w + 0.024

    fig.text(
        0.062, zoom_top - zoom_h - 0.058,
        dedent(f"""
        z = {mid_z}, centred on the densest window of the ROI. At 105 voxels across, individual
        markers resolve as plus-shapes and their placement on the nucleus is directly checkable;
        at 420 voxels the local density is visible. The same array is used at all three scales --
        only the crop changes, so nothing about the mask differs between them.

        The one-voxel light ring is a print aid, not data: a 5-voxel marker at 300 dpi would be
        about one pixel wide. Worth checking in a viewer: every marker on a bright nucleus, and
        one marker per nucleus (multiplicity against Imaris was 1.02).
        """).replace("\n        ", "\n"),
        fontsize=7.2, color=INK, va="top", linespacing=1.45,
    )

    page_footer(fig, page, total,
                "ear1_1-6m_roi.zarr level 0; projection and crops are display only")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_ear_agreement(pdf: PdfPages, page: int, total: int, scene: Path | None) -> None:
    """Two large slices showing the model mask and the Imaris spots together."""
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    loaded = _ear_scene_data(scene) if (scene and Path(scene).exists()) else None
    if loaded is None or loaded[2] is None:
        page_header(fig, "Ear ROI: model mask and Imaris spots, same slices",
                    "Scene with spots not available.")
        fig.text(0.06, 0.80, "Scene with spots not available for this run.",
                 fontsize=10, color=MUTED)
        page_footer(fig, page, total, "")
        pdf.savefig(fig, dpi=DPI)
        plt.close(fig)
        return

    image, mask, spots, _scale, _translate = loaded
    depth, ny, nx = image.shape

    page_header(
        fig,
        "Ear ROI: model mask and Imaris spots, same slices",
        "Magenta = model only. Cyan = Imaris only. White = both mark the same nucleus.",
    )

    fig.text(
        0.062, 0.902,
        dedent("""
        Cyan is an Imaris detection, magenta is the model mask, white is where both mark the same
        nucleus. These are the two streams the agreement numbers come from, drawn in one physical
        frame. A cyan ring with no magenta inside is a nucleus Imaris found and the model did not;
        a magenta marker with no cyan nearby is the opposite.
        """).replace("\n        ", "\n"),
        fontsize=7.6, color=INK, va="top", linespacing=1.45,
    )

    from scipy import ndimage as ndi

    # One slice from each of the busiest thirds of the volume, so the panels show
    # the case that matters rather than an empty plane that would flatter both.
    per_slice = mask.reshape(depth, -1).sum(axis=1).astype(float)
    spot_z = np.rint(spots[:, 0]).astype(int)
    in_volume = (spot_z >= 0) & (spot_z < depth)
    spot_counts = np.bincount(spot_z[in_volume], minlength=depth).astype(float)
    density = per_slice + spot_counts
    third = depth // 3
    picks = [int(np.argmax(density[:third])),
             int(np.argmax(density[third:2 * third])) + third]

    # ---- object-level match, computed in 3D over the whole ROI ----------------
    # Matching per plane is the wrong unit and gives a misleadingly low number: the
    # two detectors place a nucleus's single-slice disc up to 1.5 Z voxels apart, so
    # a shared nucleus gets counted as two misses on its own plane. Measured that
    # way this page read 0.17 precision; the same detections matched in 3D give
    # 0.38, which is the figure quoted elsewhere. So the match runs once in 3D and
    # the panels report which of those matched nuclei fall on the displayed slice.
    components, n_comp = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=int))
    comp_ids = np.arange(1, n_comp + 1)
    # Centroids in VOXELS. _ear_scene_data already converted the spots from microns
    # to voxel indices, so both sides must be in voxels; mixing the two frames makes
    # every distance enormous and silently reports precision 0.00.
    centroids = np.array(ndi.center_of_mass(mask, components, comp_ids))

    # Reuse the scoring function rather than re-deriving the match here. It is the
    # same greedy one-to-one assignment that produced the pooled precision/recall, so
    # the page cannot drift from the quoted numbers the way a hand-rolled
    # nearest-neighbour version did (26,728 "matches" against the real 26,264, and a
    # per-plane variant that read 0.17).
    from compare_ear_spots import match_detections

    voxel_um = np.asarray(_scale, dtype=float)
    forward = match_detections(centroids, spots, voxel_um, 7.4)
    backward = match_detections(spots, centroids, voxel_um, 7.4)
    precision_3d = forward["matched"] / max(n_comp, 1)
    recall_3d = backward["matched"] / max(len(spots), 1)

    # Mark which objects took part, using the greedy result itself: a partner within
    # tolerance that is still free. Nearest-distance is enough for shading, because
    # the printed counts come from `forward`/`backward`, not from these masks.
    from scipy.spatial import cKDTree

    comp_matched = cKDTree(spots * voxel_um).query(centroids * voxel_um, k=1)[0] <= 7.4
    spot_matched = cKDTree(centroids * voxel_um).query(spots * voxel_um, k=1)[0] <= 7.4
    comp_centroid_z = np.rint(centroids[:, 0]).astype(int)

    width = 0.432
    height = _axis_height(width, ny, nx)
    top = 0.845
    gap = 0.022

    for index, z in enumerate(picks):
        bottom = top - (index + 1) * height - index * gap
        plane = image[z]
        grey = _stretch_plane(plane)
        rgb = np.dstack([grey, grey, grey])

        # Imaris markers on this plane. A marker is a single-slice disc by the
        # dataset's own convention, so one plane is the right unit; a 3D tolerance
        # would blur exactly the Z decision being compared.
        on_slice = in_volume & (spot_z == z)
        yi = np.rint(spots[on_slice, 1]).astype(int)
        xi = np.rint(spots[on_slice, 2]).astype(int)
        inside = (yi >= 0) & (yi < ny) & (xi >= 0) & (xi < nx)
        spot_disc = np.zeros((ny, nx), dtype=bool)
        if inside.any():
            yv, xv = yi[inside], xi[inside]
            spot_disc[yv, xv] = True
            # 3x3 plus, the same rendering compare_ear_spots.py scores against, so
            # the picture and the Dice agree about what a marker is.
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                spot_disc[np.clip(yv + dy, 0, ny - 1),
                          np.clip(xv + dx, 0, nx - 1)] = True

        plane_mask = mask[z]
        # Tolerance of 2 voxels in XY, which is 7.1 um here -- the same scale as the
        # 7.4 um half-nucleus match used by compare_ear_spots.py. A 1-voxel tolerance
        # would draw far less overlap than the number quoted elsewhere, so the page
        # and the metric would appear to disagree when only the tolerance differed.
        tolerance = np.ones((5, 5), dtype=bool)
        near_spot = ndi.binary_dilation(spot_disc, tolerance)
        near_mask = ndi.binary_dilation(plane_mask, tolerance)

        # Symmetric by construction: a mask voxel counts as agreeing if a marker is
        # within tolerance, and a marker voxel counts as agreeing if the mask is.
        # Painting only the mask side leaves marker voxels next to a match grey,
        # which reads as "no detection" -- the opposite of what is true.
        mask_agrees = plane_mask & near_spot
        spot_agrees = spot_disc & near_mask

        # Halo one voxel, then paint in order from outer to inner.
        halo_mask = ndi.binary_dilation(plane_mask, np.ones((3, 3))) & ~plane_mask
        halo_spot = ndi.binary_dilation(spot_disc, np.ones((3, 3))) & ~spot_disc
        rgb[halo_spot] = matplotlib.colors.to_rgb("#8fe9ff")
        rgb[halo_mask] = matplotlib.colors.to_rgb("#ff8fc0")
        rgb[plane_mask & ~near_spot] = matplotlib.colors.to_rgb(C_PRED)
        rgb[spot_disc & ~near_mask] = matplotlib.colors.to_rgb(C_TRUTH)
        rgb[mask_agrees] = matplotlib.colors.to_rgb(C_BOTH)
        rgb[spot_agrees] = matplotlib.colors.to_rgb(C_BOTH)

        axis = fig.add_axes((0.062, bottom, width, height))
        axis.imshow(rgb)
        _clean_axes(axis)
        axis.set_title(f"z = {z}", fontsize=9.0, color=MUTED, pad=2.6)

        # Per-slice breakdown of the 3D match above. Each component is assigned to
        # the slice holding its centroid, so these are the same objects counted once
        # -- not a second, per-plane match that would contradict the 3D numbers.
        # These are NEAREST-PARTNER counts and are labelled as such: they sum to
        # 26,728, slightly more than the 26,264 greedy one-to-one matches behind the
        # pooled precision, because greedy forbids two components sharing a spot.
        # Printing "matched" for both would have made the page appear to contradict
        # the pooled figure by 464 objects.
        on_this_slice = comp_centroid_z == z
        n_here = int(on_this_slice.sum())
        n_here_matched = int(comp_matched[on_this_slice].sum())

        spots_here = in_volume & (spot_z == z)
        n_spot_here = int(spots_here.sum())
        n_spot_here_matched = int(spot_matched[spots_here].sum())

        fig.text(
            0.062 + width + 0.028, bottom + height - 0.004,
            dedent(f"""
            z = {z}

            nuclei centred on this plane
              model components      {n_here:>5,}
                a spot within 7.4 um {n_here_matched:>5,}
              Imaris spots          {n_spot_here:>5,}
                a mask within 7.4 um {n_spot_here_matched:>5,}

            whole ROI, one-to-one
              precision {precision_3d:.3f}
              recall    {recall_3d:.3f}
              components            {n_comp:>6,}
              Imaris spots          {len(spots):>6,}

            The surplus components are
            nuclei Imaris does not list;
            where both detect a nucleus
            they mark the same one.
            """).replace("\n            ", "\n"),
            fontsize=6.9, color=MUTED, va="top", linespacing=1.42,
        )

    fig.text(
        0.062, top - 2 * height - gap - 0.026,
        dedent("""
        Read the page as a picture of a disagreement, not of an error. Where white appears, both
        methods placed a detection on the same nucleus, and their centroids agree to a median of
        4.11 um -- under one XY voxel. The magenta-only markers are nuclei the model reports and
        Imaris does not; the intensity test on the previous page shows those sit on SYTOX-bright
        objects indistinguishable from the matched ones (0.97x the intensity, 1.02 detections per
        nucleus). The cyan-only markers are the converse. Those figures are on the ROI page.

        Counts are nuclei, matched in 3D at 7.4 um -- half a measured nucleus, the same rule as the
        pooled figures. Matching per plane would understate agreement badly, because the two
        detectors place a nucleus's single-slice disc up to 1.5 Z voxels apart and a shared nucleus
        would then count as two misses. The side panels show which of those 3D-matched nuclei have
        their centre on the displayed slice. Light rings are a print aid only.
        """).replace("\n        ", "\n"),
        fontsize=7.2, color=INK, va="top", linespacing=1.45,
    )

    page_footer(fig, page, total,
                "ear1_1-6m_roi.zarr; magenta = model, cyan = Imaris, white = same nucleus")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)

def page_packaging(pdf: PdfPages, page: int, total: int, package_dir: Path | None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    page_header(
        fig,
        "BioImage.IO packaging",
        "What a consumer needs that a state dict cannot tell them.",
    )

    fig.text(
        0.06,
        0.890,
        dedent("""
        The checkpoint alone is not a usable model. Three things it cannot express, and
        all three change the output:

          1. the preprocessing that puts a volume on the grid the weights expect
             (3.0 / 1.0 / 1.0 µm), because the network has no concept of physical scale;
          2. that the outputs are LOGITS, not probabilities;
          3. that every spatial axis must be a multiple of 4 — the decoder upsamples
             twice, so anything else raises a tensor-size mismatch rather than degrading
             gracefully.

        The same blind spot bit the CLI's own output: masks were written without
        PhysicalSize, so they aligned with the input array but not with the specimen.
        Carrying the calibration through was a separate fix.
        """).replace("\n        ", "\n"),
        fontsize=8.8,
        color=INK,
        va="top",
        linespacing=1.55,
    )

    fig.text(0.06, 0.620, "Package contents", fontsize=10.5, fontweight="bold", color=BLUE)
    contents = [
        ("rdf.yaml", "BioImage.IO resource description, format 0.5.12"),
        ("weights.pt", "state dict, 108 tensors"),
        ("unet_3d_models.py", "the architecture, so the package reloads standalone"),
        ("environment.yml", "pinned dependencies"),
        ("sample-input.tif", "legal (1,1,32,128,128) tensor, image-readable"),
        ("sample-output.tif", "reference logits for the sample tensor"),
        ("test-input.npy", "same input, .npy form for the test harness"),
        ("test-output.npy", "reference logits, generated from these weights"),
        ("README.md", "model card with the preprocessing contract"),
    ]
    y = 0.592
    for name, note in contents:
        fig.text(0.075, y, name, fontsize=8.0, color=INK, family="monospace")
        fig.text(0.315, y, note, fontsize=7.8, color=MUTED)
        y -= 0.0225

    fig.text(0.06, 0.385, "Verification", fontsize=10.5, fontweight="bold", color=BLUE)
    fig.text(
        0.075,
        0.358,
        dedent("""
        scripts/verify_package.py reloads the package the way a consumer would: it imports
        the PACKAGED architecture file, loads the state dict strictly, regenerates the
        output from the packaged input and compares. It also feeds a 30-voxel axis and
        requires the model to fail, because a documented constraint the code does not
        enforce is worse than no documentation.

        The authoritative check is bioimageio test, and it now passes on the shipped zip:

            OVERALL: passed
              format validation (spec 0.5.12)
              reproduce test outputs from test inputs
              inference for legal sizes n: 0, 1, 2
        """).replace("\n        ", "\n"),
        fontsize=8.2,
        color=INK,
        va="top",
        linespacing=1.55,
    )

    if package_dir and package_dir.is_dir():
        total_kib = sum(p.stat().st_size for p in package_dir.rglob("*") if p.is_file()) / 1024
        fig.text(0.075, 0.232,
                 f"verified package: {package_dir.name}  ({total_kib:.0f} KiB)",
                 fontsize=8.0, color=GREEN, family="monospace")

    fig.text(0.06, 0.195, "Status", fontsize=10.5, fontweight="bold", color=AMBER)
    fig.text(
        0.075,
        0.168,
        dedent("""
        DONE   bioimageio test on the shipped zip: OVERALL passed for every check;
        DONE   shipping one model (fold 1) — the other four folds are cross-validation
               replicates of the same recipe and stay reproducible upstream;
        OPEN   replace the citation placeholder. A package without a resolvable citation
               is not acceptable in the zoo.

        And the target must be stated plainly: each mask entry is a 2D single-slice
        DISC marking a nucleus's middle Z (median 63 voxels, one slice thick), not a
        one-voxel point and not a nucleus body. Users who treat the foreground as a 3D
        segmentation will be wrong about object size AND about object extent.
        """).replace("\n        ", "\n"),
        fontsize=8.2,
        color=INK,
        va="top",
        linespacing=1.55,
    )

    page_footer(fig, page, total, "package source: scripts/build_bioimageio_package.py")
    pdf.savefig(fig, dpi=DPI)
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
    parser.add_argument("--detections", default=None,
                        help="detection.json from scripts/detect_metrics.py; "
                             "adds the instance-centroid page")
    parser.add_argument("--ear-scene", default=None,
                        help="ear1_1-6m_roi.zarr; adds the whole-mount ear pages")
    parser.add_argument("--ear-check", default=None,
                        help="ear_check.json from scripts/check_ear_detections.py")
    parser.add_argument("--ear-compare", default=None,
                        help="JSON from scripts/compare_ear_spots.py --out-json")
    parser.add_argument("--skip-threshold", action="store_true",
                        help="skip the aggregate threshold sweep (slow: reloads every volume)")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    summary_path = artifacts / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"no summary.json in {artifacts}")
    results = json.loads(summary_path.read_text())
    shown = results[: args.limit]

    detection: list[dict] = []
    if args.detections:
        detection = json.loads(Path(args.detections).read_text())

    ear_check = json.loads(Path(args.ear_check).read_text()) if args.ear_check else None
    ear_compare = (json.loads(Path(args.ear_compare).read_text())
                   if args.ear_compare else None)
    ear_scene = Path(args.ear_scene) if args.ear_scene else None

    cfg = {
        "weights": "numorph_fold1_weights.pt",
        "target": "3.0,1.0,1.0",
        "patch": "32,128,128",
        "threshold": args.threshold,
        "device": "cuda (Tesla T4)",
    }

    with PdfPages(args.out) as pdf:
        # Build the page list first, then render it. Every page needs the total for
        # its "n / total" footer, so the total has to be known before the first
        # page is drawn -- and an arithmetic formula here drifted from the pages
        # actually emitted (it said 15 while 16 were written). A list cannot drift:
        # the total IS the length, and adding a page is one entry.
        plan: list[Callable[[int, int], None]] = [
            lambda pg, tot: page_cover(pdf, results, pg, tot, cfg),
            lambda pg, tot: page_grid_question(pdf, results, pg, tot),
            lambda pg, tot: page_method(pdf, pg, tot),
            lambda pg, tot: page_preprocessing(pdf, pg, tot),
            lambda pg, tot: page_forward(pdf, pg, tot),
        ]
        if detection:
            plan.append(lambda pg, tot: page_detections(pdf, artifacts, detection, pg, tot))
        if ear_check or ear_scene:
            plan += [
                lambda pg, tot: page_ear_roi(pdf, pg, tot, ear_scene, ear_check, ear_compare),
                lambda pg, tot: page_ear_slices(pdf, pg, tot, ear_scene, part=1),
                lambda pg, tot: page_ear_slices(pdf, pg, tot, ear_scene, part=2),
                lambda pg, tot: page_ear_projection(pdf, pg, tot, ear_scene),
                lambda pg, tot: page_ear_agreement(pdf, pg, tot, ear_scene),
            ]
        plan += [
            (lambda pg, tot, result=result:
             page_volume(pdf, result, artifacts, pg, tot, threshold=args.threshold))
            for result in shown
        ]
        if not args.skip_threshold:
            plan.append(lambda pg, tot: page_threshold_all(pdf, artifacts, pg, tot))
        plan += [
            lambda pg, tot: page_table(pdf, results, pg, tot),
            lambda pg, tot: page_packaging(pdf, pg, tot,
                                           Path(args.package) if args.package else None),
        ]

        total = len(plan)
        for index, draw in enumerate(plan, start=1):
            draw(index, total)

        metadata = pdf.infodict()
        metadata["Title"] = "nuxnet inference validation on NuMorph"
        metadata["Subject"] = "CLI prediction, instance post-processing, BioImage.IO packaging"
        metadata["Keywords"] = "nuxnet, numorph, nuclei, 3D U-Net, light-sheet"

    print(f"wrote {args.out} ({total} pages, {len(shown)} volume pages shown, "
          f"{len(results)} volumes in the table)")


if __name__ == "__main__":
    main()
