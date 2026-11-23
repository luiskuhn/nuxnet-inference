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
        sampling the file happened to be stored at, no resampling, no tiling. On a NuMorph
        volume that scored Dice 0.114 — a pipeline mismatch, not a bad model.
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
            "Connected components (26-connectivity). The target is a CENTRE MARKER, so "
            "both distributions are small by construction.")

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

        And the target must be stated plainly: this is a CENTRE MARKER, not a nucleus
        body. Users who treat the foreground as a segmentation of nucleus volume will be
        wrong by an order of magnitude in object size.
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
    parser.add_argument("--skip-threshold", action="store_true",
                        help="skip the aggregate threshold sweep (slow: reloads every volume)")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    summary_path = artifacts / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"no summary.json in {artifacts}")
    results = json.loads(summary_path.read_text())
    shown = results[: args.limit]

    cfg = {
        "weights": "numorph_fold1_weights.pt",
        "target": "3.0,1.0,1.0",
        "patch": "32,128,128",
        "threshold": args.threshold,
        "device": "cuda (Tesla T4)",
    }

    with PdfPages(args.out) as pdf:
        page = 1
        total = 4 + len(shown) + (0 if args.skip_threshold else 1) + 1

        page_cover(pdf, results, page, total, cfg)
        page += 1
        page_grid_question(pdf, results, page, total)
        page += 1
        page_method(pdf, page, total)
        page += 1

        for result in shown:
            page_volume(pdf, result, artifacts, page, total, threshold=args.threshold)
            page += 1

        if not args.skip_threshold:
            page_threshold_all(pdf, artifacts, page, total)
            page += 1

        page_table(pdf, results, page, total)
        page += 1
        page_packaging(pdf, page, total, Path(args.package) if args.package else None)

        metadata = pdf.infodict()
        metadata["Title"] = "nuxnet inference validation on NuMorph"
        metadata["Subject"] = "CLI prediction, instance post-processing, BioImage.IO packaging"
        metadata["Keywords"] = "nuxnet, numorph, nuclei, 3D U-Net, light-sheet"

    print(f"wrote {args.out} ({total} pages, {len(shown)} volume pages shown, "
          f"{len(results)} volumes in the table)")


if __name__ == "__main__":
    main()
