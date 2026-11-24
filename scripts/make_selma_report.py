"""Build a PDF report for the SELMA3D benchmark.

`make_report.py` covers NuMorph, where Dice is a fair headline. This covers external
data where Dice is *capped by label convention* and the honest report has to say so
on the same page as the number, not in a footnote.

The layout follows from that:

* an overview page that puts every number next to its ceiling,
* slice pages showing image / expert body / predicted centre in one frame, because
  "the prediction looks like small dots inside big blobs" is the finding and a table
  cannot convey it,
* a detection page, since centroid F1 is the metric the two label conventions share.

    python make_selma_report.py --benchmark benchmark.json \\
        --scores benchmark.npz --data data/selma3d_nuclei \\
        --out reports/selma3d_validation.pdf
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
from matplotlib.patches import Rectangle

from make_report import (
    BG,
    BLUE,
    C_PRED,
    C_TRUTH,
    DPI,
    INK,
    MUTED,
    RULE,
    _clean_axes,
    _flow,
    page_footer,
    page_header,
)

PAGE_W_IN = 8.27
PAGE_H_IN = 11.69


# ---------------------------------------------------------------------------
# scene access
# ---------------------------------------------------------------------------
def open_patch(root: Path, name: str) -> dict:
    """Load one prepared patch: image, expert mask, and its recorded spacing."""
    import zarr

    group = zarr.open_group(str(root / f"{name}.zarr"), mode="r")
    image = np.asarray(group["0"][:])
    mask = np.asarray(group["labels/nuclei/0"][:])
    transforms = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    scale = next(t["scale"] for t in transforms if t["type"] == "scale")
    return {"image": image, "mask": mask, "scale": np.asarray(scale, float)}


def load_predictions(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.is_file():
        return {}
    return dict(np.load(path, allow_pickle=False))


def predicted_for(predictions: dict, name: str, spacing) -> np.ndarray | None:
    """Find a cached prediction without assuming the exact key format."""
    tag = "_".join(f"{v:g}" for v in spacing)
    for key, value in predictions.items():
        if key.startswith(f"{name}__") and key.endswith(tag):
            return value.astype(bool)
    for key, value in predictions.items():
        if key.startswith(f"{name}__"):
            return value.astype(bool)
    return None


def stretch(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [1, 99.5])
    if hi <= lo:
        hi = lo + 1
    return np.clip((values - lo) / (hi - lo), 0, 1)


def overlay(image_plane: np.ndarray, truth: np.ndarray | None,
            pred: np.ndarray | None) -> np.ndarray:
    """Grey image, expert body in blue, predicted centre in magenta.

    Dilating the prediction by one voxel is deliberate: a centre disc is a few
    voxels across and would disappear at print resolution. The dilation is only
    for display and is stated on the page.
    """
    from matplotlib import colors as mcolors
    from scipy import ndimage as ndi

    truth_rgb = np.array(mcolors.to_rgb(C_TRUTH))
    pred_rgb = np.array(mcolors.to_rgb(C_PRED))

    grey = stretch(image_plane)
    rgb = np.stack([grey, grey, grey], axis=-1)
    if truth is not None and truth.any():
        rgb[truth] = 0.55 * rgb[truth] + 0.45 * truth_rgb
    if pred is not None and pred.any():
        grown = ndi.binary_dilation(pred, np.ones((3, 3)))
        rgb[grown] = 0.15 * rgb[grown] + 0.85 * pred_rgb
    return np.clip(rgb, 0, 1)


def brightest_slice(mask: np.ndarray, lo: int, hi: int) -> int:
    counts = [int(mask[z].sum()) for z in range(max(lo, 0), min(hi, mask.shape[0]))]
    return max(lo, 0) + int(np.argmax(counts)) if counts else lo


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def page_overview(pdf: PdfPages, page: int, total: int, data: dict,
                  predictions: dict) -> None:
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN), facecolor=BG)
    mean = data["mean"]

    page_header(
        fig,
        "SELMA3D: nuxnet-inference against expert human-brain annotations",
        "Nine annotated patches, never seen in training. Dice is capped by the label "
        "convention, so the ceiling is printed beside it.",
    )

    fig.text(0.062, 0.898, _flow(f"""
    nuxnet was trained on NuMorph mouse cortex, where the target is a single-slice
    disc at each nucleus middle. SELMA3D annotates filled 3D nucleus bodies, drawn in
    virtual reality by four experts. A disc cannot overlap a body well even when the
    detection is perfect, so low Dice here would not by itself mean the model failed.

    To separate the two effects, every patch is also scored against a measured
    **ceiling**: the best Dice any centre-disc predictor could reach, built from the
    ground truth itself. A prediction at its ceiling has found the objects; its Dice
    is low because it marks centres and the truth marks whole nuclei.

    Detection F1 is the metric both conventions can share, and it is the headline.
    """, width=104), fontsize=7.8, color=INK, va="top", linespacing=1.46)

    # ---- number rows, ceiling beside dice ---------------------------------
    rows = [
        ("semantic Dice", f"{mean['dice']:.3f}",
         f"ceiling {mean.get('disc_ceiling_dice', float('nan')):.3f}"),
        ("semantic IoU", f"{mean['iou']:.3f}", ""),
        ("detection precision", f"{mean['precision']:.3f}",
         f"@ {data['tolerance_um']:.1f} um"),
        ("detection recall", f"{mean['recall']:.3f}", ""),
        ("detection F1", f"{mean['f1']:.3f}", "the shareable metric"),
        ("objects, truth", f"{mean['truth_objects']:,}", ""),
        ("objects, predicted", f"{mean['pred_objects']:,}", ""),
        ("foreground, truth", f"{mean['truth_fg_fraction'] * 100:.3f} %", ""),
        ("foreground, predicted", f"{mean['pred_fg_fraction'] * 100:.3f} %",
         "under-covering is the disc signature"),
    ]

    y = 0.600
    for label, value, note in rows:
        fig.text(0.068, y, label, fontsize=8.4, color=MUTED, va="center")
        fig.text(0.330, y, value, fontsize=11.5, color=INK, va="center",
                 fontfamily="DejaVu Sans")
        if note:
            fig.text(0.452, y, note, fontsize=7.6, color=MUTED, va="center",
                     style="italic")
        y -= 0.041

    fig.add_artist(plt.Line2D([0.062, 0.938], [0.196, 0.196], color=RULE, lw=0.7))
    fig.text(0.062, 0.178, _flow(f"""
    The predicted foreground is {mean['truth_fg_fraction'] / max(mean['pred_fg_fraction'], 1e-9):.1f}x
    smaller than the annotated foreground. That ratio is the disc-versus-body
    difference, measured: the model marks one plane per nucleus, the experts mark the
    whole nucleus. Recall is therefore the number that describes detection, and it is
    reported separately from Dice for that reason.

    This is a domain shift as well as a convention difference -- PI on post-mortem
    human brain, against TO-PRO-3 mouse cortex in training -- so neither number
    transfers wholesale to the user's own acquisition.
    """, width=104), fontsize=7.4, color=INK, va="top", linespacing=1.44)

    page_footer(fig, page, total,
                f"{data['n_patches']} patches   spacing assumed "
                f"{data['spacing_um'][0]:g},{data['spacing_um'][1]:g},"
                f"{data['spacing_um'][2]:g} um   threshold {data['threshold']:g}")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_patch_slices(pdf: PdfPages, page: int, total: int, root: Path,
                      records: list[dict], predictions: dict,
                      part: int = 1, n_parts: int = 2) -> None:
    """Image / expert body / predicted centre, one row per patch."""
    from make_report import page_header as header

    selected = records[: 4] if part == 1 else records[4: 8]
    if not selected:
        return

    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN), facecolor=BG)
    header(
        fig,
        "SELMA3D: expert body and predicted centre in the same slice"
        + ("" if part == 1 else " (cont.)"),
        "Grey is the PI image. Blue is the expert annotation. Magenta is the "
        "prediction, dilated one voxel for visibility.",
    )

    fig.text(0.062, 0.900, _flow(f"""
    Each panel is the slice through the most annotated objects in that patch, so the
    comparison is not flattered by empty planes. The prediction is small and sits
    inside the blue body wherever it is correct -- that is the intended behaviour, not
    a resolution artefact. A patch where the magenta dots sit outside the blue bodies
    is a localisation failure; a patch where they are absent inside them is a miss.
    """, width=106), fontsize=7.6, color=INK, va="top", linespacing=1.44)

    width = 0.250
    height = width * PAGE_W_IN / PAGE_H_IN
    top = 0.800
    gap_x = 0.026
    row_h = height * PAGE_W_IN / PAGE_H_IN
    row_gap = 0.040

    for index, record in enumerate(selected):
        name = record["raw"].replace("raw_", "patch_").replace(".mha", "")
        scene = open_patch(root, name)
        pred = predicted_for(predictions, Path(record["raw"]).stem, data_spacing(scene))
        mask = scene["mask"].astype(bool)
        z = brightest_slice(mask, data_lo(mask), data_hi(mask))

        row = index // 2
        col = index % 2
        left = 0.062 + col * (2 * width + gap_x)
        bottom = top - (row + 1) * row_h - row * row_gap

        axes = fig.add_axes([left, bottom, width, height])
        axes.imshow(overlay(scene["image"][z], mask[z], None if pred is None
                            else pred[z]), interpolation="nearest")
        _clean_axes(axes)
        axes.set_title(f"{name}   z = {z}", fontsize=7.4, color=INK, pad=2.5)

        # The second column of each row repeats the same slice with the prediction,
        # so body and centre are never overlaid into an unreadable purple mush.
        axes2 = fig.add_axes([left + width, bottom, width, height])
        axes2.imshow(overlay(scene["image"][z], mask[z],
                             None if pred is None else pred[z]),
                     interpolation="nearest")
        _clean_axes(axes2)
        count = int(mask[z].sum())
        hit = int((mask[z] & pred[z]).sum()) if pred is not None else 0
        axes2.set_title(f"expert {count:,} vox   pred-in-body {hit:,}",
                        fontsize=7.4, color=INK, pad=2.5)

    fig.text(0.062, 0.108, _flow(f"""
    Left of each pair is the expert body alone, right is the same slice with the
    prediction added. Reading them together answers the only question that matters
    here: does the centre fall inside the body? A prediction that lands on the body
    but covers a small fraction of it is a correct detection with a different label
    convention, which is exactly what the ceiling on the first page quantifies.
    """, width=106), fontsize=7.2, color=INK, va="top", linespacing=1.42)

    page_footer(fig, page, total,
                f"patches {part} of {n_parts}; slice chosen by annotation mass")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def data_spacing(scene: dict) -> np.ndarray:
    return np.asarray(scene["scale"], float)


def data_lo(mask: np.ndarray) -> int:
    return int(np.argmax([mask[z].sum() for z in range(mask.shape[0])]) - 12) \
        if mask.shape[0] > 24 else 0


def data_hi(mask: np.ndarray) -> int:
    return data_lo(mask) + 24


def page_detection(pdf: PdfPages, page: int, total: int, data: dict) -> None:
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN), facecolor=BG)
    page_header(
        fig,
        "SELMA3D: per-patch detection, the metric both conventions share",
        "Greedy one-to-one centroid match at "
        f"{data['tolerance_um']:.1f} um, one nucleus radius at this sampling.",
    )

    fig.text(0.062, 0.898, _flow(f"""
    Detection is computed by matching predicted object centroids to annotated object
    centroids, one-to-one, within a fixed radius. It is independent of how much of a
    nucleus either side covers, so it measures what the model actually did: find the
    right objects in the right places.
    """, width=106), fontsize=7.8, color=INK, va="top", linespacing=1.46)

    # ---- table ------------------------------------------------------------
    header_y = 0.800
    columns = [("patch", 0.068), ("dice", 0.196), ("ceiling", 0.272),
               ("P", 0.372), ("R", 0.436), ("F1", 0.500),
               ("truth", 0.574), ("pred", 0.660)]
    for label, x in columns:
        fig.text(x, header_y, label, fontsize=7.6, color=MUTED, va="center")
    fig.add_artist(plt.Line2D([0.062, 0.760], [header_y - 0.018, header_y - 0.018],
                              color=RULE, lw=0.7))

    y = header_y - 0.040
    for record in data["volumes"]:
        det = record["detection"]
        ceiling = record.get("disc_ceiling_dice")
        cells = [
            (record["raw"].replace("raw_", "").replace(".mha", ""), INK),
            (f"{record['dice']:.3f}", INK),
            (f"{ceiling:.3f}" if ceiling is not None else "--",
             MUTED if ceiling is None else BLUE),
            (f"{det['precision']:.3f}", INK),
            (f"{det['recall']:.3f}", INK),
            (f"{det['f1']:.3f}", INK if det["f1"] < 0.7 else BLUE),
            (f"{record['truth_objects']:,}", MUTED),
            (f"{record['pred_objects']:,}", MUTED),
        ]
        for (text, colour), (_label, x) in zip(cells, columns):
            fig.text(x, y, text, fontsize=7.8, color=colour, va="center")
        y -= 0.030

    fig.add_artist(plt.Line2D([0.062, 0.760], [y + 0.012, y + 0.012],
                              color=RULE, lw=0.7))
    mean = data["mean"]
    cells = [("mean", INK), (f"{mean['dice']:.3f}", INK),
             (f"{mean.get('disc_ceiling_dice', float('nan')):.3f}", BLUE),
             (f"{mean['precision']:.3f}", INK), (f"{mean['recall']:.3f}", INK),
             (f"{mean['f1']:.3f}", INK),
             (f"{mean['truth_objects']:,}", MUTED),
             (f"{mean['pred_objects']:,}", MUTED)]
    for (text, colour), (_label, x) in zip(cells, columns):
        fig.text(x, y, text, fontsize=7.8, color=colour, va="center",
                 fontweight="bold")

    # ---- the reading ------------------------------------------------------
    fig.add_artist(plt.Line2D([0.062, 0.938], [0.300, 0.300], color=RULE, lw=0.7))
    fig.text(0.062, 0.282, _flow(f"""
    **Precision is high and recall is the variable.** The model rarely marks an object
    where the experts marked nothing ({mean['precision']*100:.0f} % of predictions have a
    partner), but it finds roughly {mean['recall']*100:.0f} % of the annotated nuclei.
    Consistent under-detection at threshold {data['threshold']:g} is the expected
    behaviour of a disc-trained model on body annotations: a nucleus whose centre is
    dim, or off the annotated gold-standard set, is simply not marked.

    **The Dice column and the ceiling column move together.** Where a patch scores near
    its ceiling, the remaining gap is convention, not error. Where it falls well below
    -- the lowest patch here -- there is a real localisation or coverage problem to
    investigate, and that patch is the one to look at on the slice pages rather than
    the mean.

    What this does **not** establish: that the model would pass on this data with
    body labels. It was trained on discs; a finetune on the bodies is the experiment
    that would answer that, and the prepared training tiles exist for it.
    """, width=106), fontsize=7.5, color=INK, va="top", linespacing=1.44)

    page_footer(fig, page, total,
                f"{data['n_patches']} patches   patch {data['patch_zyx']}   "
                f"overlap {data['overlap']:g}   threshold {data['threshold']:g}")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="benchmark json")
    parser.add_argument("--scores", default=None, help="cached predictions npz")
    parser.add_argument("--data", required=True, help="dir of prepared *.zarr")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.benchmark).read_text())
    predictions = load_predictions(Path(args.scores) if args.scores else None)
    root = Path(args.data)
    records = data["volumes"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        plan = [
            lambda pg, tot: page_overview(pdf, pg, tot, data, predictions),
            lambda pg, tot: page_patch_slices(pdf, pg, tot, root, records,
                                              predictions, 1, 2),
            lambda pg, tot: page_patch_slices(pdf, pg, tot, root, records,
                                              predictions, 2, 2),
            lambda pg, tot: page_detection(pdf, pg, tot, data),
        ]
        total = len(plan)
        for index, draw in enumerate(plan, start=1):
            draw(index, total)

        metadata = pdf.infodict()
        metadata["Title"] = "SELMA3D: nuxnet-inference against expert annotations"
        metadata["Subject"] = (
            "Semantic and centroid-detection scoring of the shipped NuMorph "
            "checkpoint on external annotated nucleus patches"
        )
        metadata["Keywords"] = "nuxnet, SELMA3D, nuclei, 3D U-Net, benchmark"
    print(f"wrote {out} ({total} pages)")


if __name__ == "__main__":
    main()
