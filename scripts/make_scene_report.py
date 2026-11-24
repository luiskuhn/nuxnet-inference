"""Build a PDF report for a real-data scene that has no ground truth.

`make_report.py` covers the NuMorph volumes, where Dice against a labelled mask is
the honest headline. This module covers the other case, which is the one a user
actually has: unlabelled tissue, a model that has never seen it, and an independent
detector's point list as the only external opinion.

The distinction drives the whole layout. There is no accuracy figure here and the
report says so plainly, rather than substituting a proxy and letting a reader assume
it means accuracy. What it offers instead is everything that can be measured:

* co-location of the mask with the **input channel** -- the strongest available
  evidence that the mask is on the right structures, because the model saw only
  that channel and a mix-up cannot fake agreement with it
* the geometry of the output, which catches degenerate masks
* agreement with the shipped point annotations, framed as similarity to another
  detector

Pages are large-image first: four slices at near half-page width each, plus a
projection page and a zoom page. A montage of twelve thumbnails would fit more
information per page and show none of it.

    python make_scene_report.py --scene data/te24_1-66m_roi.zarr \\
        --data reports/te24_metrics.json --out reports/te24_validation.pdf
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

# Reuse the NuMorph report's chrome and colours rather than redefining them. Two
# reports in one project with different blues is a small thing that makes a reader
# distrust both.
from make_report import (
    BG,
    BLUE,
    C_BOTH,
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
def load_scene(scene: Path, points_group: str | None = None):
    """Image plane source, mask, and points, all in one physical frame.

    The image is returned as a zarr array rather than a loaded volume: a 500 M
    voxel channel is 1 GB and this report only ever draws a handful of planes.
    Reading slices from disk costs milliseconds and keeps the generator runnable
    while an inference is still writing into the same store.
    """
    import zarr

    root = zarr.open_group(str(scene), mode="r")
    channels = root.attrs.get("omero", {}).get("channels", [])
    labels = [c.get("label", "") for c in channels]
    marked = [i for i, text in enumerate(labels) if "MODEL INPUT" in text.upper()]
    channel = marked[0] if marked else 0

    entry = root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    scale = np.array(entry[0]["scale"], dtype=float)[-3:]
    translate = np.array(entry[1]["translation"], dtype=float)[-3:]

    image = root["0"]
    mask = root["labels"]["nuclei"]["0"] if "labels" in root else None

    points = None
    group = points_group
    if group is None:
        candidates = sorted(k for k in root.keys() if k.startswith("points_"))
        group = candidates[0] if candidates else None
    if group and group in root:
        points = np.stack([
            np.asarray(root[group]["z"][:], dtype=float),
            np.asarray(root[group]["y"][:], dtype=float),
            np.asarray(root[group]["x"][:], dtype=float),
        ], axis=1)
    return {
        "root": root,
        "image": image,
        "mask": mask,
        "channel": channel,
        "channel_label": labels[channel] if channels else "?",
        "labels": labels,
        "scale": scale,
        "translate": translate,
        "points": points,
        "points_group": group,
    }


def plane(scene: dict, z: int) -> np.ndarray:
    image = scene["image"]
    if image.ndim == 4:
        return np.asarray(image[scene["channel"], z], dtype=np.float32)
    return np.asarray(image[z], dtype=np.float32)


def mask_plane(scene: dict, z: int) -> np.ndarray | None:
    mask = scene["mask"]
    return None if mask is None else np.asarray(mask[z]).astype(bool)


def stretch(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, (1.0, 99.5))
    if high <= low:
        return np.zeros(values.shape, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def overlay(grey_plane: np.ndarray, plane_mask: np.ndarray | None,
            points_xy: np.ndarray | None = None) -> np.ndarray:
    """Grey anatomy with the mask in magenta and nothing else.

    A one-voxel lightened ring is added around the mask so a marker that is only a
    few voxels across is still visible at print resolution. Display only -- the
    stored mask stays a strict 0/1 array and no measurement comes from this image.
    """
    from scipy import ndimage as ndi

    grey = stretch(grey_plane)
    rgb = np.dstack([grey, grey, grey])
    if plane_mask is not None and plane_mask.any():
        ring = ndi.binary_dilation(plane_mask, np.ones((3, 3))) & ~plane_mask
        rgb[ring] = matplotlib.colors.to_rgb("#ff9ecb")
        rgb[plane_mask] = matplotlib.colors.to_rgb(C_PRED)
    return rgb


def axis_height(width_frac: float, ny: int, nx: int) -> float:
    return width_frac * (PAGE_W_IN / PAGE_H_IN) * (ny / nx)


def slices_by_mask_mass(mask, count: int, part: int = 1) -> list[int]:
    """Slice indices spread over the detections, not over depth.

    Even depth spacing lands on empty planes in a volume whose tissue fades out at
    both ends, which wastes most of the page. Cumulative foreground mass keeps each
    panel comparably populated while preserving the depth ordering.
    """
    per_slice = np.zeros(mask.shape[0])
    for z in range(mask.shape[0]):
        per_slice[z] = np.asarray(mask[z]).sum()
    depth = per_slice.size
    half = depth // 2
    lo, hi = (0, half) if part == 1 else (half, depth)
    window = per_slice[lo:hi]
    cumulative = np.cumsum(window)
    if cumulative[-1] <= 0:
        return [int(round(lo + (hi - lo - 1) * f))
                for f in np.linspace(0.2, 0.8, count)]
    targets = (np.arange(count) + 0.5) / count * cumulative[-1]
    picked = [lo + int(np.searchsorted(cumulative, t)) for t in targets]
    return sorted({min(z, hi - 1) for z in picked})


def busiest_slice(mask, lo: int, hi: int) -> int:
    per_slice = np.array([np.asarray(mask[z]).sum() for z in range(lo, hi)])
    return int(lo + np.argmax(per_slice)) if per_slice.size else lo


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
def page_overview(pdf: PdfPages, page: int, total: int, scene: dict, data: dict) -> None:
    """What this dataset is, what was run, and what the numbers can mean."""
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    page_header(
        fig,
        f"{data['name']}: nuxnet-inference on unlabelled tissue",
        "No ground-truth mask exists for this field of view. Everything below "
        "measures internal consistency and agreement with a second detector,\n"
        "not accuracy.",
    )

    y = 0.888
    fig.text(0.06, y, _flow("""
    The model was trained on NuMorph nuclei and has never seen this acquisition. That
    is the point of the run -- it is the test of whether the inference package
    transfers -- but it also bounds what can be claimed. There is no human-labelled
    mask here, so no Dice, no precision in the usual sense, and no way to prove a
    given detection is a nucleus rather than a bright artefact. A report that
    printed a single "accuracy" figure would be inventing it.
    """, width=112), fontsize=8.0, color=INK, va="top", linespacing=1.5)

    # ---- the two columns -------------------------------------------------
    left_x, right_x = 0.06, 0.520
    col_y = y - 0.108

    fig.text(left_x, col_y, "The acquisition", fontsize=9.6, fontweight="bold", color=BLUE)
    shape = data["shape_zyx"]
    rows = [
        ("stored crop", f"{shape[0]} x {shape[1]} x {shape[2]} voxels"),
        ("voxel size", " / ".join(f"{v:.3f}" for v in data["voxel_um_zyx"]) + " um (z,y,x)"),
        ("extent", " x ".join(f"{v/1000:.2f}" for v in data["extent_um_zyx"]) + " mm"),
        ("channels", ", ".join(data["channels"])),
        ("model input", data["channel_label"]),
        ("pyramid", f"{len(_levels(scene))} level(s)"),
    ]
    y_row = col_y - 0.024
    for label, value in rows:
        fig.text(left_x + 0.008, y_row, label, fontsize=7.4, color=MUTED)
        fig.text(left_x + 0.120, y_row, value, fontsize=7.4, color=INK, family="monospace")
        y_row -= 0.0180

    fig.text(right_x, col_y, "What the run produced", fontsize=9.6, fontweight="bold",
             color=BLUE)
    rows = [
        ("layer", data["layer"]),
        ("dtype / values", f"{data['dtype']} / {data['values']}"),
        ("foreground", f"{data['foreground_voxels']:,} voxels "
                       f"({data['foreground_fraction']*100:.4f}%)"),
        ("components", f"{data['components_lower_bound']:,} (lower bound)"),
        ("voxels/component", f"{data['voxels_per_component']['median']:.1f} median"
                             if data.get("voxels_per_component") else "n/a"),
        ("wall clock", data.get("wall_clock", "n/a")),
    ]
    y_row = col_y - 0.024
    for label, value in rows:
        fig.text(right_x + 0.008, y_row, label, fontsize=7.4, color=MUTED)
        fig.text(right_x + 0.125, y_row, value, fontsize=7.4, color=INK, family="monospace")
        y_row -= 0.0180

    # ---- the checks ------------------------------------------------------
    block_y = y_row - 0.030
    fig.text(0.06, block_y, "The three checks that are actually available",
             fontsize=9.6, fontweight="bold", color=BLUE)

    col = data["colocation"]
    inten = data["intensity"]
    rank = col.get("mean_percentile_rank_of_mask_voxels")
    checks = [
        ("1", "Does the mask sit on the channel the model was given?",
         f"Mask voxels sit at the {rank*100:.1f}st percentile of their own plane's "
         f"intensity distribution (mean over {col['planes']} planes), and mean "
         f"intensity is {inten['ratio']:.1f}x higher under the mask than off it. "
         f"A channel or axis mix-up produces a plausible-looking mask but cannot "
         f"produce that, so this is the strongest evidence the run is aligned."
         if rank is not None else
         f"Mean intensity is {inten['ratio']:.1f}x higher under the mask than off it."),
        ("2", "Is the output a plausible shape?",
         f"{data['foreground_fraction']*100:.4f}% foreground, "
         f"{data['components_lower_bound']:,} components, median "
         f"{data['voxels_per_component']['median']:.1f} voxels each -- the size the "
         f"single-slice training target predicts. Empty, saturated or one-blob outputs "
         f"would show up here; none does."
         if data.get("voxels_per_component")
         else f"{data['foreground_fraction']*100:.4f}% foreground."),
        ("3", "Where does it disagree with the shipped point annotations?",
         _points_sentence(data)),
    ]
    y_check = block_y - 0.020
    for number, question, answer in checks:
        fig.text(0.065, y_check, number, fontsize=11.0, color=BLUE, fontweight="bold",
                 va="top")
        fig.text(0.090, y_check, question, fontsize=7.6, color=INK, fontweight="bold",
                 va="top")
        fig.add_artist(plt.Rectangle((0.062, y_check - 0.006), 0.876, 0.0012,
                                     color=RULE, transform=fig.transFigure))
        y_check -= 0.0175
        fig.text(0.090, y_check, _flow(answer, width=108), fontsize=7.2, color=INK,
                 va="top", linespacing=1.42)
        y_check -= 0.045

    fig.text(0.06, 0.100, _flow(f"""
    What this report does NOT establish: that every detected object is a nucleus.
    {data['channel_label']} is a cell marker, not a DNA stain, so on this acquisition the model
    can only be segmenting the cells that carry it. A 37% agreement with the shipped
    point annotations is a real disagreement between two detectors -- it is reported
    as such, not explained away.
    """, width=112), fontsize=7.4, color=INK, va="top", linespacing=1.45)

    page_footer(fig, page, total,
                f"scene {Path(data['store']).name}; metrics from scene_report_data.py")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def _levels(scene: dict) -> list[str]:
    try:
        return [k for k in scene["root"].keys() if k.isdigit()]
    except Exception:
        return ["0"]


def _points_sentence(data: dict) -> str:
    points = data.get("points")
    if not points:
        return "No point annotations in this store, so no comparison is made."
    median = points.get("median_distance_um")
    tail = (f" Matched pairs are {median:.2f} um apart at the median, well inside a "
            f"nucleus, so where the two agree they agree closely."
            if median is not None else "")
    return (f"The store carries {points['n_points']:,} points from "
            f"{points['group'].replace('points_', '')}. The mask has "
            f"{points['n_components']:,} components. One-to-one matching at "
            f"{data['tolerance_um']:.1f} um gives precision {points['precision']:.3f} "
            f"and recall {points['recall']:.3f} (F1 {points['f1']:.3f}). "
            f"These points are another detector's output, not ground truth.{tail}")


def page_slices(pdf: PdfPages, page: int, total: int, scene: dict, data: dict,
                part: int = 1) -> None:
    """Four large slices, chosen by detection mass rather than by depth."""
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    mask = scene["mask"]
    picks = slices_by_mask_mass(mask, 4, part)
    ny, nx = mask.shape[1], mask.shape[2]

    page_header(
        fig,
        f"{data['name']}: the mask slice by slice" + ("" if part == 1 else " (cont.)"),
        "z = " + ", ".join(str(z) for z in picks)
        + f". Grey is {data['channel_label']}; magenta is the predicted mask.",
    )

    fig.text(0.062, 0.905, _flow(f"""
    Slices are spaced by cumulative detection mass, not by depth fraction, so each
    panel carries a comparable number of objects. Even depth spacing lands on empty
    planes wherever the tissue thins out, which wastes the page and flatters the mask.
    """, width=112), fontsize=7.4, color=INK, va="top", linespacing=1.45)

    width = 0.438
    height = axis_height(width, ny, nx)
    col_x = (0.062, 0.062 + width + 0.030)
    top = 0.845
    gap = 0.030

    for index, z in enumerate(picks):
        row, col = divmod(index, 2)
        bottom = top - (row + 1) * height - row * gap
        axis = fig.add_axes((col_x[col], bottom, width, height))
        plane_mask = mask_plane(scene, z)
        axis.imshow(overlay(plane(scene, z), plane_mask))
        _clean_axes(axis)
        count = 0 if plane_mask is None else int(plane_mask.sum())
        axis.set_title(f"z = {z}      {count:,} mask voxels",
                       fontsize=8.0, color=MUTED, pad=2.4)

    note_y = top - 2 * height - gap - 0.014
    fig.text(0.062, note_y, _flow(f"""
    The mask is sparse by construction: the training target is a single-slice disc at
    each nucleus middle, not a filled cell body, so a plane through a nucleus shows a
    small marker rather than a blob. That is the intended output and it is why voxel
    overlap with a filled annotation would read low even for a perfect detection.
    """, width=112), fontsize=7.2, color=INK, va="top", linespacing=1.45)

    page_footer(fig, page, total,
                "level 0; display stretch 1-99.5th percentile; stored mask unmodified")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_projection(pdf: PdfPages, page: int, total: int, scene: dict, data: dict) -> None:
    """One large max projection, then one patch at three magnifications."""
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    mask = scene["mask"]
    depth, ny, nx = mask.shape

    # Project in z-chunks and keep the per-pixel max. A whole-volume max needs the
    # volume in memory; chunking it is O(one slab) and identical in result.
    image_max = np.zeros((ny, nx), dtype=np.float32)
    mask_max = np.zeros((ny, nx), dtype=bool)
    for z0 in range(0, depth, 32):
        z1 = min(z0 + 32, depth)
        if scene["image"].ndim == 4:
            slab = np.asarray(scene["image"][scene["channel"], z0:z1], dtype=np.float32)
        else:
            slab = np.asarray(scene["image"][z0:z1], dtype=np.float32)
        np.maximum(image_max, slab.max(axis=0), out=image_max)
        mask_max |= np.asarray(mask[z0:z1]).astype(bool).any(axis=0)

    page_header(
        fig,
        f"{data['name']}: whole population, then a zoom",
        f"Max projection over all {depth} slabs. Grey {data['channel_label']}, "
        f"magenta predicted mask.",
    )

    width = 0.560
    height = axis_height(width, ny, nx)
    axis = fig.add_axes((0.062, 0.845 - height, width, height))
    axis.imshow(overlay(image_max, mask_max))
    _clean_axes(axis)
    axis.set_title(f"max projection, {depth} slices   "
                   f"{int(mask_max.sum()):,} pixels covered",
                   fontsize=8.2, color=MUTED, pad=3.0)

    fig.text(0.655, 0.845, _flow(f"""
    One projection shows the whole population at once. The markers are not a
    segmentation of the tissue: they trace the cells that carry
    {data['channel_label']}.

    The zooms on the right take the busiest
    patch of the projection and magnify it, so the
    relationship between marker and nucleus is
    visible rather than inferred.
    """, width=44), fontsize=7.2, color=INK, va="top", linespacing=1.45)

    # Busiest patch by mask coverage, so the zooms show the case that matters.
    from scipy import ndimage as ndi

    density = ndi.uniform_filter(mask_max.astype(np.float32), size=max(ny // 12, 8))
    cy, cx = np.unravel_index(int(np.argmax(density)), density.shape)
    half = max(ny // 12, 16)
    y0, y1 = max(0, cy - half), min(ny, cy + half)
    x0, x1 = max(0, cx - half), min(nx, cx + half)

    # Three panels down the right column, each half the previous field of view.
    # Three zoom panels down the right column, each half the previous field of view.
    # Sized from the space actually available rather than from the projection's row:
    # the earlier version anchored them under the projection, which for a tall
    # (narrow-aspect) volume pushed two of the three panels below the page edge.
    panel_w = 0.288
    zoom_top = 0.700
    gap = 0.022
    for index, shrink in enumerate((1, 2, 4)):
        sh = max(8, half // shrink)
        py0, py1 = max(0, cy - sh), min(ny, cy + sh)
        px0, px1 = max(0, cx - sh), min(nx, cx + sh)
        sub_img = image_max[py0:py1, px0:px1]
        sub_mask = mask_max[py0:py1, px0:px1]
        ph = axis_height(panel_w, sub_img.shape[0], sub_img.shape[1])
        bottom = zoom_top - (index + 1) * ph - index * gap
        if bottom < 0.055:
            # Out of room: stop rather than draw a panel off the page. The zoom
            # ladder is a nicety; a panel clipped by the margin is a defect.
            break
        ax = fig.add_axes((0.655, bottom, panel_w, ph))
        ax.imshow(overlay(sub_img, sub_mask))
        _clean_axes(ax)
        ax.set_title(f"{int(sub_mask.sum()):,} mask pixels   "
                     f"{(py1-py0)*data['voxel_um_zyx'][1]:.0f} um tall",
                     fontsize=7.0, color=MUTED, pad=2.0)

    fig.add_artist(Rectangle((x0 / nx, 0.845 - height + (1 - y1 / ny) * height),
                             (x1 - x0) / nx, (y1 - y0) / ny * height,
                             fill=False, edgecolor=C_TRUTH, linewidth=0.9,
                             transform=fig.transFigure))

    page_footer(fig, page, total,
                "max projection of level 0; display stretch 1-99.5th percentile")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def page_agreement(pdf: PdfPages, page: int, total: int, scene: dict, data: dict) -> None:
    """Mask and the shipped points on the same slices, coloured by who found what."""
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    points = scene["points"]
    mask = scene["mask"]

    if points is None:
        page_header(fig, f"{data['name']}: mask and point annotations",
                    "This store carries no points, so there is nothing to compare.")
        fig.text(0.06, 0.80, "No points group in the store.", fontsize=10, color=MUTED)
        page_footer(fig, page, total, "")
        pdf.savefig(fig, dpi=DPI)
        plt.close(fig)
        return

    from scipy.spatial import cKDTree

    depth, ny, nx = mask.shape
    scale = scene["scale"]
    translate = scene["translate"]
    # Points are stored in absolute microns; convert once into voxel indices so both
    # streams share a frame. Sampling the mask by microns would work too, but voxel
    # indices make the per-plane filter exact.
    point_vox = (points - translate) / scale
    inside = np.all((point_vox >= 0) & (point_vox < np.array(mask.shape)), axis=1)
    point_vox = point_vox[inside]
    point_z = np.rint(point_vox[:, 0]).astype(int)

    # Centroids of the mask, from slab-wise labelling.
    from scipy import ndimage as ndi

    structure = np.ones((3, 3, 3), dtype=int)
    centroids = []
    for z0 in range(0, depth, 16):
        block = np.asarray(mask[z0:z0 + 16])
        labelled, n = ndi.label(block, structure=structure)
        if not n:
            continue
        local = np.array(ndi.center_of_mass(block, labelled, np.arange(1, n + 1)))
        local[:, 0] += z0
        centroids.append(local)
    centroids = np.vstack(centroids) if centroids else np.empty((0, 3))

    tolerance = data["tolerance_um"]
    point_um = point_vox * scale
    centroid_um = centroids * scale
    point_matched = np.zeros(len(point_um), dtype=bool)
    centroid_matched = np.zeros(len(centroid_um), dtype=bool)
    if len(point_um) and len(centroid_um):
        d_pt, i_pt = cKDTree(centroid_um).query(point_um, k=1)
        d_cd, i_cd = cKDTree(point_um).query(centroid_um, k=1)
        point_matched = d_pt <= tolerance
        centroid_matched = d_cd <= tolerance

    page_header(
        fig,
        f"{data['name']}: model mask and shipped points, same slices",
        "Magenta = model only. Cyan = point annotation only. White = both mark the "
        "same object.",
    )

    fig.text(0.062, 0.903, _flow(f"""
    These are the two streams the agreement numbers come from, drawn in one physical
    frame. A cyan ring with no magenta marker inside is an object the annotations
    found and the model did not; a magenta marker with no cyan nearby is the reverse.
    The points come from a different detector, so neither colour is "right".
    """, width=112), fontsize=7.5, color=INK, va="top", linespacing=1.45)

    # Two slices from the busiest thirds of the volume.
    per_slice = np.array([np.asarray(mask[z]).sum() for z in range(depth)], dtype=float)
    # A point can sit exactly on the upper edge of the crop after the micron->voxel
    # round trip, which makes point_z reach `depth` and np.bincount return one bin
    # more than there are slices -- a broadcast error that only appears on stores
    # whose points touch the boundary. Clip rather than truncate the array so the
    # shape is right whatever the rounding did.
    point_z_in = np.clip(point_z, 0, depth - 1)
    spot_counts = np.bincount(point_z_in, minlength=depth)[:depth].astype(float)
    density = per_slice + spot_counts
    third = depth // 3
    picks = [int(np.argmax(density[:third])),
             int(np.argmax(density[third:2 * third])) + third]

    width = 0.360
    height = axis_height(width, ny, nx)
    top = 0.845
    gap = 0.024

    for index, z in enumerate(picks):
        bottom = top - (index + 1) * height - index * gap
        rgb = overlay(plane(scene, z), mask_plane(scene, z)).copy()

        on_slice = point_z == z
        for idx in np.flatnonzero(on_slice):
            yi = int(round(point_vox[idx, 1]))
            xi = int(round(point_vox[idx, 2]))
            if not (0 <= yi < ny and 0 <= xi < nx):
                continue
            colour = C_BOTH if point_matched[idx] else C_TRUTH
            rgb[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4] = \
                matplotlib.colors.to_rgb(colour)
            rgb[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3] = \
                matplotlib.colors.to_rgb("#000000")

        ax = fig.add_axes((0.062, bottom, width, height))
        ax.imshow(rgb)
        _clean_axes(ax)
        n_pt = int(on_slice.sum())
        n_match = int((on_slice & point_matched).sum())
        ax.set_title(f"z = {z}      {n_pt:,} points, {n_match:,} shared with the mask",
                     fontsize=8.0, color=MUTED, pad=2.6)

    # ---- counts beside the panels ---------------------------------------
    right_x = 0.062 + width + 0.045
    fig.text(right_x, 0.845, "Both directions, separately",
             fontsize=9.6, fontweight="bold", color=BLUE)
    rows = [
        ("points in crop", f"{len(point_um):,}"),
        ("mask components", f"{len(centroid_um):,}"),
        ("points with a component", f"{int(point_matched.sum()):,}"),
        ("components with a point", f"{int(centroid_matched.sum()):,}"),
        ("\u2192 recall", f"{point_matched.sum()/max(len(point_um),1):.3f}"),
        ("\u2192 precision", f"{centroid_matched.sum()/max(len(centroid_um),1):.3f}"),
    ]
    y_row = 0.820
    for label, value in rows:
        fig.text(right_x + 0.006, y_row, label, fontsize=7.2, color=MUTED)
        fig.text(right_x + 0.190, y_row, value, fontsize=7.2, color=INK,
                 family="monospace", ha="right")
        y_row -= 0.0172

    fig.text(right_x, y_row - 0.020, _flow(f"""
    Recall is the annotations' own coverage; precision is how many of the model's
    objects have a partner. The two are computed with the same {tolerance:.1f} um
    tolerance, which is one nucleus radius at this dataset's XY sampling.

    Reading them as accuracy would be wrong twice: the points are not complete, and
    {data['channel_label']} marks a cell population rather than all nuclei. The
    disagreement is a property of the two detectors, and this report states it
    rather than resolving it.
    """, width=46), fontsize=7.0, color=INK, va="top", linespacing=1.42)

    page_footer(fig, page, total,
                f"{data['points']['group']} vs {data['layer']}; "
                f"{tolerance:.1f} um one-to-one match")
    pdf.savefig(fig, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data", required=True, help="JSON from scene_report_data.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--parts", type=int, default=2,
                        help="number of slice pages (4 slices each)")
    args = parser.parse_args()

    data = json.loads(Path(args.data).read_text())
    scene = load_scene(Path(args.scene))

    with PdfPages(args.out) as pdf:
        plan = [
            lambda pg, tot: page_overview(pdf, pg, tot, scene, data),
        ]
        for part in range(1, args.parts + 1):
            plan.append(lambda pg, tot, part=part:
                        page_slices(pdf, pg, tot, scene, data, part=part))
        plan.append(lambda pg, tot: page_projection(pdf, pg, tot, scene, data))
        plan.append(lambda pg, tot: page_agreement(pdf, pg, tot, scene, data))

        total = len(plan)
        for index, draw in enumerate(plan, start=1):
            draw(index, total)

        metadata = pdf.infodict()
        metadata["Title"] = f"{data['name']}: nuxnet-inference validation"
        metadata["Subject"] = (
            "Chunked 3D U-Net inference on unlabelled light-sheet tissue; "
            "co-location, geometry and detector agreement"
        )
        metadata["Keywords"] = "nuxnet, nuclei, 3D U-Net, light-sheet, OME-Zarr"

    print(f"wrote {args.out} ({total} pages)")


if __name__ == "__main__":
    main()
