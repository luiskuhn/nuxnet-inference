"""Prepare SELMA3D annotated nucleus patches as a nuxnet-inference dataset.

Turns the raw downloads into something both uses need:

* **benchmarking** -- an OME-Zarr per patch carrying image + expert mask, with the
  spacing recorded explicitly so a scoring run cannot silently assume the wrong
  one;
* **finetuning** -- optional export at the trainer's grid (3.0/1.0/1.0 um) split
  into the trainer's patch size, plus a manifest describing every array.

Three properties matter more than speed here, and each is enforced rather than
assumed:

1. **Integrity.** Several raw files arrive truncated from the EBI mirror and fail
   only at decompression, halfway through a benchmark. Every payload is checked
   against its declared size before anything is written, and the run aborts with
   names rather than producing a half-valid dataset.

2. **Spacing is recorded, not inferred later.** Every file header declares
   isotropic 1 um; the challenge paper gives 0.54 x 0.54 x 5.0 um for the source
   volumes. Both are stored in the manifest alongside the spacing implied by
   annotation geometry, and the value actually used is written on the array. A
   downstream reader then has no excuse to guess.

3. **Labels stay as the experts drew them.** SELMA3D nuclei are filled 3D bodies;
   nuxnet's training target is a single-slice disc at each nucleus middle. This
   converter does **not** reshape the labels to match the old convention by
   default -- that would be inventing annotations. ``--labels disc`` can
   synthesise the disc convention for comparability, and is labelled as derived
   wherever it is written.

Layout written
--------------
    <out>/<patch>.zarr/
        0                     (z, y, x) float32/uint16 image, multiscale
        labels/nuclei/0       (z, y, x) uint8 marker, OME-NGFF label
    <out>/manifest.json       every patch, its geometry, split and provenance
    <out>/patches/            optional --to-training-grid output (npz per patch)

    python prepare_selma3d.py --data selma3d --out selma3d_prepared
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# reading, with integrity checks
# ---------------------------------------------------------------------------
def _sha256(path: Path, limit: int = 1 << 20) -> str:
    """Hash of the first megabyte -- enough to identify a file, cheap on 12 MB."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()[:16]


def read_mha_header(raw: bytes) -> tuple[dict[str, str], int]:
    marker = b"ElementDataFile = LOCAL"
    start = raw.index(marker) + len(marker)
    start = raw.index(b"\n", start) + 1
    fields: dict[str, str] = {}
    for line in raw[:start].decode("latin-1").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields, start


def load_mha(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    raw = path.read_bytes()
    fields, start = read_mha_header(raw)
    payload = raw[start:]

    declared = fields.get("CompressedDataSize")
    if declared is not None and len(payload) < int(declared):
        raise ValueError(
            f"{path.name}: payload is {len(payload):,} bytes but the header declares "
            f"{int(declared):,}. The download is truncated; re-fetch it."
        )

    dims = [int(v) for v in fields["DimSize"].split()]
    dtype = {"MET_USHORT": np.uint16, "MET_UCHAR": np.uint8,
             "MET_SHORT": np.int16, "MET_FLOAT": np.float32}[fields["ElementType"]]
    if fields.get("CompressedData", "False") == "True":
        payload = (gzip.decompress(payload) if payload[:2] == b"\x1f\x8b"
                   else zlib.decompress(payload))

    flat = np.frombuffer(payload, dtype=dtype, count=int(np.prod(dims)))
    # x is stored fastest, so a C-order reshape into the reversed dims yields
    # (z, y, x) with no transpose. order="F" here would silently transpose a
    # cubic patch -- z and x swapped, no error, no symptom.
    return flat.reshape(tuple(dims[::-1])), fields


def load_nifti(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    dims = struct.unpack("<8h", raw[40:56])
    datatype = struct.unpack("<h", raw[70:72])[0]
    offset = int(struct.unpack("<f", raw[108:112])[0])
    pixdim = struct.unpack("<8f", raw[76:108])
    dtype = {64: np.float64, 16: np.float32, 4: np.int16, 2: np.uint8,
             512: np.uint16, 256: np.int8}[datatype]
    shape = tuple(dims[1:dims[0] + 1])
    flat = np.frombuffer(raw[offset:], dtype=dtype, count=int(np.prod(shape)))
    fields = {"pixdim": [str(round(v, 6)) for v in pixdim[:dims[0] + 1]]}
    return flat.reshape(shape), fields


def load_any(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    return load_mha(path) if path.suffix == ".mha" else load_nifti(path)


def discover(data: Path) -> list[tuple[Path, Path]]:
    raws = sorted(data.glob("raw_*.mha"))
    gts = sorted(data.glob("gt_*.mha"))
    if not raws:
        raws = sorted(data.glob("*_0000.nii.gz"))
        gts = sorted(p for p in data.glob("*.nii.gz")
                     if not p.name.endswith("_0000.nii.gz"))
    if not raws or len(raws) != len(gts):
        raise SystemExit(
            f"{data}: {len(raws)} raw vs {len(gts)} ground-truth files. Pairing is "
            "by sorted order, so an unequal count cannot be resolved safely."
        )
    return list(zip(raws, gts))


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def object_geometry(mask: np.ndarray) -> dict:
    """Object counts and extents, plus the z spacing implied by their shape.

    A nucleus is roughly a spheroid, so its z extent in micrometres should match
    its in-plane extent. Solving ``z_span * s_z = xy_span * s_xy`` recovers the
    sampling the annotators effectively worked at. This is the only independent
    check on a header that every file in this dataset gets wrong (or at least
    disagrees with the paper about).
    """
    from scipy import ndimage as ndi

    binary = mask > 0
    labelled, count = ndi.label(binary, structure=np.ones((3, 3, 3)))
    if not count:
        return {"objects": 0}

    objects = [o for o in ndi.find_objects(labelled) if o is not None]
    spans = np.array([[o[0].stop - o[0].start, o[1].stop - o[1].start,
                       o[2].stop - o[2].start] for o in objects], dtype=float)
    sizes = np.bincount(labelled.ravel())[1:]

    # Weight by size: one-voxel specks have a meaningless aspect ratio and there
    # are many of them.
    keep = sizes >= np.percentile(sizes, 50)
    xy = spans[keep, 1:3].mean(axis=1)
    z = np.maximum(spans[keep, 0], 1)

    return {
        "objects": int(count),
        "foreground_fraction": float(binary.mean()),
        "voxels_per_object_median": float(np.median(sizes)),
        "z_span_median": float(np.median(spans[:, 0])),
        "xy_span_median": float(np.median(spans[:, 1:3].mean(axis=1))),
        "implied_z_um_at_xy_1um": float(np.median(xy / z)),
    }


# ---------------------------------------------------------------------------
# writing OME-Zarr
# ---------------------------------------------------------------------------
def write_zarr(path: Path, image: np.ndarray, mask: np.ndarray | None,
               spacing: np.ndarray, translation: np.ndarray,
               attrs: dict, label_name: str = "nuclei") -> None:
    """Write image (and optional mask) as NGFF 0.4, level 0 only.

    Level 0 only is deliberate: napari's 3D view always downsamples to the
    coarsest available level, so shipping a pyramid would make 3D silently show a
    blurry volume while 2D showed the real thing. One level keeps both views
    native.
    """
    import zarr

    root = zarr.open_group(str(path), mode="w")
    axes = [{"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}]

    image_store = root.create_dataset(
        "0", data=image, chunks=(min(64, image.shape[0]), 256, 256)
        if image.shape[0] >= 64 else image.shape,
        overwrite=True,
    )
    image_store.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "name": "image",
        "axes": axes,
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [float(v) for v in spacing]},
                {"type": "translation", "translation": [float(v) for v in translation]},
            ],
        }],
    }]
    root.attrs["omero"] = {"channels": [{
        "label": attrs.get("channel_label", "nuclei stain"),
        "color": "FFFFFF",
        "window": {"start": float(np.percentile(image, 1)),
                   "end": float(np.percentile(image, 99.5)),
                   "min": float(image.min()), "max": float(image.max())},
    }]}
    for key, value in attrs.items():
        root.attrs[key] = value

    if mask is None:
        return

    labels = root.require_group("labels")
    labels.attrs["labels"] = [label_name]
    layer = labels.create_group(label_name)
    mask_store = layer.create_dataset(
        "0", data=mask.astype(np.uint8),
        chunks=(min(64, mask.shape[0]), 256, 256)
        if mask.shape[0] >= 64 else mask.shape,
        overwrite=True,
    )
    mask_store.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
    layer.attrs["multiscales"] = [{
        "version": "0.4",
        "name": label_name,
        "axes": axes,
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [float(v) for v in spacing]},
                {"type": "translation", "translation": [float(v) for v in translation]},
            ],
        }],
        "type": "label",
    }]
    layer.attrs["omero"] = {"channels": [{
        "label": label_name, "color": "FF00FF",
        "window": {"start": 0, "end": 1, "min": 0, "max": 1},
    }]}


# ---------------------------------------------------------------------------
# optional: trainer-grid export
# ---------------------------------------------------------------------------
def to_training_grid(image: np.ndarray, mask: np.ndarray, source_spacing: np.ndarray,
                     target: tuple[float, float, float]):
    """Resample image+mask to the training grid with the trainer's own function.

    The image/label pair matters: ``resample_volume_pair`` anti-aliases the image
    and nearest-neighbours the label, so calling it once for both is not a
    shortcut -- it is the only way to keep the mask binary and aligned.
    """
    from nuxnet_inference_package.data_loading.data_loader import resample_volume_pair

    images, labels = resample_volume_pair(
        image[None].astype(np.float32), mask.astype(np.int64),
        np.asarray(source_spacing, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
    )
    return images[0].astype(np.float32), (labels > 0)


def tile_volume(image: np.ndarray, mask: np.ndarray, patch: tuple[int, int, int]):
    """Cut a volume into non-overlapping training patches, dropping empty ones.

    Non-overlapping because these are training examples, not inference windows: an
    overlapping export would put the same nucleus in many patches and inflate the
    apparent dataset size.
    """
    pz, py, px = patch
    for z0 in range(0, image.shape[0] - pz + 1, pz):
        for y0 in range(0, image.shape[1] - py + 1, py):
            for x0 in range(0, image.shape[2] - px + 1, px):
                block_mask = mask[z0:z0 + pz, y0:y0 + py, x0:x0 + px]
                if block_mask.sum() == 0:
                    continue
                yield (z0, y0, x0), (image[z0:z0 + pz, y0:y0 + py, x0:x0 + px],
                                     block_mask)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dir of raw_*/gt_* pairs")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--spacing", default="2.3,1.0,1.0",
                        help="spacing to RECORD as z,y,x um. Default 2.3 is the "
                             "median implied by annotation geometry; use 1,1,1 to "
                             "match the headers.")
    parser.add_argument("--split", default="0.67,0.33,0.0",
                        help="train,val,test fractions")
    parser.add_argument("--to-training-grid", action="store_true",
                        help="also export patches at the trainer's grid")
    parser.add_argument("--target-spacing", default="3.0,1.0,1.0")
    parser.add_argument("--patch", default="32,128,128")
    parser.add_argument("--labels", choices=("body", "disc"), default="body",
                        help="keep the expert bodies, or synthesise the "
                             "single-slice disc convention nuxnet was trained on")
    parser.add_argument("--disc-radius", type=float, default=2.0,
                        help="in-plane radius in voxels for --labels disc")
    parser.add_argument("--no-zarr", action="store_true",
                        help="manifest only; skip the per-patch OME-Zarr")
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spacing = np.array([float(v) for v in args.spacing.split(",")])
    target = tuple(float(v) for v in args.target_spacing.split(","))
    patch = tuple(int(v) for v in args.patch.split(","))

    pairs = discover(data)

    # ---- pass 1: validate everything before writing anything --------------
    # A half-written dataset is worse than none: it looks usable. Every file is
    # decoded once up front so a truncated download fails here, with its name.
    print(f"validating {len(pairs)} pairs ...")
    loaded: list[tuple] = []
    problems: list[str] = []
    for raw_path, gt_path in pairs:
        try:
            image, image_fields = load_any(raw_path)
            truth, gt_fields = load_any(gt_path)
        except Exception as error:                       # noqa: BLE001
            problems.append(f"{raw_path.name}: {error}")
            continue
        if image.shape != truth.shape:
            problems.append(
                f"{raw_path.name}: image {image.shape} vs mask {truth.shape}")
            continue
        values = np.unique(truth)
        if not set(values.tolist()) <= {0, 1}:
            problems.append(f"{gt_path.name}: mask is not binary, {values[:6]}")
            continue
        if truth.sum() == 0:
            problems.append(f"{gt_path.name}: mask is empty")
            continue
        loaded.append((raw_path, gt_path, image, truth, image_fields, gt_fields))
        print(f"  ok  {raw_path.name:28s} {image.shape}  "
              f"fg {truth.mean()*100:.3f}%")

    if problems:
        print("\nFAILED validation -- nothing written:")
        for line in problems:
            print(f"  {line}")
        return 1
    if not loaded:
        print("no usable pairs")
        return 1

    # ---- split -----------------------------------------------------------
    # NOTE: these patches come from only four source volumes (hippocampus, motor
    # cortex, sensory cortex, visual cortex) and the deposit does not say which
    # patch came from which. A split here is therefore *not* guaranteed
    # subject-independent, and a validation score can be optimistic. Recorded
    # rather than glossed over.
    fractions = [float(v) for v in args.split.split(",")]
    total = sum(fractions)
    n = len(loaded)
    n_train = int(round(n * fractions[0] / total))
    n_val = int(round(n * fractions[1] / total))
    split_names = (["train"] * n_train + ["val"] * n_val
                   + ["test"] * (n - n_train - n_val))
    if len(split_names) < n:
        split_names += ["test"] * (n - len(split_names))

    # ---- pass 2: write ---------------------------------------------------
    manifest: dict = {
        "dataset": "SELMA3D annotated nucleus patches",
        "licence": "CC BY-NC 4.0 (S-BIAD1196 / S-BIAD2110) -- non-commercial",
        "citation": [
            "Y. Chen et al., SELMA3D challenge, arXiv:2501.03880",
            "S. Zhao et al., Cell 180(4):796-812, 2020 (nucleus source data)",
            "D. Kaltenecker et al., Nat Methods 21:1306-1315, 2024",
            "H.S. Bhatia et al., Cell 185(26):5040-5058, 2022",
            "M.I. Todorov et al., Nat Methods 17:442-449, 2020",
        ],
        "spacing_recorded_um_zyx": [float(v) for v in spacing],
        "spacing_note": (
            "Every file header declares isotropic 1,1,1. The challenge paper gives "
            "0.54,0.54,5.0 for the SOURCE volumes, which cannot describe these 200^3 "
            "patches. This value is what the arrays are labelled with; it was "
            "chosen, not read."
        ),
        "labels": args.labels,
        "labels_note": (
            "body = filled 3D nuclei as drawn by four experts in VR. "
            "disc = synthesised single-slice disc at each centroid, matching "
            "nuxnet's own training convention. 'disc' is DERIVED, not expert."
        ),
        "split": {"fractions": fractions,
                  "warning": "split is by patch, not by specimen; the deposit does "
                             "not identify each patch's source brain, so "
                             "subject-independence is NOT guaranteed."},
        "patches": [],
    }

    for index, (raw_path, gt_path, image, truth, image_fields, gt_fields) in enumerate(loaded):
        name = raw_path.stem.replace("raw_", "patch_").replace("_0000", "")
        geometry = object_geometry(truth)

        label_mask = truth
        if args.labels == "disc":
            # Synthesise nuxnet's own convention: a single-voxel plane at each
            # nucleus centroid, radius in-plane. Marked DERIVED in the manifest,
            # because it is no longer what the experts drew.
            label_mask = synthesise_discs(truth, radius=args.disc_radius)

        record = {
            "name": name,
            "raw_file": raw_path.name,
            "gt_file": gt_path.name,
            "sha256_head_raw": _sha256(raw_path),
            "sha256_head_gt": _sha256(gt_path),
            "shape_zyx": list(image.shape),
            "dtype": str(image.dtype),
            "value_range": [float(image.min()), float(image.max())],
            "split": split_names[index],
            "geometry": geometry,
            "spacing_declared_in_file": image_fields.get(
                "ElementSpacing", image_fields.get("pixdim", "unknown")),
            "spacing_recorded_um_zyx": [float(v) for v in spacing],
            "zarr": None,
        }

        if not args.no_zarr:
            zarr_path = out / f"{name}.zarr"
            write_zarr(
                zarr_path, image, label_mask, spacing=spacing,
                translation=np.zeros(3),
                attrs={
                    "source": {
                        "dataset": "SELMA3D",
                        "raw_file": raw_path.name,
                        "gt_file": gt_path.name,
                        "licence": "CC BY-NC 4.0",
                    },
                    "channel_label": "nuclei stain",
                },
            )
            record["zarr"] = zarr_path.name

        # ---- ground-truth centroids, the detection reference ----------------
        # Derived from the expert bodies, not chosen: a detection benchmark needs
        # one point per annotated object, and deriving it here (instead of in the
        # scorer) means every scorer uses the same reference. Written in both
        # frames because a voxel coordinate is only meaningful with its spacing.
        centres = _object_centres(label_mask)
        centre_um = centres * spacing if len(centres) else np.empty((0, 3))
        record["gt_centroids"] = {
            "count": int(len(centres)),
            "derived_from": ("expert body mask" if args.labels == "body"
                             else "DERIVED disc mask"),
            "file": f"{name}_gt_centroids.tsv",
        }
        centroid_path = out / f"{name}_gt_centroids.tsv"
        with centroid_path.open("w") as handle:
            handle.write("index\tz_vox\ty_vox\tx_vox\tz_um\ty_um\tx_um\n")
            for i, ((cz, cy, cx), (mz, my, mx)) in enumerate(
                    zip(centres, centre_um)):
                handle.write(f"{i}\t{cz:.3f}\t{cy:.3f}\t{cx:.3f}\t"
                             f"{mz:.4f}\t{my:.4f}\t{mx:.4f}\n")

        manifest["patches"].append(record)
        print(f"  wrote {name}  split={record['split']}  "
              f"objects={geometry['objects']}  "
              f"implied s_z={geometry.get('implied_z_um_at_xy_1um', 0):.2f} um  "
              f"centroids={len(centres)}")

    # ---- optional training-grid export -----------------------------------
    if args.to_training_grid:
        patch_dir = out / "patches"
        patch_dir.mkdir(exist_ok=True)
        written = 0
        for index, (raw_path, gt_path, image, truth, _f, _g) in enumerate(loaded):
            name = raw_path.stem.replace("raw_", "patch_").replace("_0000", "")
            resampled, resampled_mask = to_training_grid(image, truth, spacing, target)
            for (z0, y0, x0), (block, block_mask) in tile_volume(
                    resampled, resampled_mask, patch):
                tag = f"{name}_z{z0}_y{y0}_x{x0}"
                np.savez_compressed(
                    patch_dir / f"{tag}.npz",
                    image=block.astype(np.float32),
                    mask=block_mask.astype(np.uint8),
                    spacing=np.asarray(target),
                    source=np.asarray(name),
                )
                written += 1
        manifest["training_grid"] = {
            "target_spacing_um_zyx": list(target),
            "patch_zyx": list(patch),
            "n_patches": written,
            "note": "non-overlapping tiles; empty tiles dropped",
        }
        print(f"\nwrote {written} training-grid patches to {patch_dir}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out / 'manifest.json'}")

    counts = {s: split_names.count(s) for s in ("train", "val", "test")}
    print(f"split: {counts}")
    return 0


def _object_centres(mask: np.ndarray) -> np.ndarray:
    """Connected-component centroids, z-y-x voxel order."""
    from scipy import ndimage as ndi

    labelled, count = ndi.label(mask > 0, structure=np.ones((3, 3, 3)))
    if not count:
        return np.empty((0, 3))
    return np.array(ndi.center_of_mass(mask > 0, labelled, np.arange(1, count + 1)))


def synthesise_discs(mask: np.ndarray, radius: float) -> np.ndarray:
    """Re-draw the labels as nuxnet's convention: a disc at each nucleus middle.

    This exists to make the *label convention* comparable, not to improve the
    data. nuxnet was trained on NuMorph targets that mark one plane through each
    nucleus centre; scoring those weights against filled bodies mixes two effects
    (localisation error and convention difference) and there is no way to tell
    them apart from one Dice number.

    The output is explicitly DERIVED. Anyone using it must say so, because the
    ground truth here is the experts' bodies, not this.
    """
    centres = _object_centres(mask)
    out = np.zeros(mask.shape, dtype=bool)
    if len(centres) == 0:
        return out
    zz, yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
    for cz, cy, cx in centres:
        plane = np.abs(zz - int(round(cz))) < 0.5
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        out |= (plane & disc)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
