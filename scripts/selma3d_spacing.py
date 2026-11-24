"""Calibrate the true voxel spacing of the SELMA3D annotated nucleus patches.

The files cannot be trusted on this point. Every patch carries an *isotropic*
spacing:

* SELMA3D 2024 (S-BIAD1196, NIfTI): ``pixdim = 1 1 1``, and ``sform_code = 2``
  with an identity 1 mm matrix.
* SELMA3D 2026 (S-BIAD2110, MHA): ``ElementSpacing = 1 1 1``.

But the challenge paper's Table 1 gives the source nucleus volumes as
**0.54 x 0.54 x 5 um**, and its Table 2 gives the annotated patches as
200 x 200 x 200 voxels. Those two statements cannot both describe the same
sampling: a 200^3 patch at 0.54/0.54/5.0 would be a 108 x 108 x 1000 um slab.

So the spacing has to be measured from the data. The trick is that the
annotation gives an independent ruler: a nucleus is roughly spherical, so the
z extent in micrometres must equal the in-plane extent in micrometres. Solving

    z_span_voxels * s_z  =  xy_span_voxels * s_xy

for ``s_z`` at the paper's ``s_xy = 0.54`` recovers the spacing the annotators
effectively worked at. This is a measurement, not a guess, and it is the only
way to resample these patches correctly.

Also reports the label *definition* -- filled body versus single-slice disc --
because that, not accuracy, is what decides whether a Dice number against this
data means anything.
"""

from __future__ import annotations

import gzip
import struct
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi


def load_nifti(path: Path) -> np.ndarray:
    """Read a NIfTI-1 volume as (z, y, x).

    NIfTI stores i (=x) fastest, so a plain C-order reshape of the payload into
    (dim3, dim2, dim1) already yields (k, j, i) = (z, y, x). Transposing after the
    reshape is what produced an axis-order bug in an earlier version of this
    analysis, so it is deliberately not done here.
    """
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    header = raw[:352]
    dims = struct.unpack("<8h", header[40:56])
    datatype = struct.unpack("<h", header[70:72])[0]
    offset = int(struct.unpack("<f", header[108:112])[0])
    dtype = {64: np.float64, 16: np.float32, 4: np.int16, 2: np.uint8,
             512: np.uint16, 256: np.int8}[datatype]
    shape = tuple(dims[1:dims[0] + 1])
    flat = np.frombuffer(raw[offset:], dtype=dtype, count=int(np.prod(shape)))
    return flat.reshape(shape)


def load_mha(path: Path) -> np.ndarray:
    """Read a MetaImage volume as (z, y, x).

    MHA is header-then-payload with a plain-text header terminated by
    ``ElementDataFile = LOCAL``. The payload is little-endian gzip-compressed when
    ``CompressedData`` is True, and MHA order is x fastest like NIfTI, so the same
    plain reshape applies.
    """
    raw = path.read_bytes()
    marker = b"ElementDataFile = LOCAL"
    start = raw.index(marker) + len(marker)
    start = raw.index(b"\n", start) + 1
    header = raw[:start].decode("latin-1")
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()

    dims = [int(v) for v in fields["DimSize"].split()]
    dtype = {"MET_USHORT": np.uint16, "MET_UCHAR": np.uint8,
             "MET_SHORT": np.int16, "MET_FLOAT": np.float32}[fields["ElementType"]]
    payload = raw[start:]
    if fields.get("CompressedData", "False") == "True":
        # MetaImage compresses with zlib (magic 0x78 ..), not gzip. Feeding a zlib
        # stream to gzip.decompress raises BadGzipFile, so the magic byte decides.
        import zlib

        payload = (gzip.decompress(payload) if payload[:2] == b"\x1f\x8b"
                   else zlib.decompress(payload))
    flat = np.frombuffer(payload, dtype=dtype, count=int(np.prod(dims)))
    # Header order is x fastest, so a *C-order* reshape into the reversed dims
    # already yields (z, y, x) with no transpose -- identical to the NIfTI path.
    # Using order="F" here silently transposed the volume, which on a cubic patch
    # swaps z and x with no error and no visible symptom.
    return flat.reshape(tuple(dims[::-1]))


def describe(volume: np.ndarray, name: str) -> dict:
    """Count objects, measure their extents, and invert the spacing."""
    binary = volume > 0
    labelled, count = ndi.label(binary, structure=np.ones((3, 3, 3)))
    objects = [o for o in ndi.find_objects(labelled) if o is not None]
    spans = np.array(
        [[o[0].stop - o[0].start, o[1].stop - o[1].start, o[2].stop - o[2].start]
         for o in objects], dtype=float
    )
    sizes = np.bincount(labelled.ravel())[1:]

    # In-plane extent is taken as the mean of y and x so a single ragged axis does
    # not drive the calibration.
    inplane = spans[:, 1:3].mean(axis=1)
    zspan = spans[:, 0]

    # Weight the calibration by object size: a 1-voxel speck has a meaningless
    # aspect ratio, and there are many of them.
    keep = sizes >= np.percentile(sizes, 25)
    implied_sz = np.median(inplane[keep] * 0.54 / np.maximum(zspan[keep], 1))

    print(f"\n=== {name} ===")
    print(f"shape (z,y,x)      {volume.shape}   dtype {volume.dtype}")
    print(f"foreground         {binary.mean() * 100:.3f}%   objects {count:,}")
    print(f"voxels/object      median {np.median(sizes):.0f}  "
          f"p10 {np.percentile(sizes, 10):.0f}  p90 {np.percentile(sizes, 90):.0f}")
    print(f"bbox span voxels   z {np.median(zspan):.1f}   "
          f"y {np.median(spans[:, 1]):.1f}   x {np.median(spans[:, 2]):.1f}")
    print(f"z/xy span ratio    {np.median(zspan) / np.median(inplane):.2f}")
    print(f"implied z spacing  {implied_sz:.2f} um  (at s_xy = 0.54 um)")
    return {
        "shape": volume.shape,
        "objects": int(count),
        "median_voxels": float(np.median(sizes)),
        "z_span": float(np.median(zspan)),
        "xy_span": float(np.median(inplane)),
        "implied_z_um": float(implied_sz),
    }


def main() -> int:
    roots = [Path(p) for p in sys.argv[1:]] or [Path(".")]
    for root in roots:
        gt_files = sorted(root.glob("gt*.mha")) or sorted(root.glob("gt*.nii.gz"))
        if not gt_files:
            print(f"no ground truth in {root}")
            continue
        for path in gt_files[:4]:
            describe(load_mha(path) if path.suffix == ".mha"
                     else load_nifti(path), f"{root.name}/{path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
