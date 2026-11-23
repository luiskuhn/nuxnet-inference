# Example data

A real image / ground-truth mask pair, small enough to live in the repository, so the
CLI can be run and **scored** without first finding the training dataset.

```
c075_0007_Bottom3.image.ome.tif   float32   64 x 224 x 224   4.7 MB
c075_0007_Bottom3.mask.ome.tif    uint8     64 x 224 x 224    31 KB
sample.json                       measured facts about the pair
```

Both carry `PhysicalSizeZ/Y/X = 2.5 / 0.75 / 0.75 µm`, which is what makes them a
useful test: the volume is *not* on the 3.0 / 1.0 / 1.0 µm grid the network was trained
on, so a run exercises resampling, tiling, projection back, and the calibration
handling. A fixture already on the target grid would skip most of the code.

## Run it

```bash
python -m nuxnet_inference_package.cli_pred predict \
    --input  examples/numorph_c075_0007/c075_0007_Bottom3.image.ome.tif \
    --model  /path/to/numorph_fold1_weights.pt \
    --output runs/c075 \
    --classes 2 --dropout-rate 0.1 \
    --target-voxel-spacing 3.0,1.0,1.0 \
    --patch-size 32,128,128 \
    --threshold 0.8 \
    --postprocess-instances
```

## Score it

```bash
python scripts/score_example.py \
    --prediction runs/c075.ome.tiff \
    --truth      examples/numorph_c075_0007/c075_0007_Bottom3.mask.ome.tif
```

Expected output with `numorph_fold1_weights.pt`:

```
training-grid Dice  0.8601
round-trip ceiling  0.7877   <- a PERFECT mask after the same round trip
input-grid Dice     0.7390   <- what was written to disk
fraction of ceiling 93.8%
```

The ceiling is not decoration. 64 slices compress into 53 on the training grid and
back, which alone caps a *perfect* mask at 0.788 here. A prediction at 0.739 is 94% of
what the round trip allows; it is not "74% correct". If your number lands well below
the ceiling, the model is losing something. If it lands near it, the remaining gap is
the export.

## What the mask actually is

NuMorph masks are **2D single-slice nucleus markers for detection**, sliced at the
middle Z of each nucleus and stored in a 3D volume. They are not nucleus bodies and not
3D instance segmentations.

Measured on this pair (`sample.json`):

| property | value |
|---|---|
| foreground | 1.32% of voxels |
| components | 684 (single-slice), 680 under 26-connectivity |
| component size | median 63 voxels, range 1–134 |
| component Z-thickness | 1 slice (100%) |
| apparent disc diameter | ~14 px ≈ 14 µm in XY, 3 µm in Z |

So the target is a flat disc about one voxel thick. Two consequences:

1. **Component size is meaningful, unlike a 1-voxel centre.** A prediction that
   produces components of 2 voxels is under-segmenting badly, and the size
   distribution is a real diagnostic.
2. **Do not treat this as a 3D volume segmentation.** If you need filled nuclei, this
   model gives you detections to seed a watershed, not boundaries.

Note that the 26-connectivity count (680) is lower than the true disc count (684)
because four pairs of discs in adjacent slices touch diagonally and merge under
26-connectivity. The `component_z_thickness.max = 2` in `sample.json` is those four
pairs, not a genuinely thick object.

## Provenance and licence -- read this

The pair is a genuine sample from the NuMorph semantic-segmentation dataset that
`nuxnet-training` uses (`NUMORPH_SEM_SEG_DATASET.zip`, linked from that repository).

**The dataset's reuse terms are recorded as PENDING in the training repository.** This
pair is included for testing with attribution on that basis. It is *not* a statement
that the data is freely redistributable. If the terms turn out to forbid redistribution,
these two files must be removed and replaced with synthetic data; the mask is a flat
disc target, so a synthetic equivalent is straightforward to generate.

The only transformation applied was `zlib` compression, which is lossless. Voxel
values, dtype, shape, OME axes and physical calibration are byte-identical to the
source. `scripts/build_example_data.py` regenerates both files from the original
dataset and verifies the mask's properties.

Source files:

```
NUMORPH_SEM_SEG_C075_0007_IMAGE_c0203_Training_Bottom3_00x03.ome.tif
NUMORPH_SEM_SEG_C075_0007_MASK_c0203_Training_Bottom3_00x03.ome.tif
```

| file | sha256 |
|---|---|
| `c075_0007_Bottom3.image.ome.tif` | `66e5c726268028861622fdaf4eb550b579f305c50a390cca881cee1327292d33` |
| `c075_0007_Bottom3.mask.ome.tif` | `5ebff6dbd7e4c254c73ee3ad6dd99484a6626aa5f1e4a0153c3efa58c94c4577` |
