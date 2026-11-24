# Docker

`Dockerfile.inference` builds on `nuxnet-training:local` rather than starting from a
bare base, so the torch build is the one that produced the checkpoints. The only addition
is scikit-image, which the nuclei instance post-processing needs for `measure.label` and
the training image does not carry.

```bash
docker build -t nuxnet-inference:local -f docker/Dockerfile.inference .
```

Run inference with the dataset and weights mounted:

```bash
docker run --rm --gpus all \
  -v "$PWD":/work -w /work \
  -v /path/to/data:/data \
  -v /path/to/weights.pt:/models/weights.pt:ro \
  nuxnet-inference:local \
  python -m nuxnet_inference_package.cli_pred predict \
    --input /data/volume.ome.tif \
    --model /models/weights.pt \
    --output /work/out/volume \
    --classes 2 --dropout-rate 0.1 \
    --target-voxel-spacing 3.0,1.0,1.0 \
    --patch-size 32,128,128 \
    --postprocess-instances
```
