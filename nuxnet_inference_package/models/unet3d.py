"""Compatibility re-export of the 3D U-Net.

The canonical definition now lives at the path that mirrors ``nuxnet-training``:

    nuxnet-training       numorph_nuclei_segmentation/model/unet_3d_models.py
    nuxnet-inference      nuxnet_inference_package/model/unet_3d_models.py

That file is a verbatim copy of the trainer's. This module exists only so existing
imports keep working:

    from nuxnet_inference_package.models.unet3d import UNet3D

New code should import from ``nuxnet_inference_package.model`` instead.

Before this change the module held a hand-written lookalike that had silently
diverged -- no residual shortcuts, ``bias=True`` convolutions, different module
names -- so no trained checkpoint could be loaded.
"""

from __future__ import annotations

from nuxnet_inference_package.model.unet_3d_models import (
    ConvBlock,
    DownSamplingBlock,
    InputBlock,
    OutputBlock,
    UNet3D,
    UpSamplingBlock,
)

__all__ = [
    "UNet3D",
    "ConvBlock",
    "InputBlock",
    "DownSamplingBlock",
    "UpSamplingBlock",
    "OutputBlock",
]
