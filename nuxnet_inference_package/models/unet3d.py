"""The 3D U-Net, re-exported from the vendored training copy.

There is exactly **one** architecture definition in this package, and it lives at
``nuxnet_inference_package/training_parity/unet_3d_models.py`` -- a verbatim copy
of the trainer's:

    luiskuhn/nuxnet-training @ f716565
    numorph_nuclei_segmentation/model/unet_3d_models.py

Previously this module held a hand-written lookalike that had silently diverged --
no residual shortcuts, ``bias=True`` convolutions, different module names -- so no
trained checkpoint could be loaded. Keeping a single definition removes the
possibility of that happening again; ``tests/test_unet3d_parity.py`` pins the
parameter names and shapes against a real checkpoint.

Import from here as before:

    from nuxnet_inference_package.models.unet3d import UNet3D
"""

from __future__ import annotations

from nuxnet_inference_package.training_parity.unet_3d_models import (
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
