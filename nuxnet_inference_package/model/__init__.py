"""NuxNet model architecture, mirrored from the training package.

The layout deliberately mirrors ``nuxnet-training``:

    nuxnet-training            numorph_nuclei_segmentation/model/
    nuxnet-inference           nuxnet_inference_package/model/

``unet_3d_models.py`` is a verbatim copy of the trainer's file, so the two stay
diffable line for line. This package also keeps ``models/`` (plural), which holds
the older inference-only helpers; new code should import from here.
"""

from nuxnet_inference_package.model.unet_3d_models import UNet3D

__all__ = ["UNet3D"]
