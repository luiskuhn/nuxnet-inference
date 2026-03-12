=====
Usage
=====

Run prediction with a custom OME-TIFF 3D input volume:

.. code-block:: console

    nuxnet-pred predict --arch unet3d --input ./input_volume.ome.tiff --output ./predictions/demo

Run prediction and export nuclei instances as TSV (ID, centroid in Z/Y/X, size in voxels):

.. code-block:: console

    nuxnet-pred predict --arch unet3d --input ./input_volume.ome.tiff --output ./predictions/demo --postprocess-instances --nuclei-label 1 --neighbor-radius 2.0

Run smoke test with random input:

.. code-block:: console

    nuxnet-pred smoke-test --arch unet3d --output ./predictions/demo

Disable normalization if needed:

.. code-block:: console

    nuxnet-pred predict --arch unet3d --input ./input_volume.ome.tiff --output ./predictions/demo --no-normalize-input
