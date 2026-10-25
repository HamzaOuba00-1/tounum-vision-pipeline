"""TouNum vision pipeline.

Three deep-learning modules, refactored from the original Jupyter notebooks:

1. ``tounum.classification`` - photo vs. other-document classifier (CNN);
2. ``tounum.denoising`` - convolutional autoencoder that removes noise from photos;
3. ``tounum.captioning`` - image captioning by transfer learning (CNN + attention RNN);

and ``tounum.pipeline``, which chains them on a folder of scanned images.
"""

__version__ = "1.0.0"
