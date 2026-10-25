# Data

No dataset is stored in this repository (see `.gitignore`). Place the data
as below, or point `configs/config.yaml → paths` to another location.
Run `tounum check-data` to verify the layout before training.

```text
data/
├── raw/
│   ├── classification/          # module 1 (and training photos of module 2)
│   │   ├── Painting/  *.jpg|png
│   │   ├── Photo/
│   │   ├── Schematics/
│   │   ├── Sketch/
│   │   └── Text/
│   ├── denoising/               # module 2: validation + test photos
│   │   └── *.jpg|png
│   └── coco/                    # module 3: MS COCO 2014 (train split)
│       ├── annotations/captions_train2014.json
│       └── train2014/COCO_train2014_*.jpg
├── interim/                     # generated: resized-image caches, CNN features
└── processed/                   # reserved for derived datasets
```

## Sources

| Dataset | Origin | Size in the original project | Redistribution |
|---|---|---|---|
| Classification (5 classes) | TouNum project dataset (CESI case study) | 41 399 readable images (Painting 10 000, Photo 9 993, Schematics 10 000, Sketch 1 406, Text 10 000) | **Not public** - not included |
| Denoising evaluation photos | TouNum project dataset (Livrable 2) | 148 photos (74 validation / 74 test) | **Not public** - not included |
| MS COCO 2014 captions | <https://cocodataset.org> (`train2014` images + `captions_train2014.json`) | 82 783 captioned images | Annotations CC BY 4.0; images keep the licences of their Flickr authors - download from the official site |

The sizes above come from the outputs of the original notebooks.

## Generated folders

* `interim/classification_cache/180x180/` - every classification image resized once (JPEG, quality 95).
* `interim/denoising_cache/` - the 128x128 training photos as a `.npy` array.
* `interim/caption_features/<CNN>/` - one `float16` `.npy` file of shape `(64, C)` per COCO image.

These caches are safe to delete; they are rebuilt on the next run.
