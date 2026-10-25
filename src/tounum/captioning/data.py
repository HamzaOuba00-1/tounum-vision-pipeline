"""MS COCO 2014 captions: loading and per-image split (notebook *Livrable 3*).

Captions are grouped per image and the split is done **per image**: the five
captions of one photo always stay in the same set, so the model is never
evaluated on a photo it has seen with another caption.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CaptionSplits:
    train: list[int]
    val: list[int]
    test: list[int]
    available: int


def coco_filename(image_id: int) -> str:
    return f"COCO_train2014_{image_id:012d}.jpg"


def load_captions(annotations_file: Path) -> dict[int, list[str]]:
    """``{image_id: [caption, ...]}`` from ``captions_train2014.json``."""
    annotations_file = Path(annotations_file)
    if not annotations_file.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {annotations_file}. See data/README.md.")
    with annotations_file.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if "annotations" not in payload:
        raise ValueError(f"{annotations_file} has no 'annotations' key.")
    captions: dict[int, list[str]] = defaultdict(list)
    for item in payload["annotations"]:
        captions[int(item["image_id"])].append(item["caption"])
    return dict(captions)


def split_image_ids(
    captions: dict[int, list[str]],
    images_dir: Path,
    n_test: int,
    n_val: int,
    n_train: int,
    seed: int,
) -> CaptionSplits:
    """Keep images present on disk, shuffle with a fixed seed, then cut test / val / train."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"COCO image folder not found: {images_dir}")
    present = set(os.listdir(images_dir))
    ids = sorted(i for i in captions if coco_filename(i) in present)
    if len(ids) < n_test + n_val + 1:
        raise ValueError(
            f"Only {len(ids)} captioned images found; need more than n_test + n_val = {n_test + n_val}."
        )
    random.Random(seed).shuffle(ids)
    test = ids[:n_test]
    val = ids[n_test : n_test + n_val]
    train = ids[n_test + n_val : n_test + n_val + n_train]
    if len(train) < n_train:
        raise ValueError(f"Requested {n_train} training images, only {len(train)} available.")
    return CaptionSplits(train=train, val=val, test=test, available=len(ids))
