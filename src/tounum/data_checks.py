"""Fast checks of the input datasets, without loading any model."""

from __future__ import annotations

from tounum.config import Config


def check_datasets(config: Config) -> list[tuple[str, bool, str]]:
    """Return ``(dataset, ok, message)`` for each dataset expected by the pipeline."""
    checks: list[tuple[str, bool, str]] = []

    root = config.path("classification_data")
    cfg = config.classification
    if root.is_dir():
        missing = [c for c in cfg.source_classes if not (root / c).is_dir()]
        counts = {
            c: sum(1 for p in (root / c).rglob("*") if p.suffix.lower() in cfg.image_extensions)
            for c in cfg.source_classes
            if c not in missing
        }
        ok = not missing and all(counts.values())
        message = f"missing folders: {missing}" if missing else f"images per class: {counts}"
        checks.append(("classification", ok, message))
    else:
        checks.append(("classification", False, f"not found: {root}"))

    for name, key in [
        ("denoising (train)", "denoising_train_data"),
        ("denoising (eval)", "denoising_eval_data"),
    ]:
        folder = config.path(key)
        if folder.is_dir():
            count = sum(1 for p in folder.rglob("*") if p.suffix.lower() in config.denoising.image_extensions)
            checks.append((name, count > 0, f"{count} image(s) in {folder}"))
        else:
            checks.append((name, False, f"not found: {folder}"))

    annotations, images = config.path("coco_annotations"), config.path("coco_images")
    ok = annotations.is_file() and images.is_dir()
    checks.append(
        ("captioning (COCO)", ok, f"annotations: {annotations.is_file()} | images: {images.is_dir()}")
    )
    return checks
