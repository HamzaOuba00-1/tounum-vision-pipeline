"""Scanned images -> photo? -> denoised photo -> caption.

For every image of an input folder:

1. the classifier decides whether it is a photograph;
2. photographs are denoised by the autoencoder (saved as PNG);
3. the denoised photo is captioned.

Non-photographs are reported but not processed further. Results are written
to ``results.csv`` and ``results.json`` in a new output folder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tounum.utils.logging_config import get_logger
from tounum.utils.runs import write_csv, write_json

LOGGER = get_logger("pipeline.inference")

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ImageResult:
    path: str
    status: str  # "photo" | "other" | "error"
    probability_photo: float | None = None
    denoised_path: str | None = None
    caption: str | None = None
    error: str | None = None


def list_input_images(input_dir: Path, extensions: Sequence[str] = DEFAULT_EXTENSIONS) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    allowed = {e.lower() for e in extensions}
    images = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed)
    if not images:
        raise FileNotFoundError(f"No image with extension {sorted(allowed)} in {input_dir}")
    return images


def prepare_output_dir(output_dir: Path, overwrite: bool = False) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output folder is not empty: {output_dir} (use --overwrite).")
    (output_dir / "denoised").mkdir(parents=True, exist_ok=True)
    return output_dir


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    classifier,
    denoiser,
    captioner,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Process a folder; the three models are passed in already loaded."""
    images = list_input_images(input_dir)
    output_dir = prepare_output_dir(output_dir, overwrite)
    LOGGER.info("%d image(s) to process", len(images))

    results: list[ImageResult] = []
    for path in images:
        try:
            probability = float(classifier.predict_probabilities([path])[0])
        except (OSError, ValueError) as error:
            LOGGER.warning("Cannot read %s: %s", path, error)
            results.append(ImageResult(str(path), "error", error=str(error)))
            continue
        if probability < classifier.threshold:
            results.append(ImageResult(str(path), "other", probability))
            continue

        with Image.open(path) as image:
            original = np.asarray(image.convert("RGB"))
        denoised = denoiser.denoise(original)
        relative = path.relative_to(input_dir).with_suffix(".png")
        denoised_path = output_dir / "denoised" / relative
        denoised_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(denoised).save(denoised_path)
        caption = captioner.caption_arrays([denoised])[0]
        results.append(ImageResult(str(path), "photo", probability, str(denoised_path), caption))
        LOGGER.info("%s -> %s", path.name, caption)

    frame = pd.DataFrame([asdict(r) for r in results])
    write_csv(output_dir / "results.csv", frame)
    write_json(output_dir / "results.json", [asdict(r) for r in results])
    counts = frame["status"].value_counts().to_dict()
    LOGGER.info("Done: %s | results in %s", counts, output_dir)
    return frame
