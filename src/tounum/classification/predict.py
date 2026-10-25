"""Reusable photo/other predictor loaded from a training run."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from tounum.classification.data import preprocess_for_model
from tounum.classification.train import METADATA_FILENAME, MODEL_FILENAME
from tounum.utils.runs import read_json


@dataclass
class ClassificationResult:
    path: str
    probability_photo: float
    is_photo: bool
    threshold: float


class PhotoClassifier:
    """Loads the model once and classifies images with the training preprocessing."""

    def __init__(self, model, image_size: tuple[int, int], threshold: float) -> None:
        self.model = model
        self.image_size = tuple(image_size)
        self.threshold = float(threshold)

    @classmethod
    def from_run(cls, run_dir: str | Path) -> PhotoClassifier:
        import keras

        run_dir = Path(run_dir)
        metadata = read_json(run_dir / METADATA_FILENAME)
        model_path = run_dir / MODEL_FILENAME
        if not model_path.is_file():
            raise FileNotFoundError(f"Classifier model not found: {model_path}")
        model = keras.models.load_model(model_path)
        return cls(model, tuple(metadata["image_size"]), metadata["decision_threshold"])

    def predict_probabilities(
        self, images: Iterable[str | Path | Image.Image], batch_size: int = 32
    ) -> np.ndarray:
        batch = np.stack([preprocess_for_model(image, self.image_size) for image in images])
        return self.model.predict(batch, batch_size=batch_size, verbose=0).reshape(-1)

    def classify(self, paths: Iterable[str | Path]) -> list[ClassificationResult]:
        paths = [str(p) for p in paths]
        if not paths:
            return []
        probabilities = self.predict_probabilities(paths)
        return [
            ClassificationResult(p, float(prob), bool(prob >= self.threshold), self.threshold)
            for p, prob in zip(paths, probabilities, strict=True)
        ]
