"""Import the models saved by the original notebooks into run folders.

The notebooks saved:

* L1 - ``livrable1_artifacts/modele_classification_photo_autre.keras`` and
  ``parametres_classification.json``;
* L2 - ``debruiteur_final.keras`` (architecture A2, no metadata file);
* L3 - ``livrable3_artifacts/encodeur.weights.h5``, ``decodeur.weights.h5`` and
  ``config_captioning.json``.

Importing copies them (the originals are never modified) into the layout
expected by the ``from_run`` loaders, so the trained models can be reused
without retraining.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tounum.captioning.text import Vocabulary
from tounum.captioning.train import DECODER_WEIGHTS, ENCODER_WEIGHTS
from tounum.captioning.train import METADATA_FILENAME as CAPTION_METADATA
from tounum.classification.train import METADATA_FILENAME as CLASSIFIER_METADATA
from tounum.classification.train import MODEL_FILENAME as CLASSIFIER_MODEL
from tounum.denoising.train import METADATA_FILENAME as DENOISER_METADATA
from tounum.denoising.train import MODEL_FILENAME as DENOISER_MODEL
from tounum.utils.runs import read_json, write_json


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Expected notebook artifact not found: {path}")
    return path


def import_classifier(legacy_dir: Path, run_dir: Path) -> Path:
    legacy_dir = Path(legacy_dir)
    metadata = read_json(_require(legacy_dir / "parametres_classification.json"))
    shutil.copy2(_require(legacy_dir / "modele_classification_photo_autre.keras"), run_dir / CLASSIFIER_MODEL)
    for key in ("image_size", "decision_threshold"):
        if key not in metadata:
            raise ValueError(f"'{key}' missing from parametres_classification.json")
    metadata["imported_from"] = str(legacy_dir.resolve())
    write_json(run_dir / CLASSIFIER_METADATA, metadata)
    return run_dir


def import_denoiser(legacy_model: Path, run_dir: Path, architecture: str = "A2_unet") -> Path:
    shutil.copy2(_require(Path(legacy_model)), run_dir / DENOISER_MODEL)
    write_json(
        run_dir / DENOISER_METADATA,
        {
            "architecture": architecture,
            "size_multiple": 4,
            "imported_from": str(Path(legacy_model).resolve()),
        },
    )
    return run_dir


def import_captioner(legacy_dir: Path, run_dir: Path) -> Path:
    """Convert ``config_captioning.json`` (French keys) to the package's metadata."""
    legacy_dir = Path(legacy_dir)
    legacy = read_json(_require(legacy_dir / "config_captioning.json"))
    model = legacy["modele"]
    model_cfg = {
        "cnn": model["cnn"],
        "rnn": model["rnn"],
        "initial_state": bool(model["etat_initial"]),
        "units": int(model["units"]),
        "embedding_dim": int(model["embedding_dim"]),
        "dropout": float(model["dropout"]),
        "learning_rate": float(model["lr"]),
        "batch_size": int(model["batch_size"]),
    }
    num_words = int(legacy["vocab_size"]) - 1  # notebook: VOCAB_SIZE = TOP_K + 1
    vocab = Vocabulary.from_index_word(legacy["index_word"], num_words)
    if vocab.start_id != legacy["start_id"] or vocab.end_id != legacy["end_id"]:
        raise ValueError("Inconsistent <start>/<end> ids in config_captioning.json")
    shutil.copy2(_require(legacy_dir / "encodeur.weights.h5"), run_dir / ENCODER_WEIGHTS)
    shutil.copy2(_require(legacy_dir / "decodeur.weights.h5"), run_dir / DECODER_WEIGHTS)
    write_json(
        run_dir / CAPTION_METADATA,
        {
            "model": model_cfg,
            "vocabulary": vocab.to_dict(),
            "vocab_size": int(legacy["vocab_size"]),
            "max_len": int(legacy["max_len"]),
            "decoding": "greedy",
            "imported_from": str(legacy_dir.resolve()),
            "reported_test_bleu": legacy.get("bleu_test"),
        },
    )
    return run_dir
