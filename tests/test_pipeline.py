import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tounum.cli import main
from tounum.pipeline import legacy
from tounum.pipeline.inference import run_pipeline
from tounum.utils.runs import create_run_dir, latest_run, read_json, write_json


class StubClassifier:
    threshold = 0.5

    def predict_probabilities(self, paths):
        name = Path(paths[0]).name
        if "broken" in name:
            raise OSError("cannot identify image file")
        return np.array([0.9 if name.startswith("photo") else 0.1])


class StubDenoiser:
    def denoise(self, image):
        return image


class StubCaptioner:
    def caption_arrays(self, images):
        return [f"an image of size {images[0].shape[0]}x{images[0].shape[1]}"]


@pytest.fixture
def scans(tmp_path):
    folder = tmp_path / "scans"
    (folder / "sub").mkdir(parents=True)
    Image.fromarray(np.zeros((10, 12, 3), np.uint8)).save(folder / "photo_1.jpg")
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(folder / "sub" / "photo_2.png")
    Image.fromarray(np.full((8, 8, 3), 255, np.uint8)).save(folder / "text_1.png")
    (folder / "broken.jpg").write_bytes(b"x")
    return folder


def test_pipeline_routes_photos_and_reports_errors(scans, tmp_path):
    out = tmp_path / "out"
    frame = run_pipeline(scans, out, StubClassifier(), StubDenoiser(), StubCaptioner())
    assert frame["status"].value_counts().to_dict() == {"photo": 2, "other": 1, "error": 1}
    photos = frame[frame["status"] == "photo"]
    assert photos["caption"].str.startswith("an image").all()
    assert (out / "denoised" / "sub" / "photo_2.png").is_file()
    assert len(pd.read_csv(out / "results.csv")) == 4
    assert len(json.loads((out / "results.json").read_text())) == 4
    with pytest.raises(FileExistsError):
        run_pipeline(scans, out, StubClassifier(), StubDenoiser(), StubCaptioner())


def test_pipeline_requires_images(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "empty", tmp_path / "o", StubClassifier(), StubDenoiser(), StubCaptioner())


def test_run_dirs_are_never_overwritten(tmp_path):
    run = create_run_dir(tmp_path, "task", "r1")
    write_json(run / "metrics.json", {"a": np.float32(1.5), "b": [np.int64(2)]})
    assert read_json(run / "metrics.json") == {"a": 1.5, "b": [2]}
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "task", "r1")
    create_run_dir(tmp_path, "task", "r2")
    assert latest_run(tmp_path, "task").name == "r2"
    with pytest.raises(FileNotFoundError):
        latest_run(tmp_path, "other")


def test_import_legacy_captioner_roundtrip(tmp_path):
    """Weights saved under the notebook's file names load into the package's classes."""
    from tounum.captioning.model import build_caption_model
    from tounum.captioning.predict import Captioner
    from tounum.config import CaptionModelConfig

    legacy_dir = tmp_path / "livrable3_artifacts"
    legacy_dir.mkdir()
    cfg = CaptionModelConfig(cnn="InceptionV3", rnn="gru", initial_state=False, units=8, embedding_dim=4)
    encoder, decoder = build_caption_model(cfg, vocab_size=6, feature_channels=2048)
    encoder.save_weights(legacy_dir / "encodeur.weights.h5")
    decoder.save_weights(legacy_dir / "decodeur.weights.h5")
    (legacy_dir / "config_captioning.json").write_text(
        json.dumps(
            {
                "modele": {
                    "cnn": "InceptionV3",
                    "rnn": "gru",
                    "etat_initial": False,
                    "units": 8,
                    "embedding_dim": 4,
                    "dropout": 0.0,
                    "lr": 0.001,
                    "batch_size": 64,
                },
                "vocab_size": 6,
                "max_len": 5,
                "start_id": 3,
                "end_id": 4,
                "index_word": {"0": "<pad>", "1": "<unk>", "2": "a", "3": "<start>", "4": "<end>"},
            }
        )
    )
    run = legacy.import_captioner(legacy_dir, create_run_dir(tmp_path, "captioning", "legacy"))
    captioner = Captioner.from_run(run, extractor=object())
    features = np.random.rand(1, 64, 2048).astype("float32")
    np.testing.assert_allclose(captioner.encoder(features), encoder(features), rtol=1e-6)


def test_import_legacy_classifier_and_denoiser(tmp_path):
    from tounum.classification.model import build_classifier
    from tounum.classification.predict import PhotoClassifier
    from tounum.denoising.models import build_denoiser
    from tounum.denoising.predict import Denoiser

    l1 = tmp_path / "livrable1_artifacts"
    l1.mkdir()
    build_classifier("reference", image_size=(16, 16)).save(l1 / "modele_classification_photo_autre.keras")
    (l1 / "parametres_classification.json").write_text(
        json.dumps({"model_name": "CNN 4 blocs", "image_size": [16, 16], "decision_threshold": 0.78})
    )
    run = legacy.import_classifier(l1, create_run_dir(tmp_path, "classification", "legacy"))
    assert PhotoClassifier.from_run(run).threshold == 0.78

    build_denoiser("A2_unet").save(tmp_path / "debruiteur_final.keras")
    run = legacy.import_denoiser(
        tmp_path / "debruiteur_final.keras", create_run_dir(tmp_path, "denoising", "l")
    )
    assert Denoiser.from_run(run).denoise(np.zeros((9, 9, 3), np.uint8)).shape == (9, 9, 3)


def test_cli_check_data_reports_missing_datasets(tmp_path, capsys):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("paths:\n  root: .\n")
    assert main(["check-data", "--config", str(config_file)]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_cli_errors_are_reported_cleanly(tmp_path):
    assert main(["check-data", "--config", str(tmp_path / "nope.yaml")]) == 2
    config_file = tmp_path / "config.yaml"
    config_file.write_text("paths:\n  root: .\n")
    assert main(["classify", "predict", "x.jpg", "--config", str(config_file)]) == 2  # no run yet
