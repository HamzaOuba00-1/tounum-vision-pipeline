import pandas as pd

from tounum.classification.predict import PhotoClassifier
from tounum.classification.train import METADATA_FILENAME, MODEL_FILENAME, train_classifier
from tounum.utils.runs import create_run_dir, read_json


def test_end_to_end_training_writes_artifacts(classification_dir, tiny_config):
    run_dir = create_run_dir(tiny_config.path("artifacts"), "classification", "test")
    results = train_classifier(tiny_config, run_dir)

    assert results["selected_model"] in {"cnn_reference", "cnn_4_blocks"}
    assert 0.1 <= results["decision_threshold"] <= 0.9
    for split in ("validation", "test"):
        assert set(results[split]) >= {"f1_macro", "roc_auc", "balanced_accuracy"}
    for name in [
        MODEL_FILENAME,
        METADATA_FILENAME,
        "metrics.json",
        "validation_comparison.csv",
        "threshold_curve.csv",
        "test_confusion_matrix.csv",
        "test_by_source_class.csv",
    ]:
        assert (run_dir / name).is_file(), name

    # splits written to disk are disjoint (no leakage) and cover the dataset
    splits = {s: pd.read_csv(run_dir / "splits" / f"{s}.csv") for s in ("train", "val", "test")}
    paths = [set(s["path"]) for s in splits.values()]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    assert read_json(run_dir / METADATA_FILENAME)["decision_threshold"] == results["decision_threshold"]

    # the saved model is reusable without retraining
    classifier = PhotoClassifier.from_run(run_dir)
    images = sorted((classification_dir / "Photo").glob("*.jpg"))[:3]
    predictions = classifier.classify(images)
    assert len(predictions) == 3
    assert all(0.0 <= p.probability_photo <= 1.0 for p in predictions)
    assert classifier.classify([]) == []
