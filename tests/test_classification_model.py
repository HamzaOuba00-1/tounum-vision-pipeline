import math

import numpy as np
import pandas as pd
import pytest

from tounum.classification.evaluate import (
    compute_metrics,
    plateau_analysis,
    select_threshold,
    source_class_analysis,
)
from tounum.classification.model import build_classifier
from tounum.classification.train import compute_class_weights


def test_metrics_perfect_and_single_class():
    y = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["f1_macro"] == 1.0 and metrics["roc_auc"] == 1.0
    assert math.isnan(compute_metrics(np.array([0, 0]), np.array([0.1, 0.9]))["roc_auc"])


def test_threshold_is_selected_on_given_set():
    y = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.5, 0.6, 0.7, 0.8, 0.9])
    threshold, f1, curve = select_threshold(y, probabilities)
    assert 0.6 < threshold <= 0.7
    assert f1 == 1.0
    assert len(curve) == 81


def test_source_class_analysis_error_types():
    frame = pd.DataFrame({"source_class": ["Photo", "Photo", "Painting", "Text"], "label": [1, 1, 0, 0]})
    result = source_class_analysis(frame, np.array([0.9, 0.2, 0.8, 0.1]), 0.5).set_index("source_class")
    assert result.loc["Photo", "error_type"] == "false_negative"
    assert result.loc["Photo", "binary_error_rate"] == 0.5
    assert result.loc["Painting", "binary_error_rate"] == 1.0
    assert result.loc["Text", "binary_error_rate"] == 0.0


def test_plateau_analysis():
    result = plateau_analysis([0.5, 0.3, 0.2, 0.21, 0.22, 0.23])
    assert result["best_epoch"] == 3
    assert result["plateau_start_epoch"] == 4


def test_class_weights_balance_classes():
    weights = compute_class_weights(np.array([0, 0, 0, 1]))
    assert weights[1] > weights[0]
    assert pytest.approx(weights[0] * 3 + weights[1]) == 4


@pytest.mark.parametrize(
    "variant, expected_convs", [("reference", 3), ("regularized", 3), ("four_blocks", 4)]
)
def test_variants_architecture(variant, expected_convs):
    model = build_classifier(variant, image_size=(32, 32))
    names = [layer.name for layer in model.layers]
    assert sum(name.startswith("conv_") for name in names) == expected_convs
    assert ("augmentation" in names) == (variant != "reference")
    assert model.output_shape == (None, 1)
    output = model.predict(np.zeros((2, 32, 32, 3), dtype="float32"), verbose=0)
    assert ((output >= 0) & (output <= 1)).all()


def test_paper_architecture_parameter_counts():
    """Same parameter counts as the notebook summaries (180x180 inputs)."""
    assert build_classifier("reference").count_params() == 8_023_361
    assert build_classifier("four_blocks").count_params() == 4_353_601


def test_unknown_variant():
    with pytest.raises(ValueError, match="Unknown classifier variant"):
        build_classifier("resnet")
