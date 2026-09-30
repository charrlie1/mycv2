"""Tests for the optional NumPy shape classifier and bundled weights."""

import numpy as np
import pytest

from mycv.shape_cnn import CLASS_NAMES, load_shape_cnn, predict_shape


def test_bundled_shape_weights_load():
    model = load_shape_cnn()
    assert model is not None


def test_prediction_returns_class_and_probabilities():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 8:22] = 255

    result = predict_shape(mask)

    assert result["label"] in CLASS_NAMES
    assert set(result["probs"]) == set(CLASS_NAMES)
    assert sum(result["probs"].values()) == pytest.approx(1.0)


def test_prediction_rejects_empty_mask():
    with pytest.raises(ValueError, match="non-empty object mask"):
        predict_shape(np.zeros((16, 16), dtype=np.uint8))
