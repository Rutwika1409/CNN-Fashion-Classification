"""
test_model.py

Automated tests for the Fashion-MNIST CNN project.

Tests:
1. Model loads successfully.
2. Prediction output shape is correct.
3. Prediction probabilities sum to 1.
"""

import numpy as np

from tensorflow.keras.models import load_model


# =====================================================
# Load Model
# =====================================================

MODEL_PATH = "../models/fashion_cnn.keras"


# =====================================================
# Test Model Loading
# =====================================================

def test_model_load():

    model = load_model(MODEL_PATH)

    assert model is not None


# =====================================================
# Test Prediction Shape
# =====================================================

def test_prediction_shape():

    model = load_model(MODEL_PATH)

    sample_image = np.random.rand(
        1,
        28,
        28,
        1
    )

    prediction = model.predict(
        sample_image,
        verbose=0
    )

    assert prediction.shape == (1, 10)


# =====================================================
# Test Probability Distribution
# =====================================================

def test_probability_sum():

    model = load_model(MODEL_PATH)

    sample_image = np.random.rand(
        1,
        28,
        28,
        1
    )

    prediction = model.predict(
        sample_image,
        verbose=0
    )

    total_probability = np.sum(
        prediction
    )

    assert np.isclose(
        total_probability,
        1.0,
        atol=1e-5
    )