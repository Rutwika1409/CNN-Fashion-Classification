"""
evaluate.py

This script:
1. Loads the trained CNN model.
2. Loads the Fashion-MNIST test dataset.
3. Evaluates model performance.
4. Prints classification metrics.
5. Displays a confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# =====================================================
# Fashion-MNIST Class Names
# =====================================================

CLASS_NAMES = [
    "T-Shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]


# =====================================================
# Load Saved Model
# =====================================================

print("Loading model...")

model = load_model(
    "../models/best_fashion_cnn.keras"
)


# =====================================================
# Load Test Dataset
# =====================================================

print("Loading Fashion-MNIST test dataset...")

(_, _), (X_test, y_test) = fashion_mnist.load_data()


# =====================================================
# Normalize Pixel Values
# =====================================================

X_test = X_test.astype("float32") / 255.0


# =====================================================
# Reshape for CNN
# =====================================================

X_test = X_test.reshape(
    -1,
    28,
    28,
    1
)


# =====================================================
# Evaluate Model
# =====================================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# =====================================================
# Generate Predictions
# =====================================================

predictions = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# =====================================================
# Classification Report
# =====================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES
    )
)


# =====================================================
# Confusion Matrix
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title(
    "Fashion-MNIST Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.tight_layout()

plt.show()