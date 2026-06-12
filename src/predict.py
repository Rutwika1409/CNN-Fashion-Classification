import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from PIL import Image, ImageFilter, ImageOps


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
# Model Path
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_fashion_cnn.keras"
)

print("Loading model from:")
print(MODEL_PATH)

model = load_model(MODEL_PATH)

# =====================================================
# Image Preprocessing
# =====================================================

def preprocess_image(uploaded_image):
    """
    Converts uploaded image into Fashion-MNIST format.

    Parameters
    ----------
    uploaded_image : file-like object

    Returns
    -------
    numpy.ndarray
        Shape: (1, 28, 28, 1)
    """
    image = Image.open(uploaded_image)

    image = image.convert("L")

    image = ImageOps.autocontrast(image)

    image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
)

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=1,
            percent=100,
            threshold=3
        )
    )


    image = np.array(image)

    image = image.astype("float32") / 255.0

    image = image.reshape(
        1,
        28,
        28,
        1
    )

    return image


# =====================================================
# Prediction Function
# =====================================================

def predict_fashion_item(uploaded_image):
    """
    Predict clothing category and confidence.

    Parameters
    ----------
    uploaded_image : Uploaded file

    Returns
    -------
    tuple
        (predicted_class, confidence)
    """

    processed_image = preprocess_image(
        uploaded_image
    )

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction
    )

    confidence = float(
        np.max(prediction)
    ) * 100

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    return (
        predicted_class,
        confidence
    )