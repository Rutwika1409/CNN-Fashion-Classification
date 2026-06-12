"""
app.py

Streamlit application for Fashion-MNIST classification.

Features:
1. Upload an image.
2. Preview the uploaded image.
3. Predict clothing category.
4. Display confidence score.
"""

import streamlit as st
from PIL import Image

from src.predict import (
    predict_fashion_item,
    preprocess_image
)


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Fashion-MNIST Classifier",
    page_icon="👕",
    layout="centered"
)


# =====================================================
# Application Title
# =====================================================
st.title("👕 Fashion-MNIST Classifier")

st.sidebar.title("Project Information")

st.sidebar.write(
    """
    Dataset:
    Fashion-MNIST

    Model:
    Convolutional Neural Network

    Framework:
    TensorFlow/Keras

    Frontend:
    Streamlit
    """
)


# =====================================================
# Classes information section
# =====================================================
st.info(
    """Class labels:
    T-Shirt/Top,
    Trouser,
    Pullover,
    Dress,
    Coat,
    Sandal,
    Shirt,
    Sneaker,
    Bag,
    Ankle Boot"""
)

# =====================================================
# Image Upload Section
# =====================================================

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["png", "jpg", "jpeg"]
)


# =====================================================
# Display Uploaded Image
# =====================================================

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        # Set specific pixel size (e.g., width=400)
        # To maintain responsive sizing instead, use width="stretch" or width="content"
        st.image(
            image,
            caption="Uploaded Image",
            width=400
        )
        processed = preprocess_image(uploaded_file)

        st.image(
            processed.reshape(28, 28),
            caption="Model Input (28x28)",
            width=200
        )
    except Exception:
        # Fallback caption displayed if the image is corrupt or unreadable
        st.error("⚠️ Error: Unable to display the uploaded image file.")

    st.write("---")


    # =============================================
    # Prediction Button
    # =============================================

    if st.button("Predict Category"):

        with st.spinner("Predicting..."):
            uploaded_file.seek(0)

            predicted_class, confidence = (
                predict_fashion_item(
                    uploaded_file
                )
            )

        st.success("Prediction Completed")

        st.subheader("Prediction")

        st.write(
            f"**Category:** {predicted_class}"
        )

        st.write(
            f"**Confidence:** {confidence:.2f}%"
        )


# =====================================================
# Footer
# =====================================================

st.write("---")

st.caption(
    "Built using TensorFlow, Keras and Streamlit"
)