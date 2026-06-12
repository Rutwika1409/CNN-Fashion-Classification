"""
train.py

This script:
1. Loads the Fashion-MNIST dataset.
2. Preprocesses the data.
3. Builds a CNN model.
4. Trains the model with augmentation.
5. Evaluates performance.
6. Saves both best and final models.
"""

import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =====================================================
# Base Paths
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "fashion-mnist_train.csv"
)

TEST_PATH = os.path.join(
    DATA_DIR,
    "fashion-mnist_test.csv"
)

BEST_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "best_fashion_cnn.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fashion_cnn.keras"
)


# =====================================================
# Load Dataset
# =====================================================

print("Loading dataset...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nTraining Dataset Shape:")
print(train_df.shape)

print("\nTesting Dataset Shape:")
print(test_df.shape)


# =====================================================
# Separate Features and Labels
# =====================================================

X_train = train_df.drop(
    columns=["label"]
).values

y_train = train_df["label"].values

X_test = test_df.drop(
    columns=["label"]
).values

y_test = test_df["label"].values


# =====================================================
# Normalize Pixel Values
# =====================================================

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# =====================================================
# Reshape Images
# =====================================================

X_train = X_train.reshape(
    -1,
    28,
    28,
    1
)

X_test = X_test.reshape(
    -1,
    28,
    28,
    1
)

print("\nTraining Image Shape:")
print(X_train.shape)

print("\nTesting Image Shape:")
print(X_test.shape)

print("\nTraining Labels Shape:")
print(y_train.shape)

print("\nTesting Labels Shape:")
print(y_test.shape)

print("\nUnique Labels:")
print(np.unique(y_train))


# =====================================================
# Data Augmentation
# =====================================================

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15
)


# =====================================================
# Build CNN Model
# =====================================================

print("\nBuilding CNN Model...")

model = keras.Sequential()

model.add(
    keras.layers.Input(
        shape=(28, 28, 1)
    )
)

model.add(
    keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    )
)

model.add(
    keras.layers.BatchNormalization()
)

model.add(
    keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )
)

model.add(
    keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    )
)

model.add(
    keras.layers.BatchNormalization()
)

model.add(
    keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )
)

model.add(
    keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu"
    )
)

model.add(
    keras.layers.BatchNormalization()
)

model.add(
    keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    )
)

model.add(
    keras.layers.Flatten()
)

model.add(
    keras.layers.Dense(
        units=128,
        activation="relu"
    )
)

model.add(
    keras.layers.Dropout(
        rate=0.3
    )
)

model.add(
    keras.layers.Dense(
        units=10,
        activation="softmax"
    )
)

model.summary()


# =====================================================
# Compile Model
# =====================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =====================================================
# Callbacks
# =====================================================

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)


# =====================================================
# Train Model
# =====================================================

print("\nTraining Model...")

history = model.fit(
    datagen.flow(
        X_train,
        y_train,
        batch_size=64
    ),
    validation_data=(
        X_test,
        y_test
    ),
    epochs=20,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)


# =====================================================
# Evaluate Model
# =====================================================

print("\nEvaluating Model...")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")


# =====================================================
# Save Final Model
# =====================================================

print("\nSaving Final Model...")

model.save(
    FINAL_MODEL_PATH
)

print(
    f"\nBest model saved at:\n{BEST_MODEL_PATH}"
)

print(
    f"\nFinal model saved at:\n{FINAL_MODEL_PATH}"
)