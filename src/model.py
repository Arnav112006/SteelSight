"""
model.py
--------
CNN model architectures for steel defect detection.
Provides baseline CNN, improved CNN with BatchNorm,
and a transfer learning model using MobileNetV2.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


IMG_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)


def build_baseline_cnn():
    """3-block CNN — original college project model."""
    model = Sequential([
        Input(shape=INPUT_SHAPE),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ], name="Baseline_CNN")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_improved_cnn():
    """3-block CNN with BatchNormalization and padding for stability."""
    model = Sequential([
        Input(shape=INPUT_SHAPE),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid")
    ], name="Improved_CNN")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_mobilenet():
    """
    Transfer learning with MobileNetV2 pretrained on ImageNet.
    Base layers are frozen; only the classification head is trained.
    """
    base = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False  # freeze pretrained weights

    inputs = Input(shape=INPUT_SHAPE)
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="MobileNetV2_Transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def get_callbacks(patience=3):
    """Standard callbacks: EarlyStopping + ReduceLROnPlateau."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]


MODEL_REGISTRY = {
    "Baseline CNN": build_baseline_cnn,
    "Improved CNN": build_improved_cnn,
    "MobileNetV2 (Transfer)": build_mobilenet,
}
