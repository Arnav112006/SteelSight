"""
dataset.py
----------
Handles dataset downloading, loading, and preprocessing
for the Severstal Steel Defect Detection dataset.
"""

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def download_dataset():
    """Download the Severstal dataset via kagglehub."""
    import kagglehub
    print("Downloading Severstal Steel Defect Detection dataset...")
    path = kagglehub.competition_download("severstal-steel-defect-detection")
    print(f"Dataset downloaded to: {path}")
    return path


def load_dataframe(base_path: str):
    """
    Load train.csv and build a unified image-level DataFrame.

    Returns
    -------
    final_df : pd.DataFrame
        Columns: ImageId, Label (0=No Defect, 1=Defect)
    raw_df : pd.DataFrame
        Original train.csv with EncodedPixels for mask visualization
    paths : dict
        {'train_img': path, 'csv': path}
    """
    train_img_path = os.path.join(base_path, "train_images")
    csv_path = os.path.join(base_path, "train.csv")

    raw_df = pd.read_csv(csv_path)

    all_imgs = set(os.listdir(train_img_path))
    defect_imgs = set(raw_df["ImageId"].unique())

    data = []
    for img in all_imgs:
        label = 1 if img in defect_imgs else 0
        data.append([img, label])

    final_df = pd.DataFrame(data, columns=["ImageId", "Label"])

    paths = {"train_img": train_img_path, "csv": csv_path}
    return final_df, raw_df, paths


def get_class_counts(final_df: pd.DataFrame):
    """Return (no_defect_count, defect_count)."""
    counts = final_df["Label"].value_counts()
    return counts.get(0, 0), counts.get(1, 0)


def make_generators(final_df: pd.DataFrame, train_img_path: str,
                    img_size=(128, 128), batch_size=32, test_size=0.2):
    """
    Split data and return (train_data, val_data, val_df, val_gen).
    val_gen is kept separately so we can create shuffle=False generators for eval.
    """
    train_df, val_df = train_test_split(
        final_df,
        test_size=test_size,
        stratify=final_df["Label"],
        random_state=42
    )

    train_df["Label"] = train_df["Label"].astype(str)
    val_df["Label"] = val_df["Label"].astype(str)

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_img_path,
        x_col="ImageId",
        y_col="Label",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    val_data = val_gen.flow_from_dataframe(
        dataframe=val_df,
        directory=train_img_path,
        x_col="ImageId",
        y_col="Label",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    return train_data, val_data, val_df, val_gen


def make_eval_generator(val_df: pd.DataFrame, train_img_path: str,
                        img_size=(128, 128), batch_size=32):
    """Shuffle=False generator for correct label-prediction alignment."""
    val_gen = ImageDataGenerator(rescale=1.0 / 255)
    return val_gen.flow_from_dataframe(
        dataframe=val_df,
        directory=train_img_path,
        x_col="ImageId",
        y_col="Label",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )


def rle_decode(mask_rle: str, shape=(256, 1600)):
    """Decode run-length encoded string into a binary mask array."""
    s = mask_rle.split()
    starts = np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T


def load_image(img_path: str, target_size=None):
    """Load an image with OpenCV, convert to RGB."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size:
        img = cv2.resize(img, target_size)
    return img
