"""Cassava Leaf Disease Classification - Member 3: Model Architecture

Architecture module based on the team's 03_Model_Architecture_Member3_Final.ipynb.
This file defines preprocessing/data-pipeline helpers and the MobileNetV2 model.
Training loops, fine-tuning experiments, callbacks, and final evaluation belong
 to Member 4.
"""

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -----------------------------------------------------------------------------
# Configuration (keep consistent with the preprocessing notebook)
# -----------------------------------------------------------------------------
BASE_DIR = "/kaggle/input/notebooks/rawanhossamsaid/dl-preprocessing"
IMAGE_DIR = os.path.join(BASE_DIR, "processed")

IMG_SIZE = 256
CHANNELS = 3
NUM_CLASSES = 5
BATCH_SIZE = 16
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds to improve experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split_data(base_dir: str = BASE_DIR):
    """Load the prepared train, validation, and test CSV split files."""
    train_df = pd.read_csv(os.path.join(base_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(base_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(base_dir, "test_split.csv"))
    return train_df, val_df, test_df


def add_processed_image_paths(*dataframes, image_dir: str = IMAGE_DIR):
    """Add a proc_path column pointing to each processed PNG image."""
    for df in dataframes:
        df["proc_path"] = df["image_id"].apply(
            lambda name: os.path.join(
                image_dir, os.path.splitext(name)[0] + ".png"
            )
        )
    return dataframes


def load_image(image_path):
    """Read, resize, and apply the official MobileNetV2 preprocessing."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=CHANNELS)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    # Do not divide by 255 before preprocess_input.
    image = preprocess_input(image)
    return image


def create_dataset(df, training: bool = False, batch_size: int = BATCH_SIZE):
    """Create a batched tf.data pipeline from a dataframe with proc_path/label."""
    image_paths = df["proc_path"].values
    labels = df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
        )

    dataset = dataset.map(
        lambda path, label: (load_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_model(num_classes: int = NUM_CLASSES,
                img_size: int = IMG_SIZE,
                channels: int = CHANNELS) -> keras.Model:
    """Build the frozen ImageNet MobileNetV2 backbone and classifier head."""
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, channels),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, channels))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs,
                       name="Cassava_MobileNetV2_Classifier")


def compile_model(model: keras.Model,
                  learning_rate: float = 1e-5) -> keras.Model:
    """Compile the model with the baseline configuration from the notebook."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    set_seed()
    model = compile_model(build_model())
    model.summary()
    print("MobileNetV2 loaded successfully!")
    print("Model output shape:", model.output_shape)
    print("Trainable backbone layers:", sum(
        layer.trainable for layer in model.layers[1].layers
    ))
