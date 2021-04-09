from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense
from tensorflow import Tensor
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential


def load_img_data(path, img_height, img_width, batch_size):
    data_dir = pathlib.Path(path)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    print("ENDING LOADING DATASET")

    # %%
    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    return train_ds, val_ds, len(class_names)
