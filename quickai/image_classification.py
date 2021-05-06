"""
from quickai import ImageClassification
"""

import pathlib
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy
import matplotlib.pyplot as plt
import numpy as np


class ImageClassification:
    """
    Method use is default
    """

    def __init__(
            self,
            model,
            path,
            save,
            batch_size=8,
            data_augmentation=False,
            epochs=20,
            graph=True):
        self.model = model.lower()
        self.save = save
        self.path = path
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation
        self.epochs = epochs
        self.graph = graph
        self.use()

    @staticmethod
    def load_img_data(
            path,
            img_height,
            img_width,
            batch_size,
            grayscale=False):
        """
            :param grayscale: Grayscale or not
            :param path is path to data
            :param img_width
            :param img_height are dims of image
            :param batch_size is batch size
        """
        data_dir = pathlib.Path(path)
        color_mode = "grayscale" if grayscale else "rgb"
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode=color_mode
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode=color_mode
        )
        class_names = train_ds.class_names
        print(class_names)
        autotune = tf.data.experimental.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
        val_ds = val_ds.cache().prefetch(buffer_size=autotune)

        return train_ds, val_ds, len(class_names)

    def use(self):
        """
        self.use()
        """

        modeldata = {"eb0":      [tf.keras.applications.EfficientNetB0,    224],
                     "eb1":      [tf.keras.applications.EfficientNetB1,    240],
                     "eb2":      [tf.keras.applications.EfficientNetB2,    260],
                     "eb3":      [tf.keras.applications.EfficientNetB3,    300],
                     "eb4":      [tf.keras.applications.EfficientNetB4,    340],
                     "eb5":      [tf.keras.applications.EfficientNetB5,    456],
                     "eb6":      [tf.keras.applications.EfficientNetB6,    528],
                     "eb7":      [tf.keras.applications.EfficientNetB7,    600],
                     "vgg16":    [tf.keras.applications.VGG16,             224],
                     "vgg19":    [tf.keras.applications.VGG19,             224],
                     "dn121":    [tf.keras.applications.DenseNet121,       224],
                     "dn169":    [tf.keras.applications.DenseNet169,       224],
                     "dn201":    [tf.keras.applications.DenseNet201,       224],
                     "irnv2":    [tf.keras.applications.InceptionResNetV2, 299],
                     "iv3":      [tf.keras.applications.InceptionV3,       299],
                     "mn":       [tf.keras.applications.MobileNet,         224],
                     "mnv2":     [tf.keras.applications.MobileNetV2,       224],
                     "mnv3l":    [tf.keras.applications.MobileNetV3Large,  224],
                     "mnv3s":    [tf.keras.applications.MobileNetV3Small,  224],
                     "rn101":    [tf.keras.applications.ResNet101,         224],
                     "rn101v2":  [tf.keras.applications.ResNet101V2,       224],
                     "rn152":    [tf.keras.applications.ResNet152,         224],
                     "rn152v2":  [tf.keras.applications.ResNet152V2,       224],
                     "rn50":     [tf.keras.applications.ResNet50,          224],
                     "rn50v2":   [tf.keras.applications.ResNet50V2,        224],
                     "xception": [tf.keras.applications.Xception,          299]}

        img_size = modeldata[self.model][1]
        train, val, class_num = self.load_img_data(self.path, img_size, img_size, self.batch_size)
        body = modeldata[self.model][0](input_shape=
                                        (img_size, img_size, 3),
                                        include_top=False,
                                        weights="imagenet")

        body.trainable = False
        average_layer = GlobalAveragePooling2D()
        prediction_layer = Dense(class_num, activation='softmax')

        body = tf.keras.Sequential([
            body,
            average_layer,
            prediction_layer
        ])

        if self.data_augmentation:
            data_augmentation = Sequential(
                [
                    layers.experimental.preprocessing.RandomFlip(
                        "horizontal",
                        input_shape=(
                            img_size,
                            img_size,
                            3)),
                    layers.experimental.preprocessing.RandomRotation(0.1),
                    layers.experimental.preprocessing.RandomZoom(0.1),
                ])
            model = Sequential(data_augmentation, body)
        else:
            model = Sequential(body)

        model.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy,
                      metrics=["accuracy"])

        history = model.fit(train,
                            epochs=self.epochs,
                            validation_data=val)
        if self.graph:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

        model.save(f"{self.save}.h5")


class ImageClassificationPredictor:
    """
    from quickai import ImageClassificationPredictor
    """
    def __init__(self, save, size, path, classes):
        self.save = save
        self.size = size
        self.path = path
        self.classes = classes
        self.pred()

    def pred(self):
        """
        self.pred()
        """
        model = tf.keras.models.load_model(f"{self.save}.h5")
        sunflower_path = pathlib.Path(self.path)

        img = tf.keras.preprocessing.image.load_img(
            sunflower_path, target_size=(self.size, self.size)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        output = [self.classes[np.argmax(score)], 100 * np.max(score)]

        return output
