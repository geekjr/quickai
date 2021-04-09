from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy
from .helper import load_img_data
import matplotlib.pyplot as plt


def train_cnn_vgg19(path, save, batch_size=8, data_augmentation=False, epochs=20, graph=True):
    train, val, class_num = load_img_data(path, 224, 224, batch_size)
    vgg19 = tf.keras.applications.VGG19(input_shape=(224, 224, 3),
                                        include_top=False,
                                        weights='imagenet')
    vgg19.trainable = False
    average_layer = GlobalAveragePooling2D()
    prediction_layer = Dense(class_num, activation='softmax')

    body = tf.keras.Sequential([
        vgg19,
        average_layer,
        prediction_layer
    ])

    if data_augmentation:
        data_augmentation = Sequential(
            [
                layers.experimental.preprocessing.RandomFlip(
                    "horizontal", input_shape=(224, 224, 3)),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1),

            ]
        )
        model = Sequential(data_augmentation, body)
    else:
        model = Sequential(body)

    model.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy,
                  metrics=["accuracy"])

    history = model.fit(train,
                        epochs=epochs,
                        validation_data=val)
    if graph:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    model.save(f"{save}.h5")
