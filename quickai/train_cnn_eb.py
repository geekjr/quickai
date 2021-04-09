from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy
from .helper import load_img_data
import matplotlib.pyplot as plt


def train_cnn_eb(path, save, batch_size=8, data_augmentation=False, epochs=20, graph=True, eb_type=0):
    if eb_type == 0:
        train, val, class_num = load_img_data(path, 224, 224, batch_size)
        eb = tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3),
                                            include_top=False,
                                            weights='imagenet')
    if eb_type == 1:
        train, val, class_num = load_img_data(path, 224, 224, batch_size)
        eb = tf.keras.applications.EfficientNetB1(input_shape=(240, 240, 3),
                                            include_top=False,
                                            weights='imagenet')
    if eb_type == 2:
        train, val, class_num = load_img_data(path, 224, 224, batch_size)
        eb = tf.keras.applications.EfficientNetB2(input_shape=(260, 260, 3),
                                            include_top=False,
                                            weights='imagenet')
    if eb_type == 3:
        train, val, class_num = load_img_data(path, 224, 224, batch_size)
        eb = tf.keras.applications.EfficientNetB3(input_shape=(300, 300, 3),
                                            include_top=False,
                                            weights='imagenet')
    if eb_type == 4:
        train, val, class_num = load_img_data(path, 224, 224, batch_size)
        eb = tf.keras.applications.EfficientNetB4(input_shape=(380, 380, 3),
                                            include_top=False,
                                            weights='imagenet')
    if eb_type == 5:
        train, val, class_num = load_img_data(path, 224, 224, batch_size)
        eb = tf.keras.applications.EfficientNetB5(input_shape=(456, 456, 3),
                                            include_top=False,
                                            weights='imagenet')
    if eb_type == 6:
        train, val, class_num = load_img_data(path, 224, 224, batch_size)
        eb = tf.keras.applications.EfficientNetB6(input_shape=(528, 528, 3),
                                            include_top=False,
                                            weights='imagenet')
    if eb_type == 7:
        train, val, class_num = load_img_data(path, 224, 224, batch_size)
        eb = tf.keras.applications.EfficientNetB7(input_shape=(600, 600, 3),
                                            include_top=False,
                                            weights='imagenet')
    eb.trainable = False
    average_layer = GlobalAveragePooling2D()
    prediction_layer = Dense(class_num, activation='softmax')

    body = tf.keras.Sequential([
        eb,
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
