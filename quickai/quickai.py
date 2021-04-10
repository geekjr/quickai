from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy
import matplotlib.pyplot as plt
import pathlib
import numpy as np


class ImageClassification:
    def __init__(
            self,
            model,
            path,
            save,
            batch_size=8,
            data_augmentation=False,
            epochs=20,
            graph=True):
        self.model = model
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
        data_dir = pathlib.Path(path)
        if grayscale:
            color_mode = "grayscale"
        else:
            color_mode = "rgb"
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
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds, len(class_names)

    def use(self):
        global train, val, body, class_num, grayscale, img_size
        if self.model == "eb0":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.EfficientNetB0(input_shape=(
                img_size, img_size, 3), include_top=False, weights='imagenet')
        elif self.model == "eb1":
            img_size = 240
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.EfficientNetB1(input_shape=(
                img_size, img_size, 3), include_top=False, weights='imagenet')
        elif self.model == "eb2":
            img_size = 260
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.EfficientNetB2(input_shape=(
                img_size, img_size, 3), include_top=False, weights='imagenet')
        elif self.model == "eb3":
            img_size = 300
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.EfficientNetB3(input_shape=(
                img_size, img_size, 3), include_top=False, weights='imagenet')
        elif self.model == "eb4":
            img_size = 380
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.EfficientNetB4(input_shape=(
                img_size, img_size, 3), include_top=False, weights='imagenet')
        elif self.model == "eb5":
            img_size = 456
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.EfficientNetB5(input_shape=(
                img_size, img_size, 3), include_top=False, weights='imagenet')
        elif self.model == "eb6":
            img_size = 528
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.EfficientNetB6(input_shape=(
                img_size, img_size, 3), include_top=False, weights='imagenet')
        elif self.model == "eb7":
            img_size = 600
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.EfficientNetB7(input_shape=(
                img_size, img_size, 3), include_top=False, weights='imagenet')
        elif self.model == "vgg16":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.VGG16(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "vgg19":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.VGG19(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "dn121":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.DenseNet121(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "dn169":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.DenseNet169(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "dn201":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.DenseNet201(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "irnv2":
            img_size = 299
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.InceptionResNetV2(input_shape=(
                img_size, img_size, 3), include_top=False, weights='imagenet')
        elif self.model == "iv3":
            img_size = 299
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.InceptionV3(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "mn":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.MobileNet(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "mnv2":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.MobileNetV2(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "mnv3l":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.MobileNetV3Large(input_shape=(
                img_size, img_size, 3), include_top=False, weights='imagenet')
        elif self.model == "mnv3s":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.MobileNetV3Small(input_shape=(
                img_size, img_size, 3), include_top=False, weights='imagenet')
        elif self.model == "rn101":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.ResNet101(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "rn101v2":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.ResNet101V2(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')

        elif self.model == "rn152":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.ResNet152(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "rn152v2":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.ResNet152V2(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')

        elif self.model == "rn50":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.ResNet50(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "rn50v2":
            img_size = 224
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.ResNet50V2(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        elif self.model == "xception":
            img_size = 299
            train, val, class_num = self.load_img_data(
                self.path, img_size, img_size, self.batch_size)
            body = tf.keras.applications.Xception(
                input_shape=(
                    img_size,
                    img_size,
                    3),
                include_top=False,
                weights='imagenet')
        else:
            print("Model not found")

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
    def __init__(self, save, size, path, classes):
        self.save = save
        self.size = size
        self.path = path
        self.classes = classes
        self.pred()

    def pred(self):
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
