import time

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from .utils import *
from .yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image

# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import ConfigProto
# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import InteractiveSession
import cv2
import numpy as np
import tensorflow as tf


class YOLOV4:
    """
    Method yolov4_detect is default
    """

    def __init__(
            self, media_type, image="kite.jpg", video="road.mp4", output_format="XVID", yolo_classes="coco.names",
            framework="tf",
            weights="./checkpoints/yolov4-416",
            size=416, tiny=False,
            model="yolov4", output="./detections/", iou=0.45, score=0.25, dont_show=False):

        self.video = video
        self.image = image
        self.yolo_classes = yolo_classes
        self.framework = framework
        self.weights = weights
        self.size = size
        self.tiny = tiny
        self.model = model
        self.output = output
        self.iou = iou
        self.score = score
        self.dont_show = dont_show
        self.output_format = output_format

        if media_type == "image":
            self.yolov4_detect_image()
        elif media_type == "video":
            self.yolov4_detect_video()

    def yolov4_detect_image(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = load_config(self)
        input_size = self.size
        image = self.image

        # load model
        if self.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=self.weights)
        else:
            saved_model_loaded = tf.saved_model.load(
                self.weights, tags=[tag_constants.SERVING])

        # loop through images in list and run Yolov4 model on each
        original_image = cv2.imread(image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        if self.framework == 'tflite':
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
            interpreter.set_tensor(input_details[0]['index'], images_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(
                output_details[i]['index']) for i in range(len(output_details))]
            if self.model == 'yolov3' and self.tiny == True:
                boxes, pred_conf = filter_boxes(
                    pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(
                    pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            infer = saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou,
            score_threshold=self.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(),
                     valid_detections.numpy()]

        # read in all class names from config
        class_names = read_class_names(self.yolo_classes)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['person']

        image = draw_bbox(original_image, pred_bbox,
                                allowed_classes=allowed_classes)

        image = Image.fromarray(image.astype(np.uint8))
        if not self.dont_show:
            image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.output + 'detection' + '.png', image)

    def yolov4_detect_video(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        session = InteractiveSession(config=config)
        print(self)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = load_config(self)
        input_size = self.size
        video_path = self.video

        if self.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=self.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        else:
            saved_model_loaded = tf.saved_model.load(
                self.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']

        # begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        out = None

        if self.output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*self.output_format)
            out = cv2.VideoWriter(self.output, codec, fps, (width, height))

        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break

            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            if self.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(
                    output_details[i]['index']) for i in range(len(output_details))]
                if self.model == 'yolov3' and self.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=self.iou,
                score_threshold=self.score
            )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(),
                         valid_detections.numpy()]
            image = draw_bbox(frame, pred_bbox)
            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            result = np.asarray(image)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if not self.dont_show:
                cv2.imshow("result", result)

            if self.output:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


# YOLOV4_detect("kite.jpg")


'''
if __name__ == '__main__':
    try:
        app.run(YOLOV4_detect())
    except SystemExit:
        pass
'''
