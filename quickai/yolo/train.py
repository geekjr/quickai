import cv2
from darknet_invoke import darknet
from generate_train import generate_train
from generate_test import generate_test


class YOLOV4Train:
    def __init__(
            self,
            train_data_path="data/obj/",
            test_data_path="data/test/",
            data_file="data/obj.data",
            cfg_file="cfg/yolov4-obj.cfg",
            weights="./yolov4.conv.137"):

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data_file = data_file
        self.cfg_file = cfg_file
        self.weights = weights
        self.train()

    def train(self):
        print("[INFO] Staring train.txt generation")
        generate_train(self.train_data_path)
        print("[INFO] Finished train.txt generation")
        print("[INFO] Staring test.txt generation")
        generate_test(self.test_data_path)
        print("[INFO] Finished test.txt generation")
        print("[INFO] Staring training")
        darknet(
            f"detector train {self.data_file} {self.cfg_file} {self.weights} -dont_show -map")
