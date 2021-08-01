import os


class YOLOV4_Train:
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

    def darknet(self, term):
        """[a function to invoke the darknet command]

        Args:
            term ([string]): [the command that needs to be executed after the darknet invocation]
        """
        command = f"darknet {term}"

        os.system(command)

    def generate_train(self, path):
        image_files = []
        os.chdir(os.path.join("data", "obj"))
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".jpg"):
                image_files.append(path + filename)
        os.chdir("..")
        with open("train.txt", "w") as outfile:
            for image in image_files:
                outfile.write(image)
                outfile.write("\n")
            outfile.close()
        os.chdir("..")

    def generate_test(self, path):
        image_files = []
        os.chdir(os.path.join("data", "test"))
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".jpg"):
                image_files.append(path + filename)
        os.chdir("..")
        with open("test.txt", "w") as outfile:
            for image in image_files:
                outfile.write(image)
                outfile.write("\n")
            outfile.close()
        os.chdir("..")

    def train(self):
        print("[INFO] Staring train.txt generation")
        self.generate_train(self.train_data_path)
        print("[INFO] Finished train.txt generation")
        print("[INFO] Staring test.txt generation")
        self.generate_test(self.test_data_path)
        print("[INFO] Finished test.txt generation")
        print("[INFO] Staring training")
        self.darknet(
            f"detector train {self.data_file} {self.cfg_file} {self.weights} -dont_show -map")
