import os


class YOLOV4_Custom:
    def __init__(
            self,
            data_file="data/obj.data",
            cfg_file="cfg/yolov4-obj.cfg",
            media_type="image",
            video=None,
            image=None,
            weights="./yolov4.weights"):

        self.data_file = data_file
        self.cfg_file = cfg_file
        self.weights = weights
        self.image = image
        self.video = video

        if media_type == "image":
            self.predict_image()
        else:
            self.predict_video()

    def darknet(self, term):
        """[a function to invoke the darknet command]

        Args:
            term ([string]): [the command that needs to be executed after the darknet invocation]
        """
        command = f"darknet {term}"

        os.system(command)

    def predict_image(self):
        self.darknet(
            f"detector test {self.data_file} {self.cfg_file} {self.weights} {self.image}")

    def predict_video(self):
        self.darknet(
            f"detector demo {self.data_file} {self.cfg_file} {self.weights} {self.video}")
