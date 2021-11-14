<p align="center">
  <img src="https://raw.githubusercontent.com/geekjr/quickai/main/assets/quickai.png" alt="QuickAI logo"/>
</p>

## QuickAI is a Python library that makes it extremely easy to experiment with state-of-the-art Machine Learning models.

<table>
	<tr>
		<td align="center">PyPI Counter</td>
		<td align="center"><a href="http://pepy.tech/project/quickai"><img src="http://pepy.tech/badge/quickai"></a></td>
	</tr>
	<tr>
		<td align="center">PyPI Counter</td>
		<td align="center"><a href="http://pepy.tech/project/quickai/month"><img src="http://pepy.tech/badge/quickai/month"></a></td>
	</tr>
	<tr>
		<td align="center">PyPI Counter</td>
		<td align="center"><a href="http://pepy.tech/project/quickai/week"><img src="http://pepy.tech/badge/quickai/week"></a></td>
	</tr>
	<tr>
		<td align="center">Github Stars</td>
		<td align="center"><a href="https://github.com/geekjr/quickai"><img src="https://img.shields.io/github/stars/geekjr/quickai.svg?style=social&label=Stars"></a>			</td>
	</tr>
</table>

### Note: Even if you are not using YOLO, you will still need a file in your curent working directory called coco.names. If you are not using YOLO, this file can be empty.  

Announcement video

[![Announcement video](https://img.youtube.com/vi/kK46sJphjIs/0.jpg)](https://www.youtube.com/watch?v=kK46sJphjIs)

Demo https://deepnote.com/project/QuickAI-1r_4zvlyQMa2USJrIvB-kA/%2Fnotebook.ipynb

### Motivation

When I started to get into more advanced Machine Learning, I started to see how these famous neural network
architectures(such as EfficientNet), were doing amazing things. However, when I tried to implement these architectures
to problems that I wanted to solve, I realized that it was not super easy to implement and quickly experiment with these
architectures. That is where QuickAI came in. It allows for easy experimentation of many model architectures quickly.

### Dependencies:

Tensorflow, PyTorch, Sklearn, Matplotlib, Numpy, and Hugging Face Transformers. You should install TensorFlow and PyTorch following the instructions from their respective websites.

### Docker container:

To avoid setting up all the dependencies above, you can use the QuickAI [Docker Container](https://hub.docker.com/r/geekjr/quickai):

First pull the container:
`docker pull geekjr/quickai`

Then run it:

* CPU(on an Apple silicon Mac, you will need the `--platform linux/amd64` flag and Rosetta 2 installed):
`docker run -it ufoym/deepo:cpu bash`

* GPU:
`docker run --gpus all -it ufoym/deepo bash`

### Why you should use QuickAI

QuickAI can reduce what would take tens of lines of code into 1-2 lines. This makes fast experimentation very easy and
clean. For example, if you wanted to train EfficientNet on your own dataset, you would have to manually write the data
loading, preprocessing, model definition and training code, which would be many lines of code. Whereas, with QuickAI,
all of these steps happens automatically with just 1-2 lines of code.

### The following models are currently supported:

1. #### Image Classification
   - EfficientNet B0-B7
   - VGG16
   - VGG19
   - DenseNet121
   - DenseNet169
   - DenseNet201
   - Inception ResNet V2
   - Inception V3
   - MobileNet
   - MobileNet V2
   - MobileNet V3 Small & Large
   - ResNet 101
   - ResNet 101 V2
   - ResNet 152
   - ResNet 152 V2
   - ResNet 50
   - ResNet 50 V2
   - Xception
2. #### Natural Language Processing

   - GPT-NEO 125M(Generation, Inference)
   - GPT-NEO 350M(Generation, Inference)
   - GPT-NEO 1.3B(Generation, Inference)
   - GPT-NEO 2.7B(Generation, Inference)
   - GPT-J 6B(Generation, Inference)-BETA
   - Distill BERT Cased(Q&A, Inference and Fine Tuning)
   - Distill BERT Uncased(Named Entity Recognition, Inference)
   - Distil BART (Summarization, Inference)
   - Distill BERT Uncased(Sentiment Analysis & Text/Token Classification, Inference and Fine Tuning)

3. #### Object Detection
   - YOLOV4
   - YOLOV4 Tiny

### Installation

`pip install quickAI`

### How to use

Please see the examples folder for details. For the YOLOV4, you can download weights from [here](https://github.com/geekjr/quickai/releases/download/1.3.0/checkpoints.zip). Full documentation is in the wiki section of the repo.

### Issues/Questions

If you encounter any bugs, please open a new issue so they can be corrected. If you have general questions, please use the discussion section.

### Credits

Most of the code for the YOLO implementations were taken from "The AI Guy's" [tensorflow-yolov4-tflite](https://github.com/theAIGuysCode/tensorflow-yolov4-tflite) & [YOLOv4-Cloud-Tutorial](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial) repos. Without this, the YOLO implementation would not be possible. Thank you!
