"""
RUN: python image_classification.py
"""

from quickai import ImageClassification

ImageClassification("vgg16", "./train", "cars",)
