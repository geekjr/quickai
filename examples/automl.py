"""
RUN: python automl.py
"""

from quickai import ImageClassification

models = [
    "irnv2",
    "eb0",
    "eb1",
    "eb2",
    "eb3",
    "eb4",
    "eb5",
    "eb6",
    "eb7",
    "vgg16",
    "vgg19",
    "dn121",
    "dn169",
    "dn201",
    "iv3",
    "mn",
    "mnv2",
    "mnv3l",
    "mnv3s",
    "rn101",
    "rn101v2",
    "rn152",
    "rn152v2",
    "rn50",
    "rn50v2",
    "xception"]

for model in models:
    ImageClassification(
        model,
        "./train",
        f"cars{model}",
        epochs=1,
        graph=False)
