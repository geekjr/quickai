"""
RUN: python text_fine_tuning.py
"""

from quickai import TextFineTuning

TextFineTuning("./aclImdb", "./FUNCTIONTESTCLASSIFICATION", "classification", ["pos", "neg"],
               epochs=1)  # Text classification

TextFineTuning("./wnut17train.conll", "./FUNCTIONTESTTOKENCLASSIFICATION", "token_classification",
               epochs=1)  # Token Classification

TextFineTuning("./squad", "./FUNCTIONTESTQA", "q+a",
               epochs=1)  # Q+A
