"""
RUN: python text_inferance.py
"""

from quickai import gpt_neo

text = gpt_neo("Hello", "2.7B")
print(text)
