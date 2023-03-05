"""
This script takes advantage of the TensorFlow Lite Micro library to create the training data we need to train a model that runs on a
very constrained device. In this example it is an Arduino micro controller.

The original training script loads the entire dataset into memory. What we do here is transform the data, store it on disk and create 
a dataloader object to be more lazy and memory efficient.
"""

import tensorflow as tf


def tf_dataset()


def main():
    PREPROCESS = 'micro'
    WINDOW_STRIDE = 20