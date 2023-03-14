
import tensorflow_hub as hub
import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools
import utils
"""
This class is only used to show how the embedding offered by the pretrained Inception_V3(the style predict network) fasten the style transfer 
making it possible to use the Style Transfer in real time. 

This module implements the algorithm proposed at https://arxiv.org/abs/1705.06830
"""
def fast_stylization(content_image, style_image):
    """
    It takes the content_image and the Style_image and it computes the result through the network given at tf_hub.
    """
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    return utils.tensor_to_image(stylized_image)