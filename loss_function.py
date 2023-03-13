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
import aux_image_statistics


"""
This File is used to compute the Loss Function for our Deep Convolutional Neural Network. 
To compute this loss we have to extract information from the content image and the style image. Therefore, we will
use the image_statistics to extract it.
"""