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
import loss_function
import gradient_descent
import image_statistics

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
content_image = utils.load_img(content_path)
style_image = utils.load_img(style_path)
utils.imshow(style_image)
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
content_layers = ['block5_conv2']
#loss_function.vgg_layers_stat(style_image, style_layers)
extractor = image_statistics.StyleContentModel(style_layers, content_layers)
results = extractor(tf.constant(content_image))
extractor = image_statistics.StyleContentModel(style_layers, content_layers)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

gradient_descent.optimization(content_image, style_image, extractor, style_targets, content_targets, style_layers, content_layers,epochs=2, steps_per_epoch=10)