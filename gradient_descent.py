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
import time
import PIL.Image
import time
import functools
import image_statistics
import loss_function
import utils

"""This File will compute the Gradient Descent to update the image to improve the loss given by loss_function"""



def clip_0_1(image):
  """To keep the pixels between 0 and 1"""
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


@tf.function()
def train_step(image, opt, extractor, style_targets, content_targets, style_weight, content_weight, style_layers, content_layers):
  """
  This function will compute a train step used in the optimization. 
  """
  #We use GradientTape to update the image
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = loss_function.style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, style_layers, content_layers)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

def optimization(content_image, style_image, extractor, style_targets, content_targets, style_layers, content_layers,epochs=5, steps_per_epoch=100):
    image = tf.Variable(content_image)#We will begin by the content_image
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight=1e-2
    content_weight=1e4

    start = time.time()
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image, opt, extractor, style_targets, content_targets, style_weight, content_weight, style_layers, content_layers)
            print(".", end='', flush=True)
    display.clear_output(wait=True)
    display.display(utils.tensor_to_image(image))
    print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))
