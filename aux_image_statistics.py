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

"""
This is an auxiliary class for the image_statistics class that extracts information from the images to compute the loss. 

We can obtain this pretrained VGG19 from tf.keras. It is a Deep Convolutional Neural Network trained in the Imagenet dataset with
purpose of classification, hence it is able to extract information from the images that is different from the per-pixel information
that can not extract content or style information. 
"""
def vgg_layers_names():

    """This function is only informative to extract the names of the VGG19 Layers"""
    #We will not use the last activation function because we do not want to do classification, only use intermediate layers.
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    print("VGG19 Layers: ")
    for layer in vgg.layers:
        print(layer.name)

def vgg_layers(layer_names):
  """ Given the layer_names, it loads a VGG19 network and it will return the output of the layer_names."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model


def vgg_layers_stat(image, layers):
    """
    This function will provide the shape, min,max and mean of the output layers ("layers") of the VGG19 when we 
    enter "image". 
    """
    style_extractor = vgg_layers(layers)
    style_outputs = style_extractor(image*255)

#Look at the statistics of each layer's output
    for name, output in zip(layers, style_outputs):
        print(name)
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())
        print()

def gram_matrix(input_tensor):
  """
  It is used to compute the Style Loss (It computes correlations and is able to extract Low-Level image information)
  """
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)
