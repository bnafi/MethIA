import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import numpy as np
import time
import PIL.Image
import time
import functools
import image_statistics
import loss_function
import utils
import streamlit as st

"""
This File will compute the Gradient Descent to update the image to improve the loss given by loss_function.
  The method optimization will compute the output image given the content and style image and the hyperparameters that we have 
   tested to understand their  effect in the Style Transfer problem. 


"""



def clip_0_1(image):
  """
  clip_0_1(image):

  To keep the pixels between 0 and 1

  Parameters
  ----------
    image : Tensor

  Returns
  -------
    Tensor
    It returns the clipped tensor.
  """
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


@tf.function()
def train_step(image, opt, extractor, style_targets, content_targets, style_weight, content_weight, style_layers, content_layers):
  """
  This function will compute a train step used in the optimization. It makes all the changes directly in the image.

  Parameters
  ----------
    image : Tensor
    opt : tf.optimizer
    extractor : StyleContentModel
    style_targets : Tensor
    content_targets : Tensor
    style_weight : float
    content_weight : float
    style_layers : string list
    content_layers : string list

  
  """
  #We use GradientTape to update the image
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = loss_function.style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, style_layers, content_layers)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

def optimization(init_image,content_image, style_image, lam, style_layers, content_layers,epochs=2, steps_per_epoch=10):    
  """
    This is the function that will take an input image and will generate an output image that will minimize the loss function,
    parametrized by some weightening. init_image is the first image from where we will begin (Normally content_image)
    The content_image is the image from which we will compute the content loss. The style_image is the image from which we will 
    compute the style loss. The style_layers are the VGG19 layers that we will use to compute the style loss and the content_layers
     are the VGG19 layers that we will use to compute the content loss

    Parameters
    ----------
      init_image : Tensor
      content_image : Tensor
      style _image : Tensor
      lam : float
      style_layers : string list
      content_layers : string list
      epoch : int, optional
      steps_per_epoch : int, optional

    Returns
     -------
      Tensor
        It returns the optimized image.

  """
  #extractor: it is the network that will extract the style and the content from the images
  extractor = image_statistics.StyleContentModel(style_layers, content_layers)
  style_targets = extractor(style_image)['style']
  content_targets = extractor(content_image)['content']
  image = tf.Variable(init_image) #Normally, we will begin by the content_image
  opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
  style_weight=lam*(1e-2/1e4)
  content_weight=1

  start = time.time()
  step = 0
  for n in range(epochs):
    for m in range(steps_per_epoch):
      step += 1
      st.write(m)
      train_step(image, opt, extractor, style_targets, content_targets, style_weight, content_weight, style_layers, content_layers)
    st.image(utils.tensor_to_image(image))
    st.write("Train step :",step)
  return 
