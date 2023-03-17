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
This class is used for image treatment and plotting.
"""

def tensor_to_image(tensor):
  """
  tensor_to_image(tensor)

  Given a tensor it returns the image.

      Parameters
      ----------
          tensor : Tensor

      Returns
      -------
          Pil.Image
  """
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(img):
  """
  load_img(img)

  Given the path to an image, it loads the image.

  The maximum size of the image is 512 pixels.
      Parameters
    ----------
        path_to_img : string
          It is an URL

    Returns
    -------
        Tensor
        It returns the image.
  """
  max_dim = 512
  img = tf.cast(img, tf.float32)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  encoded_img = tf.io.encode_jpeg(img)

  img = tf.image.decode_image(encoded_img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = tf.reduce_max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = tf.expand_dims(img, axis=0)
    
  return img

def imshow(image, title=None):
  """
  imshow(image, title)
  It is used to show the image.
      Parameters
      ----------
          image : Tensor
          title : string, optional

  """
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)