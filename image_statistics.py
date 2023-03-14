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
This File is used to compute the Extractor that will extract style and content image statistics using a Deep Convolutional Neural Network. 

We have to notice that this Loss Function is not well defined as it could be in classification for example. This is 
because it is difficult to define the Style and Content of an image. The idea is to use a Deep Convolution Neural 
Network pretrained for image classification (VGG19) in ImageNet dataset. With some intermediate layers outputs,
we are able to extract some high-level (Content) and low-level (Style) from the image to compute image statistics.

We use methods from aux_image_statistics to compute the loss.
See also
----------
image_statistics

References
----------
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf 

"""

class StyleContentModel(tf.keras.models.Model):
  """
  We create a Class that inherits from models of keras. For the initialization we have to provide a list with the name of the 
  style and content layers (VGG19 layers) from which we want to extract the information. The valid names will be given in the
  streamlit. 
   Attributes
    ----------
    vgg : tf.keras.Model
        The vgg layers(style and content)
    style_layers: string list
        The List of vgg layers from which we will compute the style loss
    content_layers: string 
        The vgg layer from which we will compute the content loss
    num_style_layers: int
        Number of style layers
    vgg.trainable : boolean
        If the vgg is trainable or we use the pretrained one

    Methods
    -------
    __init__(style_layers, content_layers)
        It instantiates the class with the given layers to extract information
    call(inputs)
        It returns the outputs of the vgg after passing inputs, treated to afterward compute the loss
  """
  def __init__(self, style_layers, content_layers):
    """
    __init__(style_layers, content_layers)
    Parameters
    ----------
        style_layers : string list
        content_layers : string list

    """
    super(StyleContentModel, self).__init__()
    self.vgg = aux_image_statistics.vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    """
    This function returns the content layer output and style layers outputs given in the initialization of the class. 
    The style outputs are also treated with the gram matrix to extract correlations between different layers.

    Parameters
    ----------
    inputs : image
        It is the image that we want to pass through the information extractor.
    Returns
    -------
    Dict
        This dictionary will have as keys the content and style, with a dictionary as value. This dictionarys 
        contain the name of the layer and the output computed.
        
    """
    #Expects float input in [0,1]
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)#It is the style and content extractor
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [aux_image_statistics.gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}