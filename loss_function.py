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

    References
    -------
        This loss is better explained in the report of this project and in the references, mainly 
        Leon A.Gatys, Alexander S.Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks.
        DOI: 10.1109/CVPR.2016.265, June 27-30, 2016
        and 
        Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In
        Arxiv:1409.1556, August 28, 2017.
"""

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, style_layers, content_layers):
    """
    style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight, style_layers, content_layers)

    The outputs are given by the image_statistics and we mix content and style loss to penalize and get a compromise. They are 
    weighted by style_weight and content_weight. 

        Parameters
    ----------
        outputs : Dictionary
        style_targets : Tensor
        content_targets : Tensor
        style_weight : float
        content_weight : float
        style_layers : string list
        content_layers : string list

    Returns
    -------
        float
        It returns the loss.
    """
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss