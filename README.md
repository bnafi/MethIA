# Artificial Intelligence Methods II: Style Transfer. (*March 2023*)

This repository contains the Style Transfer project for the subject "AI Methods II" of the first year of Master of "Mathematics and AI" of Paris-Saclay University.

## Introduction
A pastiche is an image that takes the content from a picture and the style of an artwork. Style Transfer problem constist of getting a pastiche from an style image and a content image. 

## Content
In this repository, we may find the report and the presentation that may introduce you to the subject.

Also, it is posible to find the code used to implement the style transfer. Unluckily, it is really slow to train, so we have also introduced the Arbitrary Style Transfer that is able to solve this problem in real time as is uses a pretrained Deep Convolutional Neural Network to create a style embedding. To use this last application, you may need to download tensorflow_hub.

To understand the different parameters, there is an application on streamlit that allows us to interactively change the images through the training parameters.

## Methodology
As explained in the report and in the references, Style Transfer is a difficult problem as it is not clear how to separate style from the content of an image. As per-pixel information is not representative of this characteristics, we have used the Loss Function introduced by Gatys(2016) to extract information from the outputs of some intermediate layers from the pretrained Deep Convolutional Neural Network VGG19.

## References
Mainly, 
Léon A.Gatys, Alexander S.Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks.
DOI: 10.1109/CVPR.2016.265, June 27-30, 2016.

Vincent Dumoulin, Jonathon Shlens, and Manjunath Kudlur. A learned representation for artistic style. In
Arxiv:1610.07629, 2017.

Vincent Dumoulin, Jonathon Shlens, Manjunath Kudlur, Honglak Lee, and Golnaz Ghiasi. Exploring the structure
of a real-time, arbitrary neural artistic stylization network. In Arxiv:1705.06830, August 28, 2017.

Other references can be found in the report.

