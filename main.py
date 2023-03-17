import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


import streamlit as st
import numpy as np
import PIL.Image
import time
import functools
import utils
import loss_function
import gradient_descent
import image_statistics
import requests
from io import BytesIO
import tensorflow_hub as hub
import arbitrary_ST

"""
This file gives just an example of the use of the architecture. 
We use the gradient descent to optimize our initial image

If we want to change the content image and style image we would have to change the URLs given.

"""


# This function will return the output image. 
# We notice that the first parameter of the function is the initial image for the optimization(we have chosen the content_image to have
# good visual results with not many steps)


st.write("Insérez un URL pour votre image de contenu et votre image de style. (Si l'application affiche 0, le format de l'URL n'est pas bon. Il faut essayer avec une autre image, ou un autre URL.)")
url_content = st.text_input("Entrez l\'URL de la photo de contenu :")
st.write(" (ex: https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg)")

url_style = st.text_input(r"Entrez l'URL de la photo de style :")
st.write("(ex: https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg)")

if url_content:
  st.image(url_content, caption = "Image de contenu") 
  content_response = requests.get(url_content)
  content_data = content_response.content
  content_image = PIL.Image.open(BytesIO(content_data))
  content_image = utils.load_img(content_image)

if url_style:
  st.image(url_style, caption = "Image de style")
  style_response = requests.get(url_style)
  style_data = style_response.content
  style_image = PIL.Image.open(BytesIO(style_data))
  style_image = utils.load_img(style_image)

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

st.write("Voici les couches de l'architecture de VGG19 :")
url = "https://miro.medium.com/v2/resize:fit:1200/1*U6_b8aNUtOgdkRlP3b9Vpg.jpeg"
vg = PIL.Image.open(requests.get(url,stream = True).raw)
st.image(vg, caption = "Architecture de VGG19")
E=[]
for layer in vgg.layers:
  E.append(layer.name)

st.write(E)

st.write('Choisissez parmi les couches celles qui calculeront la content loss et la style loss.')

content_layers = st.text_input(r"Donnez le nom des couches de contenu :").split(",")
st.write("(ex: block5_conv2)")
style_layers = st.text_input(r"Donnez le nom des couches de style :").split(",")
st.write("(ex: block1_conv1,block2_conv1,block3_conv1,block4_conv1,block5_conv1)")

st.write(content_layers)
st.write(style_layers)

lam = st.slider("Valeur du poids d'interpolation",0.0,1.0, step = 0.1, value = 0.5)
st.write("On peut alors commencer l'entraînement du modèle. Spécifiez les paramètres d'entraînement du modèle. L'entraînement peut prendre du temps.")

epochs = st.slider("Nombre d'epochs",0,10,3)
steps_per_epoch = st.slider("Nombre de steps par epoch",0,100,10)

gradient_descent.optimization(content_image,content_image, style_image, lam, style_layers, content_layers,epochs=2, steps_per_epoch=10)

st.write("On peut aussi voir le transfert de style par fast stylization à l'aide du module hub :")
st.image(arbitrary_ST.fast_stylization(content_image,style_image), caption = "Image stylisée obtenue avec hub")