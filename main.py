from __future__ import division,print_function
import json
#glob for finding all the pathnames matching a specified pattern according to the rules used by the Unix shell
from glob import glob
#numpy import
import numpy as np
#scipy import
import scipy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import utils; reload(utils)
from utils import plots, get_batches, plot_confusion_matrix, get_data

#flask
from flask import Flask, jsonify, request, render_template
import uuid
import time
from glob import glob

#numpy modules and scipy modules import
from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

#keras import
import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

#Import utils
from utils import *
from vgg16 import Vgg16

app = Flask(__name__) # create a Flask app

#initial model
vgg = Vgg16()
model = vgg.model

model_path = 'D:/ML/fastai/Projects/hotdog_nothotdog/data/models/'

#Conv layer
def make_conv_layers():
	layers = model.layers
	#Index of the last convolutional layer
	last_conv_idx = [index for index, layer in enumerate(layers) if type(layer) is Convolution2D][-1]
	conv_layers = layers[:last_conv_idx+1]

	return conv_layers

#Dense layer
def get_bn_layers(p):
	conv_layers = make_conv_layers()
	return [
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(4096, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(2, activation='softmax')
        ]

#final model
def get_final_cnn():
	bn_model = Sequential(get_bn_layers(0.6))
	model_path = 'D:/ML/fastai/Projects/hotdog_nothotdog/data/models/final3.h5'
	bn_model.load_weights(model_path)
	bn_layers = get_bn_layers(0.6)

	conv_layers = make_conv_layers()
	final_model = Sequential(conv_layers)

	#Set conv_model not trainable
	for layer in final_model.layers: 
		layer.trainable = False

	#Add bn_layers
	for layer in bn_layers: 
		final_model.add(layer)

	#Compile the model
	print ('load weights...')
	for l1,l2 in zip(bn_model.layers, bn_layers):
		l2.set_weights(l1.get_weights())
	final_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
	return final_model

def get_img(test_path):
	test_data = get_data(test_path)
	return test_data

@app.route('/', methods=['POST', 'GET'])
def index():
	if request.method == 'GET':
		return render_template("index.html")
	elif request.method == 'POST' and 'uploaded_image' in request.files:
		#Clean out images folder
		try:
			os.remove(glob(os.path.join("static", "unknown", "*.png"))[0])
		except IndexError as err:
			print("No files to clean up")

		noise = str(uuid.uuid4())
		image_name = "%simage.png" % noise
		image_path = os.path.join("static", "unknown", image_name)
		image = request.files.get('uploaded_image', '')
		image.save(image_path)
		test_data = get_img(os.path.join("static"))
		results = model.predict_classes(test_data)

		return_string = "Hotdog" if results == 0 else "Not Hotdog"
		return render_template("results.html", 
			                   return_string=return_string,
			                   image_name=image_name)


if __name__ == '__main__':
  print ('initialize model...')
  model = get_final_cnn()

  app.run(debug=True)


