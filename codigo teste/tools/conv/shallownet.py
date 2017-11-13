# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last"
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# define the first (and only) CONV => RELU layer
		model.add(Conv2D(32, (2, 2), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))

		# define the second CONV => RELU layer VINICIUS
		model.add(Conv2D(64, (2, 2), padding="same",
						 input_shape=inputShape))
		model.add(Activation("relu"))

		# Camada so para tentar regular e softmax classifier
		model.add(Flatten())
		# model.add(Dense(10))
		# model.add(Activation("relu"))
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model