# Classe Lenet implementada para comparacao
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation

from keras.layers.core import SpatialDropout2D
from keras.layers.core import Dropout

from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as k

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Criando modelo sequencial
        model = Sequential()
        inputShape = (height, width, depth)

        # Iniciando a rede
        model.add(Conv2D(20, (5,5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Dropout(0.25))  # testando

        # Proxima camada
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Dropout(0.25))  # testando

        # Camada densa oculta
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Camada densa de saida
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model