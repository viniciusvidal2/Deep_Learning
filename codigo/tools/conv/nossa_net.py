# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K
from keras.layers.normalization import BatchNormalization


class NossaNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        act = "elu"
        ker1 = (3,3)
        ker2 = (2,2)

        chanDim = -1

        ################################################################
        #### REDE ROTATIVA #############################################
        ################################################################

        # # define the first (and only) CONV => ELU layer
        # model.add(Conv2D(20, ker1, padding="same",
        #                  input_shape=inputShape))
        # model.add(BatchNormalization())
        # model.add(Activation(act))
        # # model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=ker1, strides=3))
        #
        # # define the second CONV => ELU layer
        # model.add(Conv2D(20, ker1, padding="same"))
        # model.add(BatchNormalization())
        # model.add(Activation(act))
        # # model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=ker2, strides=3))
        #
        # # define the third CONV => ELU layer
        # model.add(Conv2D(50, ker2, padding="same"))
        # model.add(Activation(act))
        # model.add(MaxPooling2D(pool_size=ker2, strides=3))
        #
        # # define the fourth CONV => ELU layer
        # model.add(Conv2D(50, ker2, padding="same"))
        # model.add(Activation(act))
        # # model.add(MaxPooling2D(pool_size=ker2, strides=3))
        #
        # # define the fifth CONV => ELU layer
        # # model.add(Conv2D(64, ker2, padding="same"))
        # # model.add(Activation(act))
        # # model.add(Dropout(0.25))
        #
        # # define the sixth CONV => ELU layer
        # # model.add(Conv2D(128, ker2, padding="same"))
        # # model.add(Activation(act))
        # # model.add(Dropout(0.25))
        #
        # # Camada so para tentar regular e softmax classifier
        # model.add(Flatten())
        # # model.add(Dense(500))
        # model.add(Dropout(0.5))
        # model.add(Dense(classes))
        # model.add(Activation("softmax"))


        ################################################################
        #### REDE RAPIDA ###############################################
        ################################################################
        model.add(Conv2D(20, ker1, padding="same",
                         input_shape=inputShape))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation(act))
        model.add(MaxPooling2D(pool_size=ker1, strides=3))

        model.add(Conv2D(20, ker1, padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation(act))
        model.add(MaxPooling2D(pool_size=ker2, strides=3))

        model.add(Conv2D(50, ker2, padding="same"))
        model.add(Activation(act))

        model.add(Conv2D(50, ker2, padding="same"))
        model.add(Activation(act))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        ################################################################
        ################################################################

        ################################################################
        #### REDE PROFUNDA #############################################
        ################################################################
        # model.add(Conv2D(32, ker1, padding="same",
        #                  input_shape=inputShape))
        # # model.add(BatchNormalization(axis=chanDim))
        # model.add(Activation(act))
        # model.add(MaxPooling2D(pool_size=ker1, strides=3))
        #
        # model.add(Conv2D(32, ker1, padding="same"))
        # # model.add(BatchNormalization(axis=chanDim))
        # model.add(Activation(act))
        #
        # model.add(Conv2D(32, ker2, padding="same"))
        # model.add(Activation(act))
        # model.add(MaxPooling2D(pool_size=ker2, strides=3))
        #
        # model.add(Conv2D(64, ker2, padding="same"))
        # model.add(Activation(act))
        #
        # model.add(Conv2D(64, ker2, padding="same"))
        # model.add(Activation(act))
        #
        # model.add(Conv2D(64, ker2, padding="same"))
        # model.add(Activation(act))
        #
        # model.add(Conv2D(64, ker2, padding="same"))
        # model.add(Activation(act))
        #
        # model.add(Flatten())
        # model.add(Dropout(0.5))
        # model.add(Dense(classes))
        # model.add(Activation("softmax"))
        ################################################################
        ################################################################

        # return the constructed network architecture
        return model
