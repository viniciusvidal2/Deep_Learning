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

        # model.add(Conv2D(50, ker2, padding="same",
        #                  input_shape=inputShape))
        # model.add(Activation(act))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=ker1, strides=2))
        #
        # model.add(Conv2D(100, ker2, padding="same"))
        # model.add(Activation(act))
        # model.add(BatchNormalization(axis=chanDim))
        #
        # model.add(Conv2D(100, ker2, padding="same"))
        # model.add(Activation(act))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=ker1, strides=2))
        #
        # model.add(Conv2D(100, ker2, padding="same"))
        # model.add(Activation(act))
        # model.add(BatchNormalization(axis=chanDim))
        #
        # model.add(Conv2D(100, ker2, padding="same"))
        # model.add(Activation(act))
        # model.add(BatchNormalization(axis=chanDim))
        #
        # model.add(Conv2D(100, ker2, padding="same"))
        # model.add(Activation(act))
        # model.add(BatchNormalization(axis=chanDim))
        #
        # model.add(Conv2D(100, ker2, padding="same"))
        # model.add(Activation(act))
        # # model.add(BatchNormalization(axis=chanDim))
        #
        # model.add(Conv2D(100, ker2, padding="same"))
        # model.add(Activation(act))
        # # model.add(BatchNormalization(axis=chanDim))
        #
        # model.add(Flatten())
        # model.add(Dropout(0.5))
        # model.add(Dense(100))
        # model.add(Dense(classes))
        # model.add(Activation(act))
        # model.add(Activation("softmax"))
        ################################################################
        ################################################################

        ################################################################
        #### REDE RAPIDA ###############################################
        ################################################################
        # model.add(Conv2D(20, ker1, padding="same",
        #                  input_shape=inputShape))
        # model.add(Activation(act))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=ker1, strides=3))
        #
        # model.add(Conv2D(20, ker1, padding="same"))
        # model.add(Activation(act))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=ker2, strides=3))
        #
        # model.add(Conv2D(50, ker2, padding="same"))
        # model.add(Activation(act))
        #
        # model.add(Conv2D(50, ker2, padding="same"))
        # model.add(Activation(act))
        #
        # model.add(Flatten())
        # model.add(Dropout(0.5))
        # model.add(Dense(classes))
        # model.add(Activation("softmax"))
        ################################################################
        ################################################################

        ################################################################
        #### REDE PROFUNDA #############################################
        ################################################################
        model.add(Conv2D(32, ker1, padding="same",
                         input_shape=inputShape))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation(act))
        model.add(MaxPooling2D(pool_size=ker1, strides=3))

        model.add(Conv2D(32, ker1, padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation(act))

        model.add(Conv2D(32, ker2, padding="same"))
        model.add(Activation(act))
        model.add(MaxPooling2D(pool_size=ker2, strides=3))

        model.add(Conv2D(64, ker2, padding="same"))
        model.add(Activation(act))

        model.add(Conv2D(64, ker2, padding="same"))
        model.add(Activation(act))

        model.add(Conv2D(64, ker2, padding="same"))
        model.add(Activation(act))

        model.add(Conv2D(64, ker2, padding="same"))
        model.add(Activation(act))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        ################################################################
        ################################################################

        # return the constructed network architecture
        return model
