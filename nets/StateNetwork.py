from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, Input, Dense, BatchNormalization
from tensorflow.keras.models import Model

import TF2_Keras_Template as template


class NeuralNetwork(template.nnBase.NNBase):

    def __init__(self):
        # Only sets the name of this class
        self.networkName = "Autoencoder"

    def makeModel(self, inputShape, outputShape, args=[]):
        """
            overrides base function
            Create and return a Keras Model
        """

        observations_input = Input(shape=inputShape)

        x = Conv2D(32, (3, 3), activation='relu',
                   padding='same')(observations_input)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu',
                   padding='same', name="encoding")(x)
        x = Dropout(0.1)(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

        model = Model(observations_input, x)
        model.compile(optimizer='adam', loss='mse')
        return model
