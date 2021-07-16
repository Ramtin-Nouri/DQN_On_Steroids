from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, Input, Dense
from tensorflow.keras.models import Model

import TF2_Keras_Template as template

class NeuralNetwork(template.nnBase.NNBase):
    
    def __init__(self):
        #Only sets the name of this class
        self.networkName = "Autoencoder"
            
    def makeModel(self,inputShape,outputShape,args=[]):
        """
            overrides base function
            Create and return a Keras Model
        """
        
        observations_input = Input(shape=inputShape)

        conv1 = Conv2D(32, (3, 3), activation='relu',padding='same')(observations_input)
        pool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(32, (3, 3), activation='relu',padding='same')(pool1)
        pool2 = MaxPooling2D((2, 2))(conv2)
        conv3 = Conv2D(32, (3, 3), activation='relu',padding='same')(pool2)

        dense = Dense(2000)(conv3)

        conv7 = Conv2D(32, (3, 3), activation='relu',padding='same')(dense)
        up3 = UpSampling2D((2,2))(conv7)
        conv8 = Conv2D(32, (3, 3), activation='relu',padding='same')(up3)
        up4 = UpSampling2D((2,2))(conv8)
        conv9 = Conv2D(3, (3, 3), activation='relu',padding='same')(up4)
              
        model = Model(observations_input,conv9)
        model.compile(optimizer='adam', loss='mse')
        return model
