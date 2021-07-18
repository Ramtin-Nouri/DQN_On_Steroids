from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, Input, Dense
from tensorflow.keras.models import Model

import TF2_Keras_Template as template
import tensorflow as tf
tf.random.set_seed(111)


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
        action_input = Input(shape=(None,None,1))

        x = Conv2D(64, (8,8),strides=(2,2), activation='relu',padding='same')(observations_input)
        x = Conv2D(128, (6,6),strides=(2,2), activation='relu',padding='same')(x)
        x = Conv2D(128, (6,6),strides=(2,2), activation='relu',padding='same')(x)
        x = Conv2D(128, (6,6),strides=(2,2), activation='relu',padding='same')(x)
        x = Dense(1024)(x)
        x = Dense(2048)(x)
        x = Dropout(0.1)(x)

        concat = concatenate([x,action_input],axis=3)

        x = Dense(2048)(concat)
        x = Dense(1024)(x)
        x = Dropout(0.1)(x)

        x = UpSampling2D()(x)
        x = Conv2D(128, (6,6), activation='relu',padding='same')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(128, (6,6), activation='relu',padding='same')(x)
        x = Dropout(0.1)(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(64, (6,6), activation='relu',padding='same')(x)
        x = UpSampling2D((2,2))(x)
        out = Conv2D(3, (8,8), activation='relu',padding='same')(x)
              
        model = Model([observations_input,action_input],out)
        model.compile(optimizer='adam', loss='mse')
        return model
