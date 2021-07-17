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
        action_input = Input(shape=(None,None,1))

        x = Conv2D(32, (3, 3), activation='relu',padding='same')(observations_input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
        x = MaxPooling2D()(x)

        dense = Dense(1000)(x)
        concat = concatenate([dense,action_input],axis=3)

        x= UpSampling2D()(concat)
        x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
        out = Conv2D(3, (3, 3), activation='relu',padding='same')(x)
              
        model = Model([observations_input,action_input],out)
        model.compile(optimizer='adam', loss='mse')
        return model
