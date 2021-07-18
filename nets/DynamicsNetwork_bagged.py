from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, Input, Dense, BatchNormalization
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

        x = Conv2D(32, (3, 3), activation='relu',padding='same', use_bias=False)(observations_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x1 = MaxPooling2D()(x)

        x = Conv2D(32, (5,5), activation='relu',padding='same', use_bias=False)(observations_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (5,5), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x2 = MaxPooling2D()(x)

        x = Conv2D(32, (8,8), activation='relu',padding='same', use_bias=False)(observations_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(32, (8,8), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        x3 = MaxPooling2D()(x)


        x = concatenate([x1,x2,x3])

        x = Dense(1000, use_bias=False)(x)
        x = BatchNormalization()(x)
        concat = concatenate([x,action_input],axis=3)
        drop = Dropout(0.1)(concat)

        x= UpSampling2D()(drop)
        x = Conv2D(32, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu',padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        out = Conv2D(3, (3, 3), activation='relu',padding='same')(x)
              
        model = Model([observations_input,action_input],out)
        model.compile(optimizer='adam', loss='mse')
        return model
