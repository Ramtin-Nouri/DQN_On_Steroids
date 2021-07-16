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

        conv1 = Conv2D(32, (3, 3), activation='relu',padding='same')(observations_input)
        pool1 = MaxPooling2D((2, 2))(conv1)
        drop1 = Dropout(0.1)(pool1)

        conv2 = Conv2D(64, (3, 3), activation='relu',padding='same')(drop1)
        pool2 = MaxPooling2D((2, 2))(conv2)
        drop2 = Dropout(0.1)(pool2)

        conv3 = Conv2D(128, (3, 3), activation='relu',padding='same')(drop2)
        pool3 = MaxPooling2D((2, 2))(conv3)
        drop3 = Dropout(0.1)(pool3)

        conv4 = Conv2D(256, (3, 3), activation='relu',padding='same')(drop3)
        

        dense1 = Dense(2000)(conv4)#<- Hidden Representation
        concat = concatenate([dense1,action_input],axis=3)

        
        
        conv5 = Conv2D(256, (3, 3), activation='relu',padding='same')(concat)
        up1 = UpSampling2D((2,2))(conv5)
        drop5 = Dropout(0.1)(up1)
        

        conv7 = Conv2D(64, (3, 3), activation='relu',padding='same')(drop5)
        up2 = UpSampling2D((2,2))(conv7)
        drop7 = Dropout(0.1)(up2)
        

        conv8 = Conv2D(32, (3, 3), activation='relu',padding='same')(drop7)
        up3 = UpSampling2D((2,2))(conv8)
        drop8 = Dropout(0.1)(up3)

        conv9 = Conv2D(3, (3, 3), activation='relu',padding='same')(drop8)
              
        model = Model([observations_input,action_input],conv9)
        model.compile(optimizer='adam', loss='mse')
        return model
