from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, UpSampling2D, concatenate, Input, Dense
from tensorflow.keras.models import Model

import TF2_Keras_Template as template

class NeuralNetwork(template.nnBase.NNBase):
    
    def __init__(self):
        #Only sets the name of this class
        self.networkName = "Autoencoder"
            
    def makeModel(self,inputShape,outputShape):
        """
            overrides base function
            Create and return a Keras Model
            Instead of outputShape as intended ,use second parameter for list of arguments
        """        
        input_ = Input(inputShape)
        x = Conv2D(8,(3,3),strides=(2,2))(input_)
        x = Conv2D(8,(3,3),strides=(2,2))(x)
        x = Conv2D(8,(3,3),strides=(2,2))(x)
        x = Flatten()(x)
        x = Dense(outputShape)(x)

        model = Model(inputs=input_,outputs=x)
        
        model.compile(loss='mse',optimizer="adam")
        return model
