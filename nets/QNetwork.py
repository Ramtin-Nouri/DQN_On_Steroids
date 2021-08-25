from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, UpSampling2D, concatenate, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

import TF2_Keras_Template as template

class NeuralNetwork(template.nnBase.NNBase):
    
    def __init__(self):
        #Only sets the name of this class
        self.networkName = "Agent"
            
    def makeModel(self,inputShape,args):
        """
            overrides base function
            Create and return a Keras Model
            Instead of outputShape as intended ,use second parameter for list of arguments
        """        

        [outputShape,learningRate] = args

        input_ = Input(inputShape)
        x = Conv2D(8,(3,3),strides=(2,2), activation="relu")(input_)
        x = Conv2D(8,(3,3),strides=(2,2), activation="relu")(x)
        x = Flatten()(x)
        x = Dense(outputShape)(x)

        model = Model(inputs=input_,outputs=x)
        
        model.compile(loss=Huber(),optimizer=Adam(learning_rate=learningRate))
        return model
