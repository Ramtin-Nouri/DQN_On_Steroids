from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import TF2_Keras_Template as template

class NeuralNetwork(template.nnBase.NNBase):
    
    def __init__(self):
        #Only sets the name of this class
        self.networkName = "Autoencoder"
            
    def makeModel(self,inputShape,outputShape,args):
        """
            overrides base function
            Create and return a Keras Model
        """
        [basemodel,learningRate,lrdecay] = args
        layer_name = 'dense'
        encoder = Model(inputs=basemodel.input,
                                       outputs=basemodel.get_layer(layer_name).output)
        
        encoder.trainable = False
            
        input_ = Input(inputShape)
        x = encoder(input_)
        dense1 = Dense(200)(x)
        dense2 = Dense(100)(dense1)
        dense3 = Dense(50)(dense2)
        dense4 = Dense(outputShape)(dense3)

        model = Model(inputs=input_,outputs=dense4)
        
        model.compile(loss='mse',optimizer=Adam(lr=learningRate, decay = lrdecay))
        return model
