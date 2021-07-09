from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, Input, Dense
from tensorflow.keras.models import Model

import TF2_Keras_Template as template

class NeuralNetwork(template.nnBase.NNBase):
    
    def __init__(self):
        #Only sets the name of this class
        self.networkName = "Autoencoder"
            
    def makeModel(self,encoderModel,outputShape,learningRate,lrdecay):
        """
            overrides base function
            Create and return a Keras Model
        """
        

        layer_name = 'dense1' #TODO: find out real name
        model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
        
        for layer in model.layers[:]:
            layer.trainable = False
            
        model.add(Dense(200))
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(outputShape))
        
        model.compile(loss='mse',optimizer=Adam(lr=learningRate, decay = lrdecay))
        return model
