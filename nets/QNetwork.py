from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, UpSampling2D, concatenate, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import TF2_Keras_Template as template

class NeuralNetwork(template.nnBase.NNBase):
    
    def __init__(self):
        #Only sets the name of this class
        self.networkName = "Autoencoder"
            
    def makeModel(self,inputShape,args):
        """
            overrides base function
            Create and return a Keras Model
            Instead of outputShape as intended ,use second parameter for list of arguments
        """
        [outputShape,basemodel,learningRate,lrdecay] = args
        layer_name = 'encoding'
        encoder = Model(inputs=basemodel.layers[0].input,
                                       outputs=basemodel.get_layer(layer_name).output)
        
        encoder.trainable = False
            
        input_ = Input(inputShape)
        x = encoder(input_)
        x = Conv2D(8,(3,3),strides=(2,2))(x)
        x = Conv2D(8,(3,3),strides=(2,2))(x)
        x = Conv2D(8,(3,3),strides=(2,2))(x)
        x = Flatten()(x)
        x = Dense(outputShape)(x)

        model = Model(inputs=input_,outputs=x)
        
        model.compile(loss='mse',optimizer=Adam(lr=learningRate, decay = lrdecay))
        return model
