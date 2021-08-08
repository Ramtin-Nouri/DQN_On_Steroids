from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, concatenate, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow_addons.optimizers import AdamW

import TF2_Keras_Template as template
import tensorflow as tf
import numpy as np
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

        x = Conv2D(32, (3, 3), activation='relu',padding='same')(observations_input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu',padding='same')(x)
        x = MaxPooling2D()(x)
        x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
        x = MaxPooling2D()(x)

        x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(32, (3, 3), activation='relu',padding='same',name="encoding")(x)

        concat = concatenate([x,action_input],axis=3)
        drop = Dropout(0.1)(concat)

        x= UpSampling2D()(drop)
        x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
        x= UpSampling2D()(x)
        x = Conv2D(64, (3, 3), activation='relu',padding='same')(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)
        x = Dropout(0.1)(x)
        out = Conv2D(3, (3, 3), activation='relu',padding='same')(x)
              
        #Weighting red value more, so that the ball doesn't disappear
        def weightedMSE(y_true, y_pred):
            red = np.array(np.all(y_true.numpy()==(200/255,72/255,72/255),axis=-1),dtype=np.float32)*3+1
            mse = tf.math.reduce_mean(tf.math.pow(y_pred-y_true,4))
            
            return tf.multiply(mse, red)

        model = Model([observations_input,action_input],out)
        model.compile(optimizer=AdamW(weight_decay=1e-4), loss=weightedMSE,run_eagerly=True)
        return model

