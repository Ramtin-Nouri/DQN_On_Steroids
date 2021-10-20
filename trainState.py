"""OpenAI Gym Atari Autoencoder.

Intended for use with trainAgent.

Trains a model to output the input.

Uses OpenAiGym's BreakoutNoFrameskip-v4 for data.
Training data is sampled from the environment continiously,
but validation data is prerecorded.

Tensorboard is used to log loss, etc. and also example predictions.
"""
from nets import StateNetwork
import datamanager
import logger
import TF2_Keras_Template as template
import tensorflow as tf

batchsize = 4


net = StateNetwork.NeuralNetwork()
# original size is 210 but that's not divisible by 8
model, epoch = net.getModel((208, 160, 3), (208, 160, 3))

dataGen = datamanager.DataGeneratorState(
    "BreakoutNoFrameskip-v4", batchsize, debugMode=False)
valData = datamanager.ValidationDataState("data/test")


# Get Loggers
logger = template.Logger("savedata/state/", model)
logger.setTestImages("data/test")
callbacks = logger.getCallbacks(period=5)
callbacks.append(tf.keras.callbacks.EarlyStopping(
    patience=50, verbose=True, mode="min"))


model.fit(dataGen.getGenerator(),
          steps_per_epoch=100,
          epochs=1000,
          shuffle=True,
          initial_epoch=epoch,
          callbacks=callbacks,
          validation_data=(valData.getX(), valData.getY()),
          validation_batch_size=valData.getBatchsize(),
          validation_steps=valData.getSteps())

next(dataGen.getGenerator())  # Just in case it is waiting to add to queue
