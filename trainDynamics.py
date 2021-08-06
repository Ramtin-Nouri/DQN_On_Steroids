from nets import DynamicsNetwork
import datamanager,logger
import tensorflow as tf

batchsize = 8


net = DynamicsNetwork.NeuralNetwork()
model,epoch = net.getModel((208,160,12),(208,160,3)) #original size is 210 but that's not divisible by 4

dataGen = datamanager.DataGeneratorDynamics("Breakout-v4",batchsize,debugMode=False,actionShape=(26,20),nFramesIn=4)
valData = datamanager.ValidationDataDynamics("data/test",actionShape=(26,20),nFramesIn=4)

#Get Loggers
logger = logger.MultiInputLogger("savedata/dynamics/",model)
logger.setActionSize(26,20)
logger.setTestImages("data/test")
callbacks = logger.getCallbacks(period=1)

callbacks.append(tf.keras.callbacks.EarlyStopping(patience=5,verbose=True,mode="min"))

model.fit(dataGen.getGenerator(),
                steps_per_epoch=100,
                epochs=1000,
                shuffle=True,
                initial_epoch=epoch,
                callbacks=callbacks,
                validation_data = (valData.getX(),valData.getY()),
                validation_batch_size=valData.getBatchsize(),
                validation_steps=valData.getSteps())

dataGen.shouldRun = False
next(dataGen.getGenerator()) #Just in case it is waiting to add to queue