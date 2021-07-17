from nets import DynamicsNetwork
import datamanager,logger

batchsize = 8


net = DynamicsNetwork.NeuralNetwork()
model,epoch = net.getModel((208,160,12),(208,160,3)) #original size is 210 but that's not divisible by 4

dataGen = datamanager.DataGeneratorDynamics("Breakout-v4",batchsize,debugMode=False,actionShape=(52,40),nFramesIn=4)

#Get Loggers
logger = logger.MultiInputLogger("savedata/dynamics/",model)
logger.setActionSize(52,40)
logger.setTestImages("data/test")
callbacks = logger.getCallbacks(period=20)

model.fit(dataGen.getGenerator(),
                steps_per_epoch=100,
                epochs=1000,
                shuffle=True,
                initial_epoch=epoch,
                callbacks=callbacks)
