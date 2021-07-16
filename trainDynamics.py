import DynamicsNetwork,datamanager,logger

batchsize = 16


net = DynamicsNetwork.NeuralNetwork()
model,epoch = net.getModel((208,160,6),(208,160,3)) #original size is 210 but that's not divisible by 8

dataGen = datamanager.DataGeneratorDynamics("Breakout-v0",batchsize,debugMode=False)

#Get Loggers
logger = logger.DoubleInputLogger("savedata/dynamics/",model)
logger.setTestImages("data/test")
callbacks = logger.getCallbacks(period=20)

model.fit(dataGen.getGenerator(),
                steps_per_epoch=1000,
                epochs=1000,
                shuffle=True,
                initial_epoch=epoch,
                callbacks=callbacks)
