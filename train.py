import TF2_Keras_Template as template
import autoencoder,datamanager

batchsize = 16


net = autoencoder.NeuralNetwork()
model,epoch = net.getModel((None,None,3),(None,None,3)) #(None,None) basically means we don't care. Because it is a CNN the output shape will be determined by the architecture

dataGen = datamanager.DataGenerator("Breakout-v0",batchsize,debugMode=True)

#Get Loggers
logger = template.Logger("savedata/",model)
logger.setTestImages("data/test")
callbacks = logger.getCallbacks(period=20)

model.fit(dataGen.getGenerator(),
                steps_per_epoch=1000,
                epochs=1000,
                shuffle=True,
                initial_epoch=epoch,
                callbacks=callbacks)
