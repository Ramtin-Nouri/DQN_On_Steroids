import TF2_Keras_Template as template
import autoencoder,datamanager

batchsize = 16


net = autoencoder.NeuralNetwork()
model,epoch = net.getModel((160,208,6),(160,208,3)) #original size is 210 but that's not divisible by 8

dataGen = datamanager.DataGenerator("Breakout-v0",batchsize,debugMode=False)

#Get Loggers
logger = template.Logger("savedata/",model)
logger.setTestImages("data/test")
callbacks = logger.getCallbacks(period=20,predict=False) #predict set to False until its working 
#TODO:implement class inheriting Logger and overwriting getImgPredictions()

model.fit(dataGen.getGenerator(),
                steps_per_epoch=1000,
                epochs=1000,
                shuffle=True,
                initial_epoch=epoch,
                callbacks=callbacks)
