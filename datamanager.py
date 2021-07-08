import gym
from threading import Thread
from queue import Queue
import numpy as np,cv2

class DataGenerator():
    #TODO:ADD COMMENTS
    
    def __init__(self,envname,batchsize,queueSize=500,debugMode=False):
        self.batchsize=batchsize
        self.QUEUESIZE=queueSize
        self.data = Queue(maxsize=self.QUEUESIZE)
        self.env = gym.make(envname)
        self.shouldRun = True
        self.thread = Thread(target=self.gatherData)
        self.debugMode = debugMode
        self.thread.start()
    
    def gatherData(self):
        self.env.reset()
        while self.shouldRun:
            action = self.env.action_space.sample()
            obs,_,done,_ = self.env.step(action)
            if done:
                self.env.reset()
                continue
            if self.debugMode:
                self.env.render()
            
            obs = obs/255
            obs = cv2.resize(obs,(208,160)) #TODO: remove magicnumbers
            self.data.put((action,obs/255))

    def _generator(self):
        actionShape = (20,26,1)
        while True:
            batchIn1=[]
            batchIn2=[]
            batchOut=[]
            datapoint1 = self.data.get()
            datapoint2 = self.data.get()
            for _ in range(self.batchsize):
                datapoint3 = self.data.get()
                observations = np.dstack([datapoint1[1],datapoint2[1]]) #last 2 obs
                action = np.full(actionShape,datapoint3[0])
                batchIn1.append(observations)
                batchIn2.append(action)
                batchOut.append(np.array(datapoint3[1]))#predicted obs
                datapoint1=datapoint2
                datapoint2=datapoint3
            yield ([np.array(batchIn1),np.array(batchIn2)],np.array(batchOut))

    def getGenerator(self):
        return self._generator()

    def getShape():
        pass#TODO