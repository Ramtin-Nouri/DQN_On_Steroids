import gym
from threading import Thread
from queue import Queue

class DataGenerator():
    #TODO:ADD COMMENTS
    
    def __init__(self,envname,batchsize,queueSize=500,debugMode=False):
        self.batchsize=batchsize
        self.QUEUESIZE=queueSize
        self.data = Queue(maxsize=self.QUEUESIZE)
        self.env = gym.make(envname)
        self.thread = Thread(target=self.gatherData)
    
    def gatherData(self):
        self.env.reset()
        while shouldRun:
            action = self.env.action_space.sample()
            obs,_,done,_ = self.env.step(action)
            if done:
                self.env.reset()
                continue
            if debugMode:
                env.render()
            
            self.data.put((action,obs/255))

    def _generator(self):
        while True:
            batchIn=[]
            batchOut=[]
            datapoint1 = self.data.get()
            datapoint2 = self.data.get()
            for _ in range(self.batchsize):
                datapoint3 = self.data.get()
                input1 = (datapoint1[1],datapoint2[1]) #last 2 obs
                batchIn.append(inputs1,datapoint3[0]) #action
                batchout.append([[datapoint3[1]]])#predicted obs
                datapoint1=datapoint2
                datapoint2=datapoint3
            yield (batchIn,batchOut)

    def getGenerator(self):
        return self._generator()