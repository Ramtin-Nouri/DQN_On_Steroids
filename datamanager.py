import gym
from threading import Thread
from queue import Queue
import numpy as np,cv2

def getAction(obs):
    def getRed(img):
        indices = np.where(np.all(img==(200/255,72/255,72/255),axis=-1))
        return list(zip(indices[0],indices[1]))
    NOTHING = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3

    try:
        paddlePos = getRed(obs[188:-17,8:-8])[0]
    except: 
        print(obs[188:-17,8:-8])
    ballPos = getRed(obs[68:-21,8:-8])
    if len(ballPos) < 1:
        action = FIRE
    elif paddlePos[1]+17<ballPos[0][1]:
        action = RIGHT
    elif paddlePos[1]>ballPos[0][1]:
        action = LEFT
    else:
        action = NOTHING
    return action

class DataGeneratorDynamics():
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
        action = 1
        while self.shouldRun:
            obs,_,done,_ = self.env.step(action)
            if done:
                self.env.reset()
                continue
            if self.debugMode:
                self.env.render()
            
            obs = obs/255
            obs = obs[2:]
            self.data.put((action,obs))
            action = getAction(obs)


    def _generator(self):
        actionShape = (26,20,1)
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
        
class DataGeneratorState():
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
        action = 1
        while self.shouldRun:
            obs,_,done,_ = self.env.step(action)
            if done:
                self.env.reset()
                continue
            if self.debugMode:
                self.env.render()
            
            obs = obs/255
            obs = obs[2:]
            self.data.put(obs)
            action = getAction(obs)

    def _generator(self):
        while True:
            batchIn=[]
            for _ in range(self.batchsize):
                datapoint = self.data.get()
                batchIn.append(datapoint)
            yield (np.array(batchIn),np.array(batchIn))

    def getGenerator(self):
        return self._generator()

    def getShape():
        pass#TODO
