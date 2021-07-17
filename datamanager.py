import gym
from threading import Thread
from queue import Queue
import numpy as np,cv2


def getAction(obs):
    """
        Harcoded Computer Vision Agent.
        Instead of using a random sample we sample from this naiv agent that gets the ball and paddle position through their pixel values.
        
        Returns 
        -------
        action
            action to take based on only the paddle position wrt. to the ball position.
    """

    def getRed(img):
        """
            Function returning all 'red' values in specified img as list of (x,y) pairs
        """
        indices = np.where(np.all(img==(200/255,72/255,72/255),axis=-1))
        return list(zip(indices[0],indices[1]))

    #All the possible actions (cant be bothered to make this an enum)
    NOTHING = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3

    #Get paddle and ball position
    paddlePos = getRed(obs[188:-17,8:-8])[0]
    ballPos = getRed(obs[68:-21,8:-8])

    if len(ballPos) < 1: #If no ball:shoot
        action = FIRE
    elif paddlePos[1]+17<ballPos[0][1]: #If ball right to paddle:move right
        action = RIGHT
    elif paddlePos[1]>ballPos[0][1]: #If ball left to paddle:move left
        action = LEFT
    else: #ball above paddle: do nothing
        action = NOTHING

    return action


class DataGenerator():
    
    """
        Creates Data in a thread for keras to grab
        Arguments
        ---------
        batchsize: int 
        QUEUESIZE: int
            max length Queue shall have
        data: Queue
        env: OpenAi Gym Environment
        shouldRun: bool
            whether the thread should run or quit and terminate
        thread: Thread
        debugMode: bool
            For now only: whether to render the environment

    """

    def __init__(self,envname,batchsize,queueSize=500,debugMode=False):
        """
            Set fields and start thread
        """

        self.batchsize=batchsize
        self.QUEUESIZE=queueSize
        self.data = Queue(maxsize=self.QUEUESIZE)
        self.env = gym.make(envname)
        self.shouldRun = True
        self.thread = Thread(target=self.gatherData)
        self.debugMode = debugMode
        self.thread.start()

    def getGenerator(self):
        """
            Returns
            -------
            _generator: func
                function that yields the datapoints
        """

        return self._generator()


    def gatherData(self):
        """
            in a loop:
                - take action according to getAction()
                - put observation and action took into data queue
        """

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


class DataGeneratorDynamics(DataGenerator):
    """
        Implementation of DataGenerator

        Arguments
        ---------
        (See DataGenerator)
        actionShape: tuple
            shape of the layer the action shall be concatinated to aka shape the action should be filled to
        stackedObservationLength: int
            number of frames to use as input of net
    """

    def __init__(self,envname,batchsize,actionShape,queueSize=500,debugMode=False,nFramesIn=4):
        """
            super call + set its fields
        """

        super().__init__(envname,batchsize,queueSize,debugMode)
        self.actionShape = actionShape
        self.stackedObservationLength = nFramesIn




    def _generator(self):
        """
            yields batches where each datapoint is :
                input :(current observation , action)
                label: next observation
        """

        while True:
            batchIn1=[]
            batchIn2=[]
            batchOut=[]
            observationFrames = []
            for _ in range(self.stackedObservationLength):
                (_,obs) = self.data.get()
                observationFrames.append(obs)

            for _ in range(self.batchsize):
                observations = np.dstack(observationFrames)
                batchIn1.append(observations)

                nextFrame = self.data.get()
                action = np.full(self.actionShape,nextFrame[0])
                batchIn2.append(action)
                batchOut.append(np.array(nextFrame[1]))#predicted obs

                observationFrames.pop(0)
                observationFrames.append(nextFrame[1])
            yield ([np.array(batchIn1),np.array(batchIn2)],np.array(batchOut))


        
class DataGeneratorState(DataGenerator):

    def _generator(self):
        """
            Yields batches where each datapoint is:
                input: current observation
                output: current observation
            Yes its the same it shall be used in an autoencoder fashion
        """

        while True:
            batchIn=[]
            for _ in range(self.batchsize):
                _,observation = self.data.get()
                batchIn.append(observation)
            yield (np.array(batchIn),np.array(batchIn))


