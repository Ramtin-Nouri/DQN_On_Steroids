"""Data for training and loading of trainState and trainDynamics.

trainState gets data in an autoencoder fashion where input
and label are the same.
A datapoint is only a single observation as input and output.

trainDynamics gets 4 observations and an action as input
and the next observation as label.
The action is filled to an 2D array because the network works with 2D data.

Training data is generated continously with actions sampled
from an simple hardcoded computer vision agent
deciding solely on the position of the ball wrt. to the paddle.
I use this rather than random actions such that there is less bias towards
earlier game states.

Validation data is prerecorded and only read from file.
"""
import gym
from threading import Thread
from queue import Queue
import numpy as np
import cv2
import os


def getAction(obs):
    """
    Harcoded Computer Vision Agent.

    Instead of using a random sample
    we sample from this naiv agent that
    gets the ball and paddle position through their pixel values.

    Returns:
    -------
    action
        action to take based on only the paddle position
        wrt. to the ball position.
    """

    def getRed(img):
        """Return all 'red' pixel in specified img as list of (x,y) pairs."""
        indices = np.where(np.all(img == (200/255, 72/255, 72/255), axis=-1))
        return list(zip(indices[0], indices[1]))

    # All the possible actions (can't be bothered to make this an enum)
    NOTHING = 0
    FIRE = 1
    RIGHT = 2
    LEFT = 3

    # Get paddle and ball position
    paddlePos = getRed(obs[188:-17, 8:-8])[0]
    ballPos = getRed(obs[68:-21, 8:-8])

    if len(ballPos) < 1:  # If no ball:shoot
        action = FIRE
    elif paddlePos[1]+17 < ballPos[0][1]:  # If ball right to paddle:move right
        action = RIGHT
    elif paddlePos[1] > ballPos[0][1]:  # If ball left to paddle:move left
        action = LEFT
    else:  # ball above paddle: do nothing
        action = NOTHING

    return action


class DataGenerator():
    """
    Creates Data in a thread for keras to grab.

    Attributes
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

    def __init__(self, envname, batchsize, queueSize=500, debugMode=False):
        """Set fields and start thread."""
        self.batchsize = batchsize
        self.QUEUESIZE = queueSize
        self.data = Queue(maxsize=self.QUEUESIZE)
        self.env = gym.make(envname)
        self.env.seed(111)
        self.shouldRun = True
        self.thread = Thread(target=self.gatherData)
        self.debugMode = debugMode
        self.thread.start()

    def getGenerator(self):
        """Return generator.

        Returns
        -------
        _generator: func
            function that yields the datapoints
        """
        return self._generator()

    def gatherData(self):
        """
        Continously fill Queue with new data.

        in a loop:
            - take action according to getAction()
            - put observation and action took into data queue
        """
        self.env.reset()
        action = 1
        while self.shouldRun:
            obs, _, done, _ = self.env.step(action)
            if done:
                self.env.reset()
                continue
            if self.debugMode:
                self.env.render()

            obs = obs/255
            obs = obs[2:]

            self.data.put((action, obs))
            action = getAction(obs)


class DataGeneratorDynamics(DataGenerator):
    """
    Implementation of DataGenerator.

    Arguments
    ---------
    (See DataGenerator)
    actionShape: tuple
        shape of the layer the action shall be concatinated to
        aka. shape the action should be filled to
    stackedObservationLength: int
        number of frames to use as input of net
    """

    def __init__(self, envname, batchsize, actionShape, queueSize=500, debugMode=False, nFramesIn=4):
        """super call + set its fields."""
        super().__init__(envname, batchsize, queueSize, debugMode)
        self.actionShape = actionShape
        self.stackedObservationLength = nFramesIn

    def _generator(self):
        """
        Yield batches.

        Yields:
        -------
        Batch : Tuple of (input,label)
            input :(current observation , action)
            label: next observation
        """
        while True:
            batchIn1 = []
            batchIn2 = []
            batchOut = []

            for _ in range(self.batchsize):
                observationFrames = []
                for _ in range(self.stackedObservationLength):
                    (_, obs) = self.data.get()
                    observationFrames.append(obs)

                observations = np.dstack(observationFrames)
                batchIn1.append(observations)

                nextFrame = self.data.get()
                action = np.full(self.actionShape, nextFrame[0])
                batchIn2.append(action)
                batchOut.append(np.array(nextFrame[1]))  # predicted obs

                # Throw away some, so that we get more different data
                for _ in range(20):
                    self.data.get()
            yield ([np.array(batchIn1), np.array(batchIn2)], np.array(batchOut))


class DataGeneratorState(DataGenerator):
    """
    Implementation of DataGenerator.

    Attributes
    ---------
    (See DataGenerator)
    """

    def _generator(self):
        """
        Yield batches.

        Yields:
        -------
        Batch : Tuple of (input,output)
            input: current observation
            output: current observation
            (Yes it's the same it shall be used in an autoencoder fashion)
        """

        while True:
            batchIn = []
            batchOut = []
            for _ in range(self.batchsize):
                _, observation = self.data.get()
                noiseAmount = 0.1
                # add noise and brightness
                augmented = observation + \
                    np.random.uniform(-noiseAmount,
                                      noiseAmount, observation.shape)
                batchIn.append(augmented)
                batchOut.append(observation)
            yield (np.array(batchIn), np.array(batchOut))


class ValidationDataDynamics():
    """Validation data loader for multi frame input model.

    Attributes:
    x: list
        inputs
    y: list
        targets
    steps:
        number of datapoints
    """

    def __init__(self, validationFolder, actionShape, nFramesIn=4):
        """Load all images in folder.

        Extract the action from the file name and
        fill whole image of actionShape with it
        Add 4 Appended images with action'image' to x (Stacked Observation)
        Add the next image to y (prediction)

        Args:
            validationFolder : str
                path to folder with validation images
            actionShape: tuple of int
            nFramesIn: int
                Number of frames for stacked observation.
                Defaults to 4.
        """
        self.y = []

        imgpaths = os.listdir(validationFolder)
        imgpaths.sort()

        imgs = []
        for img in imgpaths:
            imgs.append(cv2.imread(F"{validationFolder}/{img}")/255)

        counter = 0
        stacks = []
        actions = []
        while len(imgs) > 0:

            in_ = []
            for _ in range(nFramesIn):
                in_.append(imgs.pop(0))
            stacks.append(np.dstack(in_))

            actionInt = int(
                imgpaths[5*counter].split("action-")[1].split(".png")[0])
            actions.append(
                np.full((actionShape[0], actionShape[1], 1), actionInt))

            self.y.append(imgs.pop(0))

            counter += 1

        print(np.array(stacks).shape, np.array(actions).shape)
        self.x = [np.array(stacks), np.array(actions)]

        self.steps = len(self.y)

    def getX(self):
        """Return x."""
        return self.x

    def getY(self):
        """Return y."""
        return np.array(self.y)

    def getSteps(self):
        """Return number of datapoints, used for steps per epoch."""
        return self.steps

    def getBatchsize(self):
        """Return batchsize."""
        return 1


class ValidationDataState():
    """Validation data loader for single frame input model.
    
    Attributes:
    x: list
        inputs
    y: list
        same as x
    steps:
        number of images"""

    def __init__(self, validationFolder):
        """Load all images from validation folder.

        Append to x and y"""
        self.y = []

        imgpaths = os.listdir(validationFolder)
        imgpaths.sort()

        imgs = []
        for img in imgpaths:
            imgs.append(cv2.imread(F"{validationFolder}/{img}")/255)

        self.x = [np.array(imgs)]
        self.y = self.x
        self.steps = len(self.y)

    def getX(self):
        """Return x."""
        return self.x

    def getY(self):
        """Return y."""
        return self.y

    def getSteps(self):
        """Return number of images, used for steps per epoch."""
        return self.steps

    def getBatchsize(self):
        """Retun batchsize."""
        return 1
