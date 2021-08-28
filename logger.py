
"""Logger for logging loss and predict example images.

to tensorboard and file.
"""
import numpy as np
import cv2
import TF2_Keras_Template as template

import tensorflow as tf
import os
import numpy as np
import cv2


class MultiInputLogger(template.Logger):
    """Logger for Multiple frames as input.

    Inherits from template.Logger and only overrides some functions.
    """

    def __init__(self, outputFolder, model):
        """init."""
        super().__init__(outputFolder, model)
        self.sequenceLength = 6

    def setActionSize(self, width, height):
        """Set models actions size."""
        self.action_width = width
        self.action_height = height

    def getImgPrediction(self):
        """Return predictions on test images."""
        outputs = []
        for i in range(self.sequenceLength):
            pred = self.model.predict(self.testImages[i][0])[0]
            outputs.append(pred)
        return outputs

    def predictAndSave2Tensorboard(self, fileWriter, epoch, name):
        """
        Predict test images and save to file and tensorboard.

        predict the images and draw side-by-side the
        target and prediction

        Arguments:
        ----------
        fileWriter: Tensorboard filewriter
        epoch: int
            current epoch
        name: str
            name of file to be saved
        """
        global sequenceLength
        both = [None]*2*self.sequenceLength
        predictions = self.getImgPrediction()
        for i in range(2*self.sequenceLength):
            if i % 2 == 0:
                both[i] = self.testImages[int(i/2)][1]
            else:
                both[i] = predictions[int(i/2)]
        stacked = self.stack(both)*255
        cv2.imwrite(name, stacked)
        conv = np.array([np.clip(stacked, 0, 255)], dtype=np.uint8)
        with fileWriter.as_default():
            tf.summary.image("Test Image", conv, step=epoch)

    def setTestImages(self, testImageFolder):
        """
        Set test images.

        Read the action from the image name and fill an array of shape
        (action_width,action_height) with it.
        Read the images and use 4 stacked images and the action image as input
        and next image as output.
        """
        imgpaths = os.listdir(testImageFolder)
        imgpaths.sort()
        # each sequence has 5 frames (4 inputs + 1 output)
        imgpaths = imgpaths[:5*self.sequenceLength]

        imgs = []
        for img in imgpaths:
            imgs.append(cv2.imread(F"{testImageFolder}/{img}")/255)

        for sequence in range(self.sequenceLength):
            in_ = []
            for _ in range(4):
                in_.append(imgs.pop(0))
            stack = np.array([np.dstack(in_)])

            actionInt = int(
                imgpaths[5*sequence].split("action-")[1].split(".png")[0])
            action = np.full(
                (1, self.action_width, self.action_height, 1), actionInt)

            self.testImages.append(([stack, action], imgs.pop(0)))
