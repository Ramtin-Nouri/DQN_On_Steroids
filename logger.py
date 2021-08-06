import numpy as np,cv2
import TF2_Keras_Template as template
 
import tensorflow as tf
import os,numpy as np,cv2

class MultiInputLogger(template.Logger):
    def __init__(self,outputFolder,model):
        super().__init__(outputFolder,model)
        self.sequenceLength = 6

    def setActionSize(self,width,height):
        self.action_width = width
        self.action_height = height

    def getImgPrediction(self):
        outputs =[]
        for i in range(self.sequenceLength):
            pred = self.model.predict( self.testImages[i][0])[0]
            outputs.append(pred)
        return outputs

    def predictAndSave2Tensorboard(self,fileWriter,epoch,name):
        global sequenceLength
        both=[None]*2*self.sequenceLength
        predictions = self.getImgPrediction()
        for i in range(2*self.sequenceLength):
            if i%2==0:
                both[i]=self.testImages[int(i/2)][1]
            else:
                both[i]=predictions[int(i/2)]
        stacked = self.stack(both)
        cv2.imwrite(name,stacked)
        conv = np.array([np.clip(stacked,0,255)],dtype=np.uint8)
        with fileWriter.as_default():
            tf.summary.image("Test Image", conv, step=epoch)

    
    def setTestImages(self,testImageFolder):
        imgpaths = os.listdir(testImageFolder)
        imgpaths.sort()
        imgpaths = imgpaths[:5*self.sequenceLength] #each sequence has 5 frames (4 inputs + 1 output)
        
        imgs = []
        for img in imgpaths:
            imgs.append(cv2.imread(F"{testImageFolder}/{img}"))
        
        for sequence in range(self.sequenceLength):
            in_ = []
            for _ in range(4):
                in_.append(imgs.pop(0))
            stack = np.array([np.dstack(in_)])

            actionInt = int(imgpaths[5*sequence].split("action-")[1].split(".png")[0])
            action = np.full((1,self.action_width,self.action_height,1),actionInt)

            self.testImages.append( ([stack,action],imgs.pop(0)) )
