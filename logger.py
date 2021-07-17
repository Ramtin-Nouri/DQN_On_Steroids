import numpy as np,cv2
import TF2_Keras_Template as template
 
import tensorflow as tf
import os,numpy as np,cv2

class DoubleInputLogger(template.Logger):
    def setActionSize(self,width,height):
        self.action_width = width
        self.action_height = height
    def getImgPrediction(self):
        outputs =[]
        for i in range(len(self.testImages)):
            if i%2==1:continue
            img=self.testImages[i][2:]
            img2=self.testImages[i+1][2:]
            both = np.array([np.dstack([img,img2])])
            in_ = [ both , np.zeros((1,self.action_width,self.action_height,1)) ]
            pred = self.model.predict(in_)[0]
            outputs.append(pred)
        return outputs

    def predictAndSave2Tensorboard(self,fileWriter,epoch,name):
        predictions = self.getImgPrediction()
        stacked = self.stack(predictions)
        cv2.imwrite(name,stacked)
        conv = np.array([np.clip(stacked,0,255)],dtype=np.uint8)
        with fileWriter.as_default():
            tf.summary.image("Test Image", conv, step=epoch)

    
    def setTestImages(self,testImageFolder):
        imgpaths = os.listdir(testImageFolder)
        #Pray they are actually images
        for img in imgpaths:
            self.testImages.append(cv2.imread(F"{testImageFolder}/{img}"))

class ClippedLogger(template.Logger):

    def setTestImages(self,testImageFolder):
        imgpaths = os.listdir(testImageFolder)[:8]
        #Pray they are actually images
        for img in imgpaths:
            self.testImages.append(cv2.imread(F"{testImageFolder}/{img}")[2:])