import numpy as np
import sys
import math
from functools import reduce

class Cfc(object):
    def __init__(self,inputSize,outputNum=2,name=None,exname=""):
        self.shape=inputSize
        self.batch=inputSize[0]
        self.W=np.random.standard_normal((reduce(lambda x,y:x*y,self.shape[1:]),outputNum))/100
        self.B=np.random.standard_normal(outputNum)/100
        self.output=np.zeros((self.batch,outputNum))
        self.dW=np.zeros(self.W.shape)
        self.dB=np.zeros(self.B.shape)
        self.outputShape=self.output.shape

        self.type="Fc"
        self.input_name=f'{exname}y' if exname else f'{name}x'
        self.output_name=name + "y"
        self.input_size=[1]+list(inputSize)[1:]
        self.output_size=[1]+list(self.outputShape)[1:]

        self.weights_name=name+"w"
        self.bias_name=name+"b"
        self.weights_size=self.W.shape
        self.bias_size=self.B.shape

    def forward(self,image):
        image=np.reshape(image,[self.batch,-1])
        fcout=np.dot(image,self.weights)+self.bias
        self.output=fcout
        self.image=image
        return fcout

    def gradient(self,preError): 
        for i in range(self.batch):
            imagei=self.image[i][:,np.newaxis]
            preErrori=preError[i][:,np.newaxis].T
            self.dW+=np.dot(imagei,preErrori)
            self.dB+=np.reshape(preErrori,self.dB.shape)

        return np.reshape(np.dot(preError,self.W.T),self.shape)

    def backward(self,learningRate=0.001,weightsDecay=0.004):
        weights=(1-weightsDecay)*self.W
        bias=(1-weightsDecay)*self.B
        self.W-=learningRate*self.dW
        self.B-=learningRate*self.dB
        self.dW=np.zeros(self.W.shape)
        self.dB=np.zeros(self.B.shape)
