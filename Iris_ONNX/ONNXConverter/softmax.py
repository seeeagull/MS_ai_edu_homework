import numpy as np

class Csoftmax(object):
    def __init__(self,inputSize,name=None,exname=""):
        self.shape=inputSize
        self.batchSize=inputSize[0]
        self.error=np.zeros(inputSize)
        self.type="Softmax"
        self.input_name=f'{exname}y' if exname else f'{name}x'
        self.input_size=[1]+list(inputSize)[1:]
        self.output_name=name+"y"
        self.output_size=[1]+list(inputSize)[1:]
    def calLoss(self,labels,perdiction):
        self.softmax=np.zeros(self.shape)
        self.label=labels
        self.loss=0
        for i in range(self.batchSize):
            perdiction[i]-=np.max(perdiction[i])
            perdiction[i]=np.exp(perdiction[i])
            self.softmax[i]=perdiction[i]/np.sum(perdiction[i])
            self.loss=self.loss-np.log(self.softmax[i,labels[i]])
        return self.loss
    def gradient(self):
        self.error=self.softmax.copy()
        for i in range(self.batchSize):
            self.error[i,self.label[i]]-=1;
        return self.error