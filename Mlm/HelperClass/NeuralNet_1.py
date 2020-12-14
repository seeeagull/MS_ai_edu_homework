import numpy as np
import math
import matplotlib.pyplot as plt

class NeuralNet_1(object):
    def __init__(self,hp):
        self.hp=hp
        self.W=np.zeros((self.hp.input_size,self.hp.output_size))
        self.B=np.zeros((1,self.hp.output_size))
    def __forwardBatch(self,x_batch):
        Z=np.dot(x_batch,self.W)+self.B
        return Z
    def __backwardBatch(self,x_batch,y_batch,z_batch):
        dZ=z_batch-y_batch
        m=x_batch.shape[0]
        dW=np.dot(x_batch.T,dZ)/m
        dB=dZ.sum(axis=0,keepdims=True)/m
        return dW,dB
    def __update(self,dW,dB):
        self.W=self.W-self.hp.eta*dW
        self.B=self.B-self.hp.eta*dB
    def __checkLoss(self,dataReader):
        X,Y=dataReader.GetWholeTrainSamples()
        Z=self.__forwardBatch(X)
        LOSS=(Z-Y)**2
        m=X.shape[0]
        loss=LOSS.sum(axis=0)/m/2
        return loss
    def train(self,dataReader,checkpoint=0.1):
        if self.hp.batch_size==-1:
            self.hp.batch_size=dataReader.num_train
        max_iteration=int(dataReader.num_train/self.hp.batch_size)
        checkpoint_iteration=math.ceil(checkpoint*max_iteration)
        loss=100
        for epoch in range(self.hp.max_epoch):
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                X,Y=dataReader.GetBatchTrainSamples(self.hp.batch_size,iteration)
                Z=self.__forwardBatch(X)
                dW,dB=self.__backwardBatch(X,Y,Z)
                self.__update(dW,dB)
                now_iteration=max_iteration*epoch+iteration
                if now_iteration%checkpoint_iteration==0:
                    loss=self.__checkLoss(dataReader)
                    if loss<self.hp.eps:
                        break
            if loss<self.hp.eps:
                break
    def inference(self,X):
        return self.__forwardBatch(X)
