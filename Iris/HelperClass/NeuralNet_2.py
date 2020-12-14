import numpy as np
import time
import math
import os
import sys
from pathlib import Path

from HelperClass.EnumDef_2 import *
from HelperClass.DataReader_2 import *
from HelperClass.LossFunction_2 import *
from HelperClass.ClassifierFunction_2 import *
from HelperClass.WeightBias_2 import *
from HelperClass.TrainingHistory_2 import *
from HelperClass.HyperParameters_2 import *
from HelperClass.ActivatorFunction_2 import *

class NeuralNet_2(object):
    def __init__(self,hp,model_name):
        self.hp=hp
        self.model_name=model_name
        self.subfolder=os.getcwd()+"\\"+self.__create_subfolder()
        self.wb1=WeightBias_2(self.hp.n_input,self.hp.n_hidden,self.hp.init_meth,self.hp.eta)
        self.wb1.InitializeWeights(self.subfolder,False)
        self.wb2=WeightBias_2(self.hp.n_hidden,self.hp.n_output,self.hp.init_meth,self.hp.eta)
        self.wb2.InitializeWeights(self.subfolder,False)
    def __create_subfolder(self):
        if self.model_name!=None:
            path=self.model_name.strip()
            path=path.rstrip("\\")
            isExists=os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path
    def __forward(self,X):
        self.Z1=np.dot(X,self.wb1.W)+self.wb1.B
        self.A1=Sigmoid().forward(self.Z1)
        self.Z2=np.dot(self.A1,self.wb2.W)+self.wb2.B
        self.A2=Softmax().forward(self.Z2)
        self.output=self.A2
        return self.A2
    def __backward(self,batch_x,batch_y):
        m=batch_x.shape[0]
        dZ2=self.A2-batch_y
        self.wb2.dW=np.dot(self.A1.T,dZ2)/m
        self.wb2.dB=np.sum(dZ2,axis=0,keepdims=True)/m
        dA1=np.dot(dZ2,self.wb2.W.T)
        dZ1,_=Sigmoid().backward(None,self.A1,dA1)
        self.wb1.dW=np.dot(batch_x.T,dZ1)/m
        self.wb1.dB=np.sum(dZ1,axis=0,keepdims=True)/m
    def __update(self):
        self.wb1.Update()
        self.wb2.Update()
    def __CalAccuracy(self,A,Y):
        assert(A.shape==Y.shape)
        m=A.shape[0]
        rA=np.argmax(A,axis=1)
        rY=np.argmax(Y,axis=1)
        R=(rA==rY)
        r=R.sum()/m
        return r
    def CheckErrorAndLoss(self,sdr,train_x,train_y,epoch,iteration):
        self.__forward(train_x)
        train_loss=self.loss_func.CheckLoss(self.output,train_y)
        train_accuracy=self.__CalAccuracy(self.output,train_y)
        vld_x,vld_y=sdr.GetVldSet()
        self.__forward(vld_x)
        vld_loss=self.loss_func.CheckLoss(self.output,vld_y)
        vld_accuracy=self.__CalAccuracy(self.output,vld_y)
        stop_flag=self.loss_trace.Add(epoch,iteration,train_loss,train_accuracy,vld_loss,vld_accuracy)
        if vld_loss<=self.hp.eps:
            stop_flag=True
        return stop_flag
    def train(self,sdr,checkpoint):
        self.loss_trace=TrainingHistory_2()
        self.loss_func=LossFunction_2()
        if self.hp.batch_size==-1:
            self.hp.batch_size=sdr.num_train
        max_iteration=math.ceil(sdr.num_train/self.hp.batch_size)
        iteration_cpt=int(max_iteration*checkpoint)
        stop_flag=False
        for epoch in range(self.hp.max_epoch):
            sdr.Shuffle()
            for iteration in range(max_iteration):
                X1,Y1=sdr.GetBatchTrainSamples(self.hp.batch_size,iteration)
                A2=self.__forward(X1)
                self.__backward(X1,Y1)
                self.__update()
                iteration_now=max_iteration*epoch+iteration
                if (iteration_now+1)%iteration_cpt==0:
                    stop_flag=self.CheckErrorAndLoss(sdr,X1,Y1,epoch,iteration_now)
                    if stop_flag:
                        break
            if stop_flag:
                break
        self.SaveResult()
    def Test(self,sdr):
        x,y=sdr.GetTestSet()
        self.__forward(x)
        accuracy_tst=self.__CalAccuracy(self.output,y)
        return accuracy_tst
    def SaveResult(self):
        self.wb1.SaveResultValue(self.subfolder,"wb1")
        self.wb2.SaveResultValue(self.subfolder,"wb2")
    def LoadResult(self):
        self.wb1.LoadResultValue(self.subfolder,"wb1")
        self.wb2.LoadResultValue(self.subfolder,"wb2")
    def ShowTrainingHistory(self):
        self.loss_trace.ShowLossHistory(self.hp)
    def GetTrainingHistory(self):
        return self.loss_trace
    def inference(self,X):
        return self.__forward(X)