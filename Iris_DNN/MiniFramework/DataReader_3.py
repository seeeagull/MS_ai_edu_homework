import numpy as np
import pandas as pd
import csv
from pathlib import Path

from MiniFramework.EnumDef_3 import *

class DataReader_3(object):
    def __init__(self,train_file=None,test_file=None):
        self.train_file_name=train_file
        self.test_file_name=test_file
        self.num_train=0
        self.num_tst=0
        self.num_vld=0
        self.num_feat=0
        self.num_cat=0
        self.XTrainRaw=None
        self.YTrainRaw=None
        self.XTestRaw=None
        self.YTestRaw=None
        self.XTrain=None
        self.YTrain=None
        self.XVld=None
        self.YVld=None
        self.XTest=None
        self.YTest=None

    def ReadData(self):
        train_file=Path(self.train_file_name)
        if train_file.exists():
            data=np.load(self.train_file_name)
            self.XTrainRaw=data["data"].astype('float32')
            self.YTrainRaw=data["label"].astype('int32')
            assert(self.XTrainRaw.shape[0]==self.YTrainRaw.shape[0])
            self.num_train=self.XTrainRaw.shape[0]
            self.num_feat=self.XTrainRaw.shape[1]
            self.num_cat=len(np.unique(self.YTrainRaw))
            self.XTrain,self.YTrain=self.XTrainRaw,self.YTrainRaw
        else:
            raise Exception("Cannot find train file!")
        test_file=Path(self.test_file_name)
        if test_file.exists():
            data=np.load(self.test_file_name)
            self.XTestRaw=data["data"].astype('float32')
            self.YTestRaw=data["label"].astype('int32')
            assert(self.XTestRaw.shape[0]==self.YTestRaw.shape[0])
            self.num_tst=self.XTestRaw.shape[0]
            self.XTest,self.YTest=self.XTestRaw,self.YTestRaw
            self.XVld,self.YVld=self.XTestRaw,self.YTestRaw
        else:
            raise Exception("Cannot find test file!")
    
    def ReadDataIris(self):
        data=pd.read_csv("iris.csv",dtype={"sepal length":np.float64,"sepal width":np.float64,"petal length":np.float64,"petal width":np.float64,"class":object})
        x1,x2,x3,x4,y=[],[],[],[],[]
        for i in range(150):
            x1.append([data.values[i][0]])
            x2.append([data.values[i][1]])
            x3.append([data.values[i][2]])
            x4.append([data.values[i][3]])
            if data.values[i][4]=="Iris-setosa":
                y.append(0)
            elif data.values[i][4]=="Iris-versicolor":
                y.append(1)
            else:
                y.append(2)
        self.XTrainRaw=np.hstack((np.array(x1),np.array(x2),np.array(x3),np.array(x4)))
        self.YTrainRaw=np.array(y).reshape(150,1)
        self.num_train=150
        self.num_feat=4
        self.num_cat=3

    def NormalizeX(self):
        if self.XTestRaw is not None:
            X_merge=np.vstack((self.XTrainRaw,self.XTestRaw))
        else:
            X_merge=self.XTrainRaw
        self.X_norm=np.zeros((2,self.num_feat)).astype('float32')
        X_new=np.zeros_like(X_merge).astype('float32')
        for i in range(self.num_feat):
            x=X_merge[:,i]
            mx,mn=np.max(x),np.min(x)
            self.X_norm[0,i],self.X_norm[1,i]=mn,mx-mn
            x=(x-self.X_norm[0,i])/self.X_norm[1,i]
            X_new[:,i]=x
        self.XTrain=X_new[0:self.num_train,:]
        self.XTest=X_new[self.num_train:,:]

    def NormalizeY(self,net_tp,base=0):
        if net_tp==NetType.Fitting:
            if self.YTestRaw is not None:
                Y_merge=np.vstack((self.YTrainRaw,self.YTestRaw))
            else:
                Y_merge=self.YTrainRaw
            Y_new=np.zeros_like(Y_merge).astype('float32')
            self.Y_norm=np.zeros((2,1)).astype('float32')
            y=Y_merge[:,0]
            mx,mn=np.max(y),np.min(y)
            self.Y_norm[0,0],self.Y_norm[1,0]=mn,mx-mn
            y=(y-self.Y_norm[0,0])/self.Y_norm[1,0]
            Y_new[:,0]=y
            self.YTrain=Y_new[0:self.num_train,:]
            self.YTest=Y_new[self.num_train:,:]
        elif net_tp==NetType.BinaryClassifier:
            self.YTrain=self.__ToZeroOne(self.YTrainRaw)
            if self.YTestRaw is not None:
                self.YTest=self.__ToZeroOne(self.YTestRaw)
        else:
            self.YTrain=self.__ToOneHot(self.YTrainRaw,base)
            if self.YTestRaw is not None:
                self.YTest=self.__ToOneHot(self.YTestRaw,base)

    def DeNormalizeY(self,predict_data):
        real_val=predict_data*self.Y_norm[1,0]+self.Y_norm[0,0]
        return real_val

    #if use tanh function, need to set n_val=-1
    def __ToZeroOne(self,Y,p_label=1,n_label=0,p_val=1,n_val=0):
        Y_new=np.zeros_like(Y).astype('float32')
        m=Y.shape[0]
        for i in range(m):
            if Y[i,0]==n_label:
                Y_new[i,0]=n_val
            else:
                Y_new[i,0]=p_val
        return Y_new

    def __ToOneHot(self,Y,base=0):
        m=Y.shape[0]
        Y_new=np.zeros((m,self.num_cat)).astype('float32')
        for i in range(m):
            x=int(Y[i,0])
            Y_new[i,x-base]=1
        return Y_new

    def NormalizePredicateData(self,X_predicate):
        X_new=np.zeros(X_predicate.shape).astype('float32')
        n_feat=X_predicate.shape[0]
        for i in range(n_feat):
            X_new[i,:]=(X_predicate[i,:]-X_norm[0,i])/self.X_norm[1,i]
        return X_new

    def GenerateValidationSet(self,k=10):
        self.num_vld=int(self.num_train/k)
        self.num_train-=self.num_vld
        self.XVld=self.XTrain[0:self.num_vld]
        self.YVld=self.YTrain[0:self.num_vld]
        self.XTrain=self.XTrain[self.num_vld:]
        self.YTrain=self.YTrain[self.num_vld:]

    def GenerateTestSet(self,k=10):
        self.num_tst=int(self.num_train/k)
        self.num_train-=self.num_tst
        self.XTest=self.XTrain[0:self.num_tst]
        self.YTest=self.YTrain[0:self.num_tst]
        self.XTrain=self.XTrain[self.num_tst:]
        self.YTrain=self.YTrain[self.num_tst:]

    def GetBatchTrainSamples(self,batch_size,iteration):
        start=batch_size*iteration
        end=start+batch_size
        XBatch=self.XTrain[start:end,:]
        YBatch=self.YTrain[start:end,:]
        return XBatch,YBatch

    def GetTrainSet(self):
        return self.XTrain,self.YTrain

    def GetTestSet(self):
        return self.XTest,self.YTest

    def GetVldSet(self):
        return self.XVld,self.YVld

    def Shuffle(self):
        seed=np.random.randint(1,100)
        np.random.seed(seed)
        XP=np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP=np.random.permutation(self.YTrain)
        self.XTrain=XP
        self.YTrain=YP
