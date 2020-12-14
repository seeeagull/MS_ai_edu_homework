import numpy as np
import pandas as pd
import csv

from HelperClass.EnumDef_2 import *

class DataReader_2(object):
    def __init__(self):
        self.num_train=0
        self.num_tst=0
        self.num_vld=0
        self.num_feat=0
        self.num_cat=0
        self.XTrainRaw=None
        self.YTrainRaw=None
        self.XTrain=None
        self.YTrain=None
        self.XVld=None
        self.YVld=None
        self.XTest=None
        self.YTest=None
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
        X_merge=self.XTrainRaw
        self.X_norm=np.zeros((2,self.num_feat))
        X_new=np.zeros_like(X_merge)
        for i in range(self.num_feat):
            x=X_merge[:,i]
            mx,mn=np.max(x),np.min(x)
            self.X_norm[0,i],self.X_norm[1,i]=mn,mx-mn
            x=(x-self.X_norm[0,i])/self.X_norm[1,i]
            X_new[:,i]=x
        self.XTrain=X_new
    def NormalizeY(self,base=0):
        self.YTrain=self.__ToOneHot(self.YTrainRaw,base)
    def GenerateTestSet(self,k=10):
        self.num_tst=int(self.num_train/k)
        self.num_train-=self.num_tst
        self.XTest=self.XTrain[0:self.num_tst]
        self.YTest=self.YTrain[0:self.num_tst]
        self.XTrain=self.XTrain[self.num_tst:]
        self.YTrain=self.YTrain[self.num_tst:]
    def GenerateValidationSet(self,k=10):
        self.num_vld=self.num_tst
        self.num_train-=self.num_vld
        self.XVld=self.XTrain[0:self.num_vld]
        self.YVld=self.YTrain[0:self.num_vld]
        self.XTrain=self.XTrain[self.num_vld:]
        self.YTrain=self.YTrain[self.num_vld:]
    def __ToOneHot(self,Y,base=0):
        m=Y.shape[0]
        Y_new=np.zeros((m,self.num_cat))
        for i in range(m):
            x=int(Y[i,0])
            Y_new[i,x-base]=1
        return Y_new
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
        XNew=np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YNew=np.random.permutation(self.YTrain)
        self.XTrain=XNew
        self.YTrain=YNew