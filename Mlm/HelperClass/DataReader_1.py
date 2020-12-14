import numpy as np
import pandas as pd

class DataReader_1(object):
    def __init__(self):
        self.num_train=100
        self.XTrain=None
        self.YTrain=None
        self.XRaw=None
        self.YRaw=None
    def ReadData(self):
        data=pd.read_csv("mlm.csv")
        x,y=[],[]
        for i in range(1000):
            x.append([data.values[i][0],data.values[i][1]])
            y.append([data.values[i][2]])
        self.XRaw=np.array(x)
        self.YRaw=np.array(y)
    def NormalizeX(self):
        X_new=np.zeros_like(self.XRaw)
        num_feature=self.XRaw.shape[1]
        self.X_norm=np.zeros((num_feature,2))
        for i in range(num_feature):
            col_i=self.XRaw[:,i]
            max_value=np.max(col_i)
            min_value=np.min(col_i)
            self.X_norm[i,0]=min_value
            self.X_norm[i,1]=max_value-min_value
            new_col=(col_i-self.X_norm[i,0])/self.X_norm[i,1]
            X_new[:,i]=new_col
        self.XTrain=X_new
    def NormalizeY(self):
        self.Y_norm=np.zeros((1,2))
        max_value=np.max(self.YRaw)
        min_value=np.min(self.YRaw)
        self.Y_norm[0,0]=min_value
        self.Y_norm[0,1]=max_value-min_value
        self.YTrain=(self.YRaw-self.Y_norm[0,0])/self.Y_norm[0,1]
    def GetBatchTrainSamples(self,batch_size,iteration):
        start=batch_size*iteration
        end=start+batch_size
        batch_X=self.XTrain[start:end,:]
        batch_Y=self.YTrain[start:end,:]
        return batch_X,batch_Y
    def GetWholeTrainSamples(self):
        return self.XTrain,self.YTrain
    def GetWholeRawTrainSamples(self):
        return self.XRaw,self.YRaw
    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP
