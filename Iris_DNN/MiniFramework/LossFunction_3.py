from MiniFramework.EnumDef_3 import *

class LossFunction_3(object):
    def __init__(self,net_type):
        self.net_type=net_type

    def CheckLoss(self,A,Y):
        m=Y.shape[0]
        if self.net_type==NetType.Fitting:
            loss,acc=self.MSE(A,Y,m)
        elif self.net_type==NetType.BinaryClassifier:
            loss,acc=self.CE2(A,Y,m)
        elif self.net_type==NetType.MultipleClassifier:
            loss,acc=self.CE3(A,Y,m)
        if loss.ndim==1: #?
            return loss[0],acc[0]
        return loss,acc

    def MSE(self,A,Y,m):
        p1=A-Y
        LOSS=np.multiply(p1,p1)
        loss=np.sum(LOSS)/m/2
        var=np.var(Y)
        mse=np.sum(LOSS)/m
        r2=1-mse/var
        return loss,r2

    def CE2(self,A,Y,m):
        p1=np.multiply(np.log(A),Y)
        p2=np.multiply(np.log(1-A),1-Y)
        LOSS=np.sum(-(p1+p2))
        loss=LOSS/m
        B=np.round(A)
        R=(B==Y)
        r=np.sum(R)/m
        return loss,r

    def CE3(self,A,Y,m):
        p1=np.multiply(np.log(A+1e-7),Y)
        LOSS=np.sum(-p1)
        loss=LOSS/m
        rA=np.argmax(A,axis=1)
        rY=np.argmax(Y,axis=1)
        R=(rA==rY)
        r=np.sum(R)/m
        return loss,r