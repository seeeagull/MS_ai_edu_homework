import numpy as np

class LossFunction_2(object):
    def CheckLoss(self,A,Y):
        m=Y.shape[0]
        p1=np.multiply(Y,np.log(A))
        LOSS=-p1
        loss=np.sum(LOSS)/m
        return loss