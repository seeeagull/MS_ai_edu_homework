import numpy as np

class CActivator(object):
    def forward(self,Z):
        pass
    def backward(self,Z,A):
        pass

class Sigmoid(CActivator):
    def forward(self,Z):
        A=1.0/(1.0+np.exp(-Z))
        return A
    def backward(self,Z,A,delta):
        dA=np.multiply(A,1-A)
        dZ=np.multiply(delta,dA)
        return dZ,dA
