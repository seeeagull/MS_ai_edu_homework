import numpy as np


class CClassifier(object):
    def forward(self,z):
        pass

class Logistic(CClassifier):
    def forward(self,z):
        return 1.0/(1.0+np.exp(-z))

class Softmax(CClassifier):
    def forward(self,z):
        z_shift=z-np.max(z,axis=1,keepdims=True)
        z_exp=np.exp(z_shift)
        return z_exp/np.sum(z_exp,axis=1,keepdims=True)