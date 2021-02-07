from MiniFramework.EnumDef_3 import *
from MiniFramework.Layer_3 import *

class ClassificationLayer(CLayer):
    def __init__(self,activator):
        self.activator=activator

    def forward(self,x,train=True):
        self.x=x
        self.a=self.activator.forward(self.x)
        return self.a

    def backward(self,delta_in,layer_id): #已经在其他地方求过了
        return delta_in

class CClassifier(object):
    def forward(self,z):
        pass

class Logistic(CClassifier):
    def forward(self,z):
        a=1.0/(1.0+np.exp(-z))
        return a

class Softmax(CClassifier):
    def forward(self,z):
        shift_z=z-np.max(z,axis=1,keepdims=True)
        exp_z=np.exp(shift_z)
        a=exp_z/np.sum(exp_z,axis=1,keepdims=True)
        return a