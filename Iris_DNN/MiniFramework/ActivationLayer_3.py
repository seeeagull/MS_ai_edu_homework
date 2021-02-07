from MiniFramework.EnumDef_3 import *
from MiniFramework.Layer_3 import *

class ActivationLayer(CLayer):
    def __init__(self,activator):
        self.activator=activator

    def forward(self,z,train=True):
        self.z=z
        self.a=self.activator.forward(self.z)
        return self.a

    def backward(self,delta_in,layer_id):
        dZ=self.activator.backward(self.z,self.a,delta_in)
        return dZ

class CActivator(object):
    def forward(self,z):
        pass
    def backward(self,z,a,delta_in):
        pass
    def get_name(self):
        return self.__class__.__name__

class Identity(CActivator):
    def forward(self,z):
        return z
    def backward(self,z,a,delta_in):
        return delta_in

class Sigmoid(CActivator):
    def forward(self,z):
        a=1.0/(1.0+np.exp(-z))
        return a
    def backward(self,z,a,delta_in):
        dA=np.multiply(a,1-a)
        dZ=np.multiply(delta_in,dA)
        return dZ

class Tanh(CActivator):
    def forward(self,z):
        a=2.0/(1.0+np.exp(-2.0*z))-1.0
        return a
    def backward(self,z,a,delta_in):
        dA=1.0-np.multiply(a,a)
        dZ=np.multiply(delta_in,dA)
        return dZ

class Relu(CActivator):
    def forward(self,z):
        a=np.maximum(z,0)
        return a
    def backward(self,z,a,delta_in):#根据Z而不是A是否>0
        dA=np.zeros(z.shape).astype('float32')
        dA[z>0]=1
        dZ=np.multiply(dA,delta_in)
        return dZ