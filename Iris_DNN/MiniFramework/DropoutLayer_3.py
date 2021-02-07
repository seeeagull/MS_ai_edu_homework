from MiniFramework.EnumDef_3 import *
from MiniFramework.Layer_3 import *

class DropoutLayer_3(CLayer):
    def __init__(self,input_size,ratio=0.5):
        self.input_size=input_size
        self.output_size=input_size
        self.dropout_ratio=ratio
        self.mask=None

    def forward(self,input,train=True):
        self.input_shape=input.shape
        if train:
            self.mask=np.random.rand(*input.shape)>self.dropout_ratio
            self.z=input*self.mask
        else:
            self.z=input*(1.0-self.dropout_ratio)
        return self.z

    def backward(self,delta_in,layer_id):
        delta_out=self.mask*delta_in
        return delta_out