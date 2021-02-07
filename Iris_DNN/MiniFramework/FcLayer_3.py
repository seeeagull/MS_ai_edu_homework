from MiniFramework.EnumDef_3 import *
from MiniFramework.Layer_3 import *
from MiniFramework.WeightsBias_3 import *
from MiniFramework.HyperParameters_3 import *

class FcLayer(CLayer):
    def __init__(self,input_size,output_size,hp):
        self.input_size=input_size
        self.output_size=output_size
        self.wb=WeightsBias_3(input_size,output_size,hp.init_meth,hp.optimizer_name,hp.eta)
        self.regular_name=hp.regular_name
        self.regular_val=hp.regular_val

    def initialize(self,folder,name):
        self.wb.Initialize(folder,name,False)

    def forward(self,input,train=True):
        self.input_shape=input.shape
        if input.ndim==4: #from pooling layer
            self.x=input.reshape(self.input_shape[0],-1)
        else:
            self.x=input
        self.z=np.dot(self.x,self.wb.W)+self.wb.B
        return self.z

    def backward(self,delta_in,layer_id):
        dZ=delta_in
        m=delta_in.shape[0]
        if self.regular_name==RegularMethod.L2:
            self.wb.dW=(np.dot(self.x.T,dZ)+self.regular_val*self.wb.W)/m
        elif self.regular_name==RegularMethod.L1:
            self.wb.dW=(np.dot(self.x.T,dZ)+self.regular_val*np.sign(self.wb.W))/m
        else:
            self.wb.dW=np.dot(self.x.T,dZ)
        self.wb.dB=np.sum(dZ,axis=0,keepdims=True)/m
        #calculate delta_out for lower level
        if layer_id==0:
            return None
        delta_out=np.dot(dZ,self.wb.W.T)
        if len(self.input_shape)>2:
            return delta_out.reshape(self.input_shape)
        else:
            return delta_out

    def pre_update(self):
        self.wb.pre_Update()

    def update(self):
        self.wb.Update()

    def save_parameters(self):
        self.wb.SaveResultValue()

    def load_parameters(self):
        self.wb.LoadResultValue()
