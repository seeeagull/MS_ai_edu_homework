from pathlib import Path
from MiniFramework.Layer_3 import *
from MiniFramework.EnumDef_3 import *

class BnLayer(CLayer):
    def __init__(self,input_size,momentum=0.9):
        self.gamma=np.ones((1,input_size)).astype('float32')
        self.beta=np.zeros((1,input_size)).astype('float32')
        self.eps=1e-5
        self.momentum=momentum
        self.input_size=input_size
        self.output_size=input_size
        self.running_mean=np.zeros((1,input_size)).astype('float32')
        self.running_var=np.zeros((1,input_size)).astype('float32')

    def initialize(self,folder,name,create_new=False):
        self.result_file_name=str.format("{0}/{1}_result.npz",folder,name)

    def forward(self,input,train=True):
        assert(input.ndim==2 or input.ndim==4) #after fc or conv layer
        self.x=input
        if train:
            self.mean=np.mean(self.x,axis=0,keepdims=True)
            self.x_mean=self.x-self.mean
            self.var=np.mean(self.x_mean**2,axis=0,keepdims=True)+self.eps
            self.std=np.sqrt(self.var)
            self.norm_x=self.x_mean/self.std
            self.z=self.gamma*self.norm_x+self.beta
            self.running_mean=self.momentum*self.running_mean+(1-self.momentum)*self.mean
            self.running_var=self.momentum*self.running_var+(1-self.momentum)*self.var
        else:
            self.mean=self.running_mean
            self.var=self.running_var
            self.norm_x=(self.x-self.mean)/np.sqrt(self.var+self.eps)
            self.z=self.gamma*self.norm_x+self.beta
        return self.z

    def backward(self,delta_in,flag):
        #seems that we dont actually need "flag" here but Im lazy
        assert(delta_in.ndim==2 or delta_in.ndim==4) #after fc or conv layer
        m=self.x.shape[0]
        self.d_gamma=np.sum(delta_in*self.norm_x,axis=0,keepdims=True)
        self.d_beta=np.sum(delta_in,axis=0,keepdims=True)
        d_norm_x=self.gamma*delta_in
        d_var=np.sum(d_norm_x*self.x_mean,axis=0,keepdims=True)/(-2*self.std*self.var)
        d_mean=-np.sum(d_norm_x/self.std,axis=0,keepdims=True)-d_var*2*np.sum(self.x_mean,axis=0,keepdims=True)/m
        delta_out=d_norm_x/self.std+d_var*2*self.x_mean/m+d_mean/m
        if flag==-1:
            return delta_out,self.d_gamma,self.d_beta
        else:
            return delta_out

    def update(self,learning_rate=0.1):
        self.gamma-=self.d_gamma*learning_rate
        self.beta-=self.d_beta*learning_rate

    def save_parameters(self):
        np.savez(self.result_file_name,gamma=self.gamma,beta=self.beta,mean=self.running_mean,var=self.running_var)

    def load_parameters(self):
        data=np.load(self.result_file_name)
        self.gamma=data["gamma"]
        self.beta=data["beta"]
        self.running_mean=data["mean"]
        self.running_var=data["var"]
