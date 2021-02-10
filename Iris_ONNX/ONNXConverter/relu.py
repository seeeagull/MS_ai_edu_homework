import numpy as np

class Crelu(object):
    def __init__(self,inputSize,name=None,exname=""):
        self.shape=inputSize
        self.type="Relu"
        self.input_name=f'{exname}y' if exname else f'{name}x'
        self.input_size=[1]+list(inputSize)[1:]
        self.output_name=name+"y"
        self.output_size=[1]+list(inputSize)[1:]
    def forward(self,image):
        self.memory=np.zeros(self.shape)
        self.memory[image>0]=1
        return np.maximum(0,image)
    def gradient(self,preError):
        return np.multiply(self.memory,preError)
