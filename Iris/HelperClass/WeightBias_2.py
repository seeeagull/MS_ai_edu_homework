import numpy as np
from pathlib import Path

from HelperClass.EnumDef_2 import *

class WeightBias_2(object):
    def __init__(self,n_input,n_output,init_meth,eta):
        self.n_input=n_input
        self.n_output=n_output
        self.init_meth=init_meth
        self.eta=eta
        self.init_val_filename=str.format("w_{0}_{1}_{2}_init",self.n_input,self.n_output,self.init_meth.name)
    def InitializeWeights(self,folder,creat_new):
        self.folder=folder
        if creat_new:
            self.__CreateNew
        else:
            self.__LoadExistingParameters()
        self.dW=np.zeros_like(self.W)
        self.dB=np.zeros_like(self.B)
    def __CreateNew(self):
        self.W,self.B=WeightBias_2.InitialParameters(self.n_input,self.n_output,self.init_meth)
        self.__SaveInitialValue()
    def __LoadExistingParameters(self):
        file_name=str.format("{0}/{1}.npz",self.folder,self.init_val_filename)
        w_file=Path(file_name)
        if w_file.exists():
            self.__LoadInitialValue()
        else:
            self.__CreateNew()
    def Update(self):
        self.W-=self.eta*self.dW
        self.B-=self.eta*self.dB
    def __SaveInitialValue(self):
        file_name=str.format("{0}/{1}.npz",self.folder,self.init_val_filename)
        np.savez(file_name,weights=self.W,bias=self.B)
    def __LoadInitialValue(self):
        file_name=str.format("{0}/{1}.npz",self.folder,self.init_val_filename)
        data=np.load(file_name)
        self.W,self.B=data["weights"],data["bias"]
    def SaveResultValue(self,folder,name):
        file_name=str.format("{0}/{1}.npz",folder,name)
        np.savez(file_name,weights=self.W,bias=self.B)
    def LoadResultValue(self,folder,name):
        file_name=str.format("{0}/{1}.npz",folder,name)
        data=np.load(file_name)
        self.W,self.B=data["weights"],data["bias"]
    @staticmethod
    def InitialParameters(n_input,n_output,meth):
        if meth==InitialMethod.Zero:
            W=np.zeros((n_input,n_output))
        elif meth==InitialMethod.Normal:
            W=np.random.normal(size=(n_input,n_output))
        elif meth==InitialMethod.MSRA:
            W=np.random.normal(0,np.sqrt(2/n_output),siz=(n_input,n_output))
        elif meth==InitialMethod.Xavier:
            W=np.random.uniform(-np.sqrt(6/(n_input+n_output)),np.sqrt(6/(n_input+n_output)),size=(n_input,n_output))
        B=np.zeros((1,n_output))
        return W,B