import numpy as np
import math
from pathlib import Path

from MiniFramework.EnumDef_3 import *
from MiniFramework.Optimizer_3 import *

class WeightsBias_3(object):
    def __init__(self,n_input,n_output,init_meth,optimizer_name,eta):
        self.num_input=n_input
        self.num_output=n_output
        self.init_meth=init_meth
        self.optimizer_name=optimizer_name
        self.learning_rate=eta
 
    def Initialize(self,folder,name,create_new):
        self.folder=folder
        self.init_file_name=str.format("{0}/{1}_{2}_{3}_init.npz",folder,name,self.num_input,self.num_output)
        self.result_file_name=str.format("{0}/{1}_result.npz",folder,name)
        if create_new:
            self.CreateNew()
        else:
            self.LoadExistingParameters()
        self.CreateOptimizers()
        self.dW=np.zeros(self.W.shape).astype('float32')
        self.dB=np.zeros(self.B.shape).astype('float32')

    def CreateNew(self):
        self.W,self.B=WeightsBias_3.InitialParameters(self.num_input,self.num_output,self.init_meth)
        self.SaveInitialValue()

    def LoadExistingParameters(self):
        w_file=Path(self.init_file_name)
        if w_file.exists():
            self.LoadInitialValue()
        else:
            self.CreateNew()

    def CreateOptimizers(self):
        self.oW=OptimizerFactory.CreateOptimizer(self.learning_rate,self.optimizer_name)
        self.oB=OptimizerFactory.CreateOptimizer(self.learning_rate,self.optimizer_name)

    def pre_Update(self):
        if self.optimizer_name==OptimizerName.Nag:
            self.W=self.oW.pre_update(self.W)
            self.B=self.oB.pre_update(self.B)

    def Update(self):
        self.W=self.oW.update(self.W,self.dW)
        self.B=self.oB.update(self.B,self.dB)

    def SaveInitialValue(self):
        np.savez(self.init_file_name,W=self.W,B=self.B)

    def LoadInitialValue(self):
        data=np.load(self.init_file_name)
        self.W,self.B=data["W"],data["B"]

    def SaveResultValue(self):
        np.savez(self.result_file_name,W=self.W,B=self.B)

    def LoadResultValue(self):
        data=np.load(self.result_file_name)
        self.W,self.B=data["W"],data["B"]

    @staticmethod
    def InitialParameters(n_input,n_output,meth):
        if meth==InitialMethod.Zero:
            W=np.zeros((n_input,n_output)).astype('float32')
        elif meth==InitialMethod.Normal:
            W=np.random.normal(size=(n_input,n_output)).astype('float32')
        elif meth==InitialMethod.MSRA:
            W=np.random.normal(0,np.sqrt(2/n_output),size=(n_input,n_output)).astype('float32')
        elif meth==InitialMethod.Xavier:
            t=math.sqrt(6/(n_input+n_output))
            W=np.random.uniform(-t,t,(n_input,n_output)).astype('float32')
        B=np.zeros((1,n_output)).astype('float32')
        return W,B
