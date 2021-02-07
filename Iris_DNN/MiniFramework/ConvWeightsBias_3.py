import math

from MiniFramework.EnumDef_3 import *
from MiniFramework.WeightsBias_3 import *
from MiniFramework.Optimizer_3 import *

class ConvWeightsBias_3(WeightsBias_3):
    def __init__(self,output_c,input_c,filter_h,filter_w,init_meth,optimizer_name,eta):
        self.FilterCount=output_c
        self.KernalCount=input_c
        self.FilterHeight=filter_h
        self.FilterWidth=filter_w
        self.init_meth=init_meth
        self.optimizer_name=optimizer_name
        self.learning_rate=eta

    def Initialize(self,folder,name,create_new):
        self.WBShape=(self.FilterCount,self.KernalCount,self.FilterHeight,self.FilterWidth)
        self.init_file_name=str.format("{0}/{1}_{2}_{3}_{4}_{5}_init.npz",folder,name,self.FilterCount,self.KernalCount,self.FilterHeight,self.FilterWidth)
        self.result_file_name=str.format("{0}/{1}_result.npz",folder,name)
        if create_new:
            self.CreateNew()
        else:
            self.LoadExistingParameters()
        self.CreateOptimizers()
        self.dW=np.zeros(self.W.shape).astype('float32')
        self.dB=np.zeros(self.B.shape).astype('float32')

    def CreateNew(self):
        self.W=ConvWeightsBias_3.InitialConvParameters(self.WBShape,self.init_meth)
        self.B=np.zeros((self.FilterCount,1)).astype('float32')
        self.SaveInitialValue()

    def LoadExistingParameters(self):
        w_file=Path(self.init_file_name)
        if w_file.exists():
            self.LoadInitialValue()
        else:
            self.CreateNew()

    def Rotate180(self):
        self.WT=np.zeros(self.W.shape).astype(np.float32)
        for i in range(self.FilterCount):
            for j in range(self.KernalCount):
                self.WT[i,j]=np.rot90(self.W[i,j],2)
        return self.WT

    def ClearGrads(self):
        self.dW=np.zeros(self.W.shape).astype(np.float32)
        self.dB=np.zeros(self.B.shape).astype(np.float32)

    def MeanGrads(self,m):
        self.dW/=m
        self.dB/=m

    @staticmethod
    def InitialConvParameters(shape,meth):
        assert(len(shape)==4)
        num_input=shape[2] #FilterHeight
        num_output=shape[3] #FilterWidth
        if meth==InitialMethod.Zero:
            W=np.zeros(shape).astype('float32')
        elif meth==InitialMethod.Normal:
            W=np.random.normal(shape).astype('float32')
        elif meth==InitialMethod.MSRA:
            #np.random.normal(loc,scale,size) 均值，标准差，尺寸
            W=np.random.normal(0,np.sqrt(2/num_input*num_output),shape).astype('float32')
        elif meth==InitialMethod.Xavier:
            t=math.sqrt(6/(num_input+num_output))
            #从[-t,t)随机采样
            W=np.random.uniform(-t,t,shape).astype('float32')
        return W
