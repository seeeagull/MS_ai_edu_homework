import os
import numpy as np
from ONNXConverter.fc import Cfc
from ONNXConverter.relu import Crelu
from ONNXConverter.sigmoid import Csigmoid
from ONNXConverter.softmax import Csoftmax
from ONNXConverter.save import model_save
from ONNXConverter.transfer import ModelTransfer

f1_wb=np.load("./iris_dnn/f1_result.npz")
f3_wb=np.load("./iris_dnn/f3_result.npz")
f4_wb=np.load("./iris_dnn/f4_result.npz")

class Cmodel(object):
    def __init__(self):
        self.f1=Cfc([1,4],f1_wb["W"].shape[1],name="f1",exname="")
        self.r1=Crelu(self.f1.outputShape,name="r1",exname="f1")
        
        self.f3=Cfc(self.f1.outputShape,f3_wb["W"].shape[1],name="f3",exname="r1")
        self.r3=Crelu(self.f3.outputShape,name="r3",exname="f3")

        self.f4=Cfc(self.f3.outputShape,f4_wb["W"].shape[1],name="f4",exname="r3")
        self.s4=Csoftmax(self.f4.outputShape,name="s4",exname="f4")

        self.model=[self.f1,self.r1,
                    self.f3,self.r3,
                    self.f4,self.s4]

        self.f1.W=f1_wb["W"]
        self.f1.B=f1_wb["B"].reshape(self.f1.B.shape)
        self.f3.W=f3_wb["W"]
        self.f3.B=f3_wb["B"].reshape(self.f3.B.shape)
        self.f4.W=f4_wb["W"]
        self.f4.B=f4_wb["B"].reshape(self.f4.B.shape)

    def save_model(self,path="./"):
        model_path=os.path.join(path,"model.json")
        model_save(self.model,path)
        ModelTransfer(model_path,os.path.join(path,"iris.onnx"))

if __name__=='__main__':
    model=Cmodel()
    save_path='ONNX'
    model.save_model(save_path)
    print(f'Succeed! Your modelfile is {os.path.join(save_path,"iris.onnx")}')