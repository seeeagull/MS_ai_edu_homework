from MiniFramework.EnumDef_3 import *
from MiniFramework.Layer_3 import *
from MiniFramework.ConvWeightsBias_3 import*
from MiniFramework.jit_utility import *

class ConvLayer(CLayer):
    def __init__(self,
                 input_shape, #(InputChannelCount,H,W)
                 output_shape, #(OutputChannelCount,FH,FW)
                 conv_param, #(stride,padding)
                 hp):
        self.InC,self.InH,self.InW=input_shape[0],input_shape[1],input_shape[2]
        self.OutC,self.FH,self.FW=output_shape[0],output_shape[1],output_shape[2]
        self.stride,self.padding=conv_param[0],conv_param[1]
        self.hp=hp

    def initialize(self,folder,name,create_new=False):
        self.WB=ConvWeightsBias_3(self.OutC,self.InC,self.FH,self.FW,
                                self.hp.init_meth,self.hp.optimizer_name,self.hp.eta)
        self.WB.Initialize(folder,name,create_new)
        (self.OutH,self.OutW)=calculate_output_size(self.InH,self.InW,self.FH,self.FW,self.padding,self.stride)
        self.output_shape=(self.OutC,self.OutH,self.OutW)

    def set_filter(self,w,b):
        if w is not None:
            self.WB.W=w
        if b is not None:
            self.WB.B=b

    def forward(self,x,train=True):
        return self.forward_img2col(x,train)

    def backward(self,delta_in,layer_id):
        delta_out,dw,db=self.backward_numba(delta_in,layer_id)
        return delta_out
    
    def forward_img2col(self,x,train=True):
        self.x=x
        self.batch_size=self.x.shape[0]
        assert(self.x.shape==(self.batch_size,self.InC,self.InH,self.InW))
        self.col_x=img2col(x,self.FH,self.FW,self.stride,self.padding)
        self.col_w=self.WB.W.reshape(self.OutC,-1).T
        self.col_b=self.WB.B.reshape(-1,self.OutC)
        out1=np.dot(self.col_x,self.col_w)+self.col_b
        out2=out1.reshape(self.batch_size,self.OutH,self.OutW,-1)
        self.z=np.transpose(out2,axes=(0,3,1,2))
        return self.z

    def backward_col2img(self,delta_in,layer_idx):
        col_delta_in=np.transpose(delta_in,axes=(0,2,3,1)).reshape(-1,self.OutC)
        self.WB.dB=np.sum(col_delta_in,axis=0,keepdims=True).T/self.batch_size
        col_dW=np.dot(self.col_x.T,col_delta_in)/self.batch_size
        self.WB.dW=np.transpose(col_dW,axes=(1,0)).reshape(self.OutC,self.InC,self.FH,self.FW)
        col_delta_out=np.dot(col_delta_in,self.col_w.T)
        delta_out=col2img(col_delta_out,self.x.shape,self.FH,self.FW,self.stride,self.padding,self.OutH,self.OutW)
        return delta_out,self.WB.dW,self.WB.dB

    def forward_numba(self,x,train=True):
        assert(x.ndim==4)
        self.x=x
        assert(self.x.shape[1]==self.InC)
        assert(self.x.shape[2]==self.InH)
        assert(self.x.shape[3]==self.InW)
        self.batch_size=x.shape[0]
        if self.padding>0:
            self.padded=np.pad(self.x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),'constant')
        else:
            self.padded=self.x
        self.z=jit_conv_4d(self.padded,self.WB.W,self.WB.B,self.OutH,self.OutW,self.stride)
        return self.z

    def backward_numba(self,delta_in,layer_id):
        assert(delta_in.ndim==4)
        assert(delta_in.shape==self.z.shape)
        #如果正向计算时stride不是1，先把delta_in转换成是1的等价误差数组
        dz_stride_1=expand_delta_map(
            delta_in,self.batch_size,
            self.OutC,self.InH,self.InW,
            self.OutH,self.OutW,
            self.FH,self.FW,
            self.padding,self.stride)
        #计算本层权重矩阵梯度 dW=A*delta_in dB=delta_in
        self._calculate_weightsbias_grad(dz_stride_1)
        #Z=W*A+B delta_in为Z的梯度，求对A的梯度dA=delta_in_padded*W_rot180,所有卷积核结果相加
        #pad delta_in 使输出dA与A尺寸一致
        (pad_h,pad_w)=calculate_padding_size(
            dz_stride_1.shape[2],dz_stride_1.shape[3],
            self.FH,self.FW,self.InH,self.InW)
        dz_padded=np.pad(dz_stride_1,((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)),'constant')
        delta_out=self._calculate_delta_out(dz_padded,layer_id)
        return delta_out,self.WB.dW,self.WB.dB

    def _calculate_weightsbias_grad(self,dz):#公式见上
        self.WB.ClearGrads()
        #padding
        (pad_h,pad_w)=calculate_padding_size(
            self.InH,self.InW,
            dz.shape[2],dz.shape[3],
            self.FH,self.FW,1)
        input_padded=np.pad(self.x,((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)),'constant')
        #calculate
        (self.WB.dW,self.WB.dB)=calculate_weights_grad(
            input_padded,dz,self.batch_size,
            self.OutC,self.InC,self.FH,self.FW,
            self.WB.dW,self.WB.dB)
        self.WB.MeanGrads(self.batch_size)

    def _calculate_delta_out(self,dz,layer_id):
        #此时dz已经pad过了
        if layer_id==0:
            return None
        rot_weights=self.WB.Rotate180()
        #定义输出矩阵形状  同前向计算时输入矩阵
        delta_out=np.zeros(self.x.shape).astype(np.float32)
        delta_out=calculate_delta_out(dz,rot_weights,self.batch_size,
                                      self.InC,self.OutC,self.InH,self.InW,delta_out)
        return delta_out

    def pre_update(self):
        pass

    def update(self):
        self.WB.Update()

    def save_parameters(self):
        self.WB.SaveResultValue()

    def load_parameters(self):
        self.WB.LoadResultValue()
