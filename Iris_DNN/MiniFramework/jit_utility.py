from MiniFramework.EnumDef_3 import *
import numba as nb
from numba import float32,int32

@nb.jit(nopython=True)
def jit_maxpool_forward(x,batch_size,input_c,output_h,output_w,pool_h,pool_w,pool_stride):
    z=np.zeros((batch_size,input_c,output_h,output_w))
    for b in range(batch_size):
        for c in range(input_c):
            for i in range(output_h):
                i_start=i*pool_stride
                i_end=i_start+pool_h
                for j in range(output_w):
                    j_start=j*pool_stride
                    j_end=j_start+pool_w
                    target_array=x[b,c,i_start:i_end,j_start:j_end]
                    t=np.max(target_array)
                    z[b,c,i,j]=t
    return z

@nb.jit(nopython=True)
def jit_maxpool_backward(x,delta_in,batch_size,input_c,output_h,output_w,pool_h,pool_w,pool_stride):
    delta_out=np.zeros(x.shape)
    for b in range(batch_size):
        for c in range(input_c):
            for i in range(output_h):
                i_start=i*pool_stride
                i_end=i_start+pool_h
                for j in range(output_w):
                    j_start=j*pool_stride
                    j_end=j_start+pool_w
                    m,n=jit_get_max_index(x[b,c],i_start,i_end,j_start,j_end)
                    delta_out[b,c,m,n]=delta_in[b,c,i,j]
    return delta_out

@nb.jit(nopython=True)
def jit_get_max_index(input,i_start,i_end,j_start,j_end):
    assert(input.ndim==2)
    max_i,max_j=i_start,j_start
    max_val=input[max_i,max_j]
    for i in range(i_start,i_end):
        for j in range(j_start,j_end):
            if input[i,j]>max_val:
                max_val=input[i,j]
                max_i,max_j=i,j
    return max_i,max_j

@nb.jit(nopython=True)
def jit_conv_2d(input_array,kernal,bias,output_array):
    assert(input_array.ndim==2)
    assert(output_array.ndim==2)
    assert(kernal.ndim==2)
    output_h,output_w=output_array.shape[0],output_array.shape[1]
    kernal_h,kernal_w=kernal.shape[0],kernal.shape[1]
    for i in range(output_h):
        for j in range(output_w):
            target_array=input_array[i:i+kernal_h,j:j+kernal_w]
            output_array[i,j]=np.sum(np.multiply(target_array,kernal))+bias

@nb.jit(nopython=True)
def jit_conv_4d(x,weights,bias,out_h,out_w,stride=1):
    assert(x.ndim==4)
    assert(x.shape[1]==weights.shape[1])
    #weights.shape(filter,kernal,height,width)
    batch_size,input_c=x.shape[0],x.shape[1]
    output_c,filter_h,filter_w=weights.shape[0],weights.shape[2],weights.shape[3]
    rs=np.zeros((batch_size,output_c,out_h,out_w))
    for b in range(batch_size):
        for oc in range(output_c):
            rs[b,oc]+=bias[oc]
            for ic in range(input_c):
                for i in range(out_h):
                    for j in range(out_w):
                        ii,jj=i*stride,j*stride
                        target_array=x[b,ic,ii:ii+filter_h,jj:jj+filter_w]
                        rs[b,oc,i,j]+=np.sum(np.multiply(target_array,weights[oc,ic]))
    return rs

@nb.jit(nopython=True)
def calculate_output_size(input_h,input_w,filter_h,filter_w,padding,stride=1):
    output_h=(input_h+2*padding-filter_h)//stride+1
    output_w=(input_w+2*padding-filter_w)//stride+1
    return (output_h,output_w)

@nb.jit(nopython=True)
def calculate_padding_size(input_h,input_w,filter_h,filter_w,output_h,output_w,stride=1):
    pad_h=((output_h-1)*stride+filter_h-input_h)//2
    pad_w=((output_w-1)*stride+filter_w-input_w)//2
    return (pad_h,pad_w)

@nb.jit(nopython=True)
def expand_delta_map(dZ,batch_size,input_c,input_h,input_w,output_h,output_w,filter_h,filter_w,padding,stride):
    assert(dZ.ndim==4)
    expand_h=0
    expand_w=0
    if stride==1:
        dZ_stride_1=dZ
        expand_h=dZ.shape[2]
        expand_w=dZ.shape[3]
    else:
        #计算如果stride=1输出的尺寸大小，然后按这个大小隔一个stride填一个数
        (expand_h,expand_w)=calculate_output_size(input_h,input_w,filter_h,filter_w,padding,1)
        dZ_stride_1=np.zeros((batch_size,input_c,expand_h,expand_w))
        for b in range(batch_size):
            for c in range(input_c):
                for i in range(output_h):
                    for j in range(output_w):
                        ii=i*stride
                        jj=j*stride
                        dZ_stride_1[b,c,ii,jj]=dZ[b,c,i,j]
    return dZ_stride_1

@nb.jit(nopython=True)
def calculate_weights_grad(x,dZ,batch_size,output_c,input_c,filter_h,filter_w,dW,dB):
    for b in range(batch_size):
        for oc in range(output_c): #filter count
            for ic in range(input_c): #kernal count
                tmp_dw=np.zeros((filter_h,filter_w)).astype(np.float32)
                jit_conv_2d(x[b,ic],dZ[b,oc],0,tmp_dw)
                dW[oc,ic]+=tmp_dw
            dB[oc]+=dZ[b,oc].sum()
    return (dW,dB)

@nb.jit(nopython=True)
def calculate_delta_out(dZ,rot_weights,batch_size,input_c,output_c,input_h,input_w,delta_out):
    for b in range(batch_size):
        for oc in range(output_c): #filter count
            delta_per_input=np.zeros((input_h,input_w)).astype(np.float32)
            for ic in range(input_c): #kernal count
                jit_conv_2d(dZ[b,oc],rot_weights[oc,ic],0,delta_per_input)
                delta_out[b,ic]+=delta_per_input
    return delta_out

def img2col(input_data,filter_h,filter_w,stride=1,pad=0):
    N, C, H, W = input_data.shape
    out_h=(H+2*pad-filter_h)//stride+1
    out_w=(W+2*pad-filter_w)//stride+1
    img=input_data
    if pad>0:
        img=np.pad(input_data,[(0,0),(0,0),(pad,pad),(pad,pad)],'constant')
    col=np.zeros((N,C,filter_h,filter_w,out_h,out_w))

    for i in range(filter_h):
        i_max=i+stride*out_h
        for j in range(filter_w):
            j_max=j+stride*out_w
            col[:,:,i,j,:,:]= img[:,:,i:i_max:stride,j:j_max:stride]
    col=np.transpose(col,axes=(0,4,5,1,2,3)).reshape(N*out_h*out_w,-1)
    return col


def col2img(col,input_shape,filter_h,filter_w,stride,pad,out_h,out_w):
    N,C,H,W=input_shape
    tmp1=col.reshape(N,out_h,out_w,C,filter_h,filter_w)
    tmp2=np.transpose(tmp1,axes=(0,3,4,5,1,2))
    img=np.zeros((N,C,H+2*pad+stride-1,W+2*pad+stride-1))
    for i in range(filter_h):
        i_max=i+stride*out_h
        for j in range(filter_w):
            j_max=j+stride*out_w
            img[:,:,i:i_max:stride,j:j_max:stride]+=tmp2[:,:,i,j,:,:]
    return img[:,:,pad:H+pad,pad:W+pad]


def col2img2(col,input_shape,filter_h,filter_w,stride,pad,out_h,out_w):
    N,C,H,W=input_shape
    tmp1=col.reshape(N,out_h,out_w,C,filter_h,filter_w)
    tmp2=np.transpose(tmp1,axes=(0,3,4,5,1,2))
    a=fill(filter_h,filter_w,stride,out_h,out_w,pad,tmp2,H,W,N,C)
    return a

@nb.jit(nopython=True)
def fill(filter_h,filter_w,stride,out_h,out_w,pad,tmp2,H,W,N,C):
    img=np.zeros((N,C,H+2*pad+stride-1,W+2*pad+stride-1))
    for i in range(filter_h):
        i_max=i+stride*out_h
        for j in range(filter_w):
            j_max=j+stride*out_w
            img[:,:,i:i_max:stride,j:j_max:stride]+=tmp2[:,:,i,j,:,:]
    return img[:,:,pad:H+pad,pad:W+pad]