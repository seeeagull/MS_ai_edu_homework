from MiniFramework.EnumDef_3 import *

class OptimizerFactory(object):
    @staticmethod
    def CreateOptimizer(lr,name=OptimizerName.SGD): #lr stands for learning rate
        if name==OptimizerName.SGD:
            optimizer=SGD(lr)
        elif name==OptimizerName.Momentum:
            optimizer=Momentum(lr)
        elif name==OptimizerName.Nag:
            optimizer=Nag(lr)
        elif name==OptimizerName.AdaGrad:
            optimizer=AdaGrad(lr)
        elif name==OptimizerName.AdaDelta:
            optimizer=AdaDelta(lr)
        elif name==OptimizerName.RMSProp:
            optimizer=RMSProp(lr)
        elif name==OptimizerName.Adam:
            optimizer=Adam(lr)
        return optimizer

class Optimizer(object):
    def __init__(self):
        pass

    def pre_update(self,theta):
        pass

    def update(self,theta,grad):
        pass

class SGD(Optimizer):
    def __init__(self,lr):
        self.lr=lr

    def update(self,theta,grad):
        theta-=grad*self.lr
        return theta

class Momentum(Optimizer):
    def __init__(self,lr):
        self.vt=0
        self.alpha=0.9
        self.lr=lr

    def update(self,theta,grad):
        self.vt=self.alpha*self.vt+self.lr*grad
        theta-=self.vt
        return theta

class Nag(Optimizer):
    def __init__(self,lr):
        self.vt=0
        self.alpha=0.9
        self.lr=lr
    
    #先用预测的梯度更新一次
    def pre_update(self,theta):
        theta-=self.alpha*self.vt
        return theta

    #再用动量法更新一次
    def update(self,theta,grad):
        self.vt=self.alpha*self.vt+self.lr*grad
        theta-=self.vt
        return theta

class AdaGrad(Optimizer):
    def __init__(self,lr):
        self.eps=1e-6
        self.r=0
        self.lr=lr

    def update(self,theta,grad):
        self.r+=np.multiply(grad,grad) #按元素乘
        theta-=np.multiply(self.lr/(self.eps+np.sqrt(self.r)),grad)
        return theta

class AdaDelta(Optimizer):
    def __init__(self,lr):
        self.eps=1e-5
        self.alpha=0.9
        self.s=0 #累积变量矩阵
        self.r=0 #累积变变化量矩阵  代替lr
        self.lr=lr

    def update(self,theta,grad):
        self.s=self.alpha*self.s+(1.0-self.alpha)*np.multiply(grad,grad)
        delta_theta=np.multiply(np.sqrt((self.r+self.eps)/(self.s+self.eps)),grad)
        theta-=delta_theta
        self.r=self.alpha*self.r+(1.0-self.alpha)*np.multiply(delta_theta,delta_theta)
        return theta

class RMSProp(Optimizer):
    def __init__(self,lr):
        self.eps=1e-8
        self.alpha=0.9
        self.r=0 #累计变量矩阵
        self.lr=lr

    def update(self,theta,grad):
        self.r=self.alpha*self.r+(1.0-self.alpha)*np.multiply(grad,grad)
        delta_theta=np.multiply(self.lr/np.sqrt(self.r+self.eps),grad)
        theta-=delta_theta
        return theta

class Adam(Optimizer):
    def __init__(self,lr):
        self.t=0
        self.eps=1e-8
        self.beta1=0.9
        self.beta2=0.999
        self.lr=lr
        self.m=np.empty((1,1))
        self.v=np.empty((1,1))

    def update(self,theta,grad):
        self.t+=1
        self.m=self.beta1*self.m+(1.0-self.beta1)*grad
        self.v=self.beta2*self.v+(1.0-self.beta2)*np.multiply(grad,grad)
        m_hat=self.m/(1.0-self.beta1**self.t)
        v_hat=self.v/(1.0-self.beta2**self.t)
        delta_theta=self.lr*m_hat/(self.eps+np.sqrt(v_hat))
        theta-=delta_theta
        return theta
