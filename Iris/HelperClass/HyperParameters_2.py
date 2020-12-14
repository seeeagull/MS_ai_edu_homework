from HelperClass.EnumDef_2 import *

class HyperParameters_2(object):
    def __init__(self,num_input,num_hidden,num_output,eta=0.1,max_epoch=10000,
                 batch_size=5,eps=0.1,init_meth=InitialMethod.Xavier):
        self.n_input=num_input
        self.n_hidden=num_hidden
        self.n_output=num_output
        self.eta=eta
        self.max_epoch=max_epoch
        self.batch_size=batch_size
        self.eps=eps
        self.init_meth=init_meth
    def GetTitle(self):
        title="batch=%d,eta=%.3f,hidden=%d"%(self.batch_size,self.eta,self.n_hidden)
        return title
