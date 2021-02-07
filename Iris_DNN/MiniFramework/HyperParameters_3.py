from MiniFramework.EnumDef_3 import *

class HyperParameters_3(object):
    def __init__(self,eta=0.1,max_epoch=10000,batch_size=5,
        net_type=NetType.Fitting,
        init_meth=InitialMethod.Xavier,
        optimizer_name=OptimizerName.SGD,
        stopper=None,
        regular_name=RegularMethod.Nothing,regular_val=1.0):
        self.eta=eta
        self.max_epoch=max_epoch
        if batch_size==-1:
            self.batch_size=self.num_example
        else:
            self.batch_size=batch_size
        self.net_type=net_type
        self.init_meth=init_meth
        self.optimizer_name=optimizer_name
        self.stopper=stopper
        self.regular_name=regular_name
        self.regular_val=regular_val

    def toString(self):
        title=str.format("bz:{0},eta:{1},init:{2},op:{3}",self.batch_size,self.eta,self.init_meth.name,self.optimizer_name.name)
        if self.regular_name!=RegularMethod.Nothing:
            title+=str.format(",rgl:{0}:{1}",self.regular_name,self.regular_val)
        return title