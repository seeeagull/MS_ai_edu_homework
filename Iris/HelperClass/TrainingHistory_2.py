import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

class TrainingHistory_2(object):
    def __init__(self):
        self.loss_train=[]
        self.accuracy_train=[]
        self.iteration_seq=[]
        self.epoch_seq=[]
        self.loss_val=[]
        self.accuracy_val=[]
    def Add(self,epoch,iteration,loss_train,accuracy_train,loss_vld,accuracy_vld):
        self.iteration_seq.append(iteration)
        self.epoch_seq.append(epoch)
        self.loss_train.append(loss_train)
        self.accuracy_train.append(accuracy_train)
        if loss_vld is not None:
            self.loss_val.append(loss_vld)
        if accuracy_vld is not None:
            self.accuracy_val.append(accuracy_vld)
        return False
    def ShowLossHistory(self,hp,xmin=None,xmax=None,ymin=None,ymax=None):
        fig=plt.figure(figsize=(12,5))
        axes=plt.subplot(1,2,1)
        p2,=axes.plot(self.epoch_seq,self.loss_train)
        p1,=axes.plot(self.epoch_seq,self.loss_val)
        axes.legend([p1,p2],["validation","train"])
        axes.set_title("Loss")
        axes.set_ylabel("loss")
        axes.set_xlabel("epoch")
        if xmax!=None or xmin!=None or ymin!=None or ymax!=None:
            axes.axis([xmin,xmax,ymin,ymax])
        axes=plt.subplot(1,2,2)
        p2,=axes.plot(self.epoch_seq,self.accuracy_train)
        p1,=axes.plot(self.epoch_seq,self.accuracy_val)
        axes.legend([p1,p2],["validation","train"])
        axes.set_title("Accuracy")
        axes.set_ylabel("accuracy")
        axes.set_xlabel("epoch")
        title=hp.GetTitle()
        plt.suptitle(title)
        plt.show()
        return title
    def Dump(self,file_name):
        f=open(file_name,'wb')
        pickle.dump(self,f)
    def Load(file_name):
        f=open(file_name,'rb')
        lh=pickle.load(f)
        return lh