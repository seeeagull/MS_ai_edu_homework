import matplotlib.pyplot as plt
import pickle

from MiniFramework.EnumDef_3 import *

class TrainingHistory_3(object):
    def __init__(self,need_earlyStop=False,patience=5):
        self.epoch_seq=[]
        self.iteration_seq=[]
        self.loss_train=[]
        self.accuracy_train=[]
        self.loss_val=[]
        self.accuracy_val=[]
        self.cnt=0
        self.early_stop=need_earlyStop
        self.patience=patience
        self.patience_cnt=0
        self.max_vld_acc=0
        self.last_vld_loss=float("inf")

    def Add(self,epoch,iteration,loss_train,accuracy_train,loss_vld,accuracy_vld,stopper):
        self.epoch_seq.append(epoch)
        self.iteration_seq.append(iteration)
        self.loss_train.append(loss_train)
        self.accuracy_train.append(accuracy_train)
        if loss_vld is not None:
            self.loss_val.append(loss_vld)
        if accuracy_vld is not None:
            self.accuracy_val.append(accuracy_vld)

        if stopper is not None:
            if stopper.stop_condition==StopCondition.StopDiff: # loss基本稳定时就停止
                if len(loss_val)>1:
                    if abs(loss_val[-1]-loss_val[-2])<stopper.stop_value:
                        self.cnt+=1
                        if self.cnt>3:
                            return True #stop
                    else:
                        self.cnt=0
            elif stopper.stop_condition==StopCondition.StopLoss:
                if loss_vld<=stopper.stop_value:
                    return True

        if self.early_stop: #loss出现上升趋势就及时停止
            if loss_vld<self.last_vld_loss:
                self.last_vld_loss=loss_vld
                self.patience_cnt=0
            else:
                self.patience_cnt+=1
                if self.patience_cnt>=self.patience:
                    return True #stop
        return False

    def IsMaximum(self,acc_vld):
        if acc_vld is not None:
            if self.max_vld_acc<acc_vld:
                self.max_vld_acc=acc_vld
                return True
        return False

    def ShowLossHistory(self,title,xcoord,xmin=None,xmax=None,ymin=None,ymax=None):
        fig=plt.figure(figsize=[12,5])

        axes=plt.subplot(1,2,1)
        if xcoord==XCoordinate.Epoch:
            p1,=axes.plot(self.epoch_seq,self.loss_train)
            p2,=axes.plot(self.epoch_seq,self.loss_val)
            axes.set_xlabel("epoch")
        elif xcoord==XCoordinate.Iteration:
            p1,=axes.plot(self.iteration_seq,self.loss_train)
            p2,=axes.plot(self.iteration_seq,self.loss_val)
            axes.set_xlabel("iteration")
        axes.set_ylabel("loss")
        axes.legend([p1,p2],["train","validation"])
        axes.set_title("Loss")

        axes=plt.subplot(1,2,2)
        if xcoord==XCoordinate.Epoch:
            p1,=axes.plot(self.epoch_seq,self.accuracy_train)
            p2,=axes.plot(self.epoch_seq,self.accuracy_val)
            axes.set_xlabel("epoch")
        elif xcoord==XCoordinate.Iteration:
            p1,=axes.plot(self.iteration_seq,self.accuracy_train)
            p2,=axes.plot(self.iteration_seq,self.accuracy_val)
            axes.set_xlabel("iteration")
        axes.set_ylabel("accuracy")
        axes.legend([p1,p2],["train","validation"])
        axes.set_title("Accuracy")

        plt.suptitle(title)
        plt.show()
        return title

    def GetEpochNumber(self):
        return self.epoch_seq[-1]

    def GetLatestAverageLoss(self,count=10):
        tot=len(self.loss_val)
        if count>tot:
            count=tot
        tmp=self.loss_val[tot-count:tot]
        return sum(tmp)/count

    def Dump(self,file_name):
        f=open(file_name,'wb')
        pickle.dump(self,f)

    def Load(file_name):
        f=open(file_name,'rb')
        lh=pickle.load(f)
        return lh
