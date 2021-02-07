import math
import os
import time
import sys
import numpy as np

from MiniFramework.EnumDef_3 import *
from MiniFramework.Layer_3 import *
from MiniFramework.ConvLayer_3 import *
from MiniFramework.FcLayer_3 import *
from MiniFramework.PoolingLayer_3 import *
from MiniFramework.DropoutLayer_3 import *
from MiniFramework.BatchNormLayer_3 import *
from MiniFramework.ActivationLayer_3 import *
from MiniFramework.ClassificationLayer_3 import *
from MiniFramework.HyperParameters_3 import *
from MiniFramework.TrainingHistory_3 import *
from MiniFramework.LossFunction_3 import *
from MiniFramework.DataReader_3 import *

class NeuralNet_3(object):
    def __init__(self,params,model_name):
        self.model_name=model_name
        self.hp=params
        self.layer_list=[]
        self.output=None
        self.layer_count=0
        self.subfolder=os.getcwd()+"/"+self.__create_subfolder()
        self.accuracy=0

    def __create_subfolder(self):
        if self.model_name != None:
            path=self.model_name.strip()
            path=path.rstrip("/") #?
            isExists=os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path

    def add_layer(self,layer,name):
        layer.initialize(self.subfolder,name)
        self.layer_count+=1
        self.layer_list.append(layer)

    def __forward(self,X,train=True):
        xinput=X
        for i in range(self.layer_count):
            layer=self.layer_list[i]
            xoutput=layer.forward(xinput,train)
            xinput=xoutput
        self.output=xoutput
        return self.output

    def inference(self,X):
        output=self.__forward(X,train=False)
        return output

    def __backward(self,Y):
        delta_in=self.output-Y
        for i in range(self.layer_count-1,-1,-1):
            layer=self.layer_list[i]
            delta_out=layer.backward(delta_in,i)
            delta_in=delta_out

    def __pre_update(self):
        for i in range(self.layer_count-1,-1,-1):
            layer=self.layer_list[i]
            layer.pre_update()

    def __update(self):
        for i in range(self.layer_count-1,-1,-1):
            layer=self.layer_list[i]
            layer.update()

    def __get_regular_cost_from_fc_layer(self,regularName):
        if regularName!=RegularMethod.L1 and regularName!=RegularMethod.L2:
            return 0

        regular_cost=0
        for i in range(self.layer_count-1,-1,-1):
            layer=self.layer_list[i]
            if isinstance(layer,Fclayer_3):
                if self.hp.regular_name==RegularMethod.L1:
                    regular_cost+=np.sum(np.abs(layer.wb.W))
                elif self.hp.regular_name==RegularMethod.L2:
                    regular_cost+=np.sum(np.square(layer.wb.W))
        return regular_cost*self.hp.regular_val

    def __check_weights_from_fc_layer(self):
        weights=0
        total=0
        littles=0
        zeros=0
        for i in range(self.layer_count-1,-1,-1):
            layer=self.layer_list[i]
            if isinstance(layer,Fclayer_3):
                weights+=np.sum(np.abs(layer.wb.W))
                total+=np.size(layer.wb.W)
                littles+=len(np.where(np.abs(layer.wb.W)<=0.01)[0]) #np.where返回满足condition的元素位置数组，维数和原数组相同
                zeros+=len(np.where(np.abs(layer.wb.W)<=0.0001)[0])

        print("total weights abs sum=",weights)
        print("total weights=",total)
        print("little weights=",littles)
        print("zero weights=",zeros)


    def train(self,sdr,checkpoint=0.1,need_test=True):
        t0=time.time()

        self.lossFunc=LossFunction_3(self.hp.net_type)
        if self.hp.regular_name==RegularMethod.EarlyStop:
            self.loss_trace=TrainingHistory_3(True,self.hp.regular_value)
        else:
            self.loss_trace=TrainingHistory_3()

        if self.hp.batch_size==-1 or self.hp.batch_size>sdr.num_train:
            self.hp.batch_size=sdr.num_train
        max_iteration=math.ceil(sdr.num_train/self.hp.batch_size)
        checkpoint_iteration=(int)(math.ceil(checkpoint*max_iteration))
        need_stop=False

        for epoch in range(self.hp.max_epoch):
            sdr.Shuffle()
            for iteration in range(max_iteration):
                X,Y=sdr.GetBatchTrainSamples(self.hp.batch_size,iteration)
                if self.hp.optimizer_name==OptimizerName.Nag:
                    self.__pre_update()
                self.__forward(X,train=True)
                self.__backward(Y)
                self.__update()

                tot_iteration=epoch*max_iteration+iteration
                if (tot_iteration+1)%checkpoint_iteration==0:
                    need_stop=self.CheckErrorAndLoss(sdr,X,Y,epoch,tot_iteration)
                    if need_stop:
                        break
            if need_stop:
                break

        t1=time.time()
        print("time used:",t1-t0)

        if need_test:
            print("testing...")
            self.accuracy=self.Test(sdr)

    def CheckErrorAndLoss(self,sdr,train_x,train_y,epoch,tot_iteration):
        print("epoch=%d,total_iteration=%d"%(epoch,tot_iteration))

        #l1/l2 cost
        regular_cost=self.__get_regular_cost_from_fc_layer(self.hp.regular_name)

        #calculate train loss
        self.__forward(train_x,train=False)
        loss_train,accuracy_train=self.lossFunc.CheckLoss(self.output,train_y)
        loss_train=loss_train+regular_cost/train_x.shape[0]
        print("loss_train=%.6f,accuracy_train=%f"%(loss_train,accuracy_train))

        #calculate validation loss
        vld_x,vld_y=sdr.GetVldSet()
        self.__forward(vld_x,train=False)
        loss_vld,accuracy_vld=self.lossFunc.CheckLoss(self.output,vld_y)
        loss_vld=loss_vld+regular_cost/vld_x.shape[0]
        print("loss_valid=%.6f,accuracy_valid=%f"%(loss_vld,accuracy_vld))

        if self.loss_trace.IsMaximum(accuracy_vld):
            self.save_parameters()

        need_stop=self.loss_trace.Add(epoch,tot_iteration,loss_train,accuracy_train,loss_vld,accuracy_vld,self.hp.stopper)
        return need_stop

    def Test(self,sdr):
        X,Y=sdr.GetTestSet()
        self.__forward(X,train=False)
        _,acc1=self.lossFunc.CheckLoss(self.output,Y)
        print(acc1)

        self.load_parameters()
        self.__forward(X,train=False)
        _,acc2=self.lossFunc.CheckLoss(self.output,Y)
        print(acc2)

        return acc2

    def save_parameters(self):
        print("save parameters")
        for i in range(self.layer_count):
            layer=self.layer_list[i]
            layer.save_parameters()
            
    def load_parameters(self):
        print("load parameters")
        for i in range(self.layer_count):
            layer=self.layer_list[i]
            layer.load_parameters()

    def ShowLossHistory(self,xcoor,xmin=None,xmax=None,ymin=None,ymax=None):
        title=str.format("{0},accuracy:{1:.4f}",self.hp.toString(),self.accuracy)
        self.loss_trace.ShowLossHistory(title,xcoor,xmin,xmax,ymin,ymax)
