import time
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_3 import *
from MiniFramework.EnumDef_3 import *
from MiniFramework.DataReader_3 import *


def LoadData():
    print("reading data...")
    dr=DataReader_3()
    dr.ReadDataIris()
    dr.NormalizeX()
    dr.NormalizeY(NetType.MultipleClassifier,base=0)
    dr.Shuffle()
    dr.GenerateValidationSet(k=10)
    dr.GenerateTestSet(k=10)
    return dr

def dnn_model():
    num_output=3
    max_epoch=2500
    batch_size=5
    learning_rate=0.00004
    params=HyperParameters_3(
        learning_rate,max_epoch,batch_size,
        net_type=NetType.MultipleClassifier,
        init_meth=InitialMethod.Xavier,
        optimizer_name=OptimizerName.Adam)

    net=NeuralNet_3(params,"iris_dnn")
    
    f1=FcLayer(4,16,params)
    net.add_layer(f1,"f1")
    r1=ActivationLayer(Relu())
    net.add_layer(r1,"r1")
    '''
    f2=FcLayer(f1.output_size,16,params)
    net.add_layer(f2,"f2")
    r2=ActivationLayer(Relu())
    net.add_layer(r2,"r2")
    '''
    f3=FcLayer(f1.output_size,8,params)
    net.add_layer(f3,"f3")
    r3=ActivationLayer(Relu())
    net.add_layer(r3,"r3")

    f4=FcLayer(f3.output_size,num_output,params)
    net.add_layer(f4,"f4")
    s4=ClassificationLayer(Softmax())
    net.add_layer(s4,"s4")
    
    return net

if __name__=='__main__':
    dataReader=LoadData()
    net=dnn_model()
    net.train(dataReader,checkpoint=30,need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)

    net.inference(dataReader.XTest)
    rA=np.argmax(net.output,axis=1)
    rY=np.argmax(dataReader.YTest,axis=1)
    R=(rA==rY)
    rate=R.sum()/dataReader.num_tst
    print("test result: %.2lf%%"%(rate*100.0))
