import matplotlib.pyplot as plt
from pathlib import Path

from HelperClass.DataReader_2 import *
from HelperClass.NeuralNet_2 import *
from HelperClass.HyperParameters_2 import *
from HelperClass.EnumDef_2 import *

if __name__=="__main__":
    sdr=DataReader_2()
    sdr.ReadDataIris()
    hp=HyperParameters_2(4,3,3,eta=0.1,max_epoch=5000,batch_size=5,eps=1-4)
    net=NeuralNet_2(hp,"iris_2")
    sdr.NormalizeX()
    sdr.NormalizeY(base=0)
    sdr.Shuffle()
    sdr.GenerateTestSet()
    sdr.GenerateValidationSet()
    net.train(sdr,checkpoint=10)
    net.ShowTrainingHistory()
    net.inference(sdr.XTest)
    rA=np.argmax(net.output,axis=1)
    rY=np.argmax(sdr.YTest,axis=1)
    R=(rA==rY)
    rate=R.sum()/sdr.num_tst
    print("%.2lf%%"%(rate*100.0))
