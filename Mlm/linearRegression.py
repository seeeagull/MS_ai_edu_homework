import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HelperClass.DataReader_1 import *
from HelperClass.NeuralNet_1 import *
from HelperClass.HyperParameters_1 import *


def ShowResult(net,reader):
    X,Y=reader.GetWholeRawTrainSamples()
    fig=plt.figure(num="LinearRegression")
    ax=Axes3D(fig)
    ax.scatter(X[:,0],X[:,1],Y)
    p=np.linspace(0,1)
    q=np.linspace(0,1)
    P,Q=np.meshgrid(p,q)
    R=np.hstack((P.ravel().reshape(2500,1),Q.ravel().reshape(2500,1)))
    Z=net.inference(R)
    Z=Z.reshape(50,50)
    P_new=P*reader.X_norm[0,1]+reader.X_norm[0,0]
    Q_new=Q*reader.X_norm[1,1]+reader.X_norm[1,0]
    Z_new=Z*reader.Y_norm[0,1]+reader.Y_norm[0,0]
    ax.plot_surface(P_new,Q_new,Z_new,cmap="plasma",alpha=0.5)
    plt.show()

if __name__=="__main__":
    sdr=DataReader_1()
    sdr.ReadData()
    hp=HyperParameters_1(2,1,eta=0.1,max_epoch=100,batch_size=10,eps=1-6)
    net=NeuralNet_1(hp)
    sdr.NormalizeX()
    sdr.NormalizeY()
    net.train(sdr,checkpoint=0.1)
    ShowResult(net,sdr)
