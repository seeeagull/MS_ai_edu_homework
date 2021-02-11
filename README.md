# MS_ai_edu_homework
Mlm文件夹完成线性回归任务。</br>
Iris与Iris_DNN文件夹均完成非线性多分类任务。</br>
# 任务报告
## 1 线性回归模型</br>
### 任务描述</br>
给定含有1000条记录的数据集mlm.csv，其中每条记录均包含两个自变量x,y和一个因变量z，它们之间存在较为明显的线性关系。</br>
**请对数据进行三维可视化分析，并训练出良好的线性回归模型。**</br>
### 实现方法</br>
依照教程手工搭建神经网络</br>
### 提交文件说明</br>
* linearRegression - 主程序。
* 帮助类子目录
  1. DataReader_1 - 数据读取类，从文件中读入数据、标准化、输出数据、随机打乱顺序。</br>
  2. NeuralNet_1 - 神经网络类，初始化、正向、反向、更新、训练、验证、测试等一系列方法。</br>
  3. HyperParameters_1 - 超参数类，初始化神经网络的各种超参数。</br>
### 运行结果</br>
三维可视化结果：</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Mlm/resultMlm.png)</br>
## 2 非线性多分类器</br>
### 任务描述</br>
鸢尾花数据集iris.csv含有150条记录，每条记录包含萼片长度sepal length、萼片宽度sepal width、 花瓣长度petal length和花瓣宽度petal width四个数值型特征，以及它的所属类别class（可能为Iris-setosa,Iris-versicolor,Iris-virginica三者之一）。</br>
**请利用该数据集训练出一个良好的非线性分类器。**</br>
### 实现方法1.0</br>
依照教程手工搭建神经网络</br>
采用双层神经网络，隐层3个神经元。</br>
学习率为0.1，训练代数为5000，批大小为5。</br>
初始化权重矩阵方法为Xavier。</br>
检验集和测试集由训练集随机打乱顺序后分别抽取10%构成。</br>
### 提交文件说明1.0</br>
* nonLinearMultipleClassificationn - 主程序。</br>
* 帮助类子目录
  1. DataReader_2 - 数据读取类，从文件中读入数据、标准化、生成检验集和测试集、输出数据、随机打乱顺序。</br>
  2. NeuralNet_2 - 神经网络类，初始化、正向、反向、更新、训练、验证、测试等一系列方法。</br>
  3. HyperParameters_2 - 超参数类，神经网络的各种超参数。</br>
  4. LossFunction_2 - 损失函数类，使用交叉熵函数。</br>
  5. ActivatorFunction_2 - 激活函数类，使用Sigmoid函数。</br>
  6. ClassifierFunction_2 - 分类函数类，使用Softmax函数。</br>
  7. TrainingHistory_2 - 训练记录类，记录训练过程中的损失函数值、验证精度，并绘制图像。</br>
  8. WeightsBias_2 - 权重矩阵与偏置矩阵类，初始化、加载数据、保存数据。</br>
  9. EnumDef_2 - 枚举类，枚举了不同初始化方法对应的编号。</br>
### 运行结果1.0</br>
训练过程中loss与accuracy的记录：</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Iris/lossAndAccuracy.png)</br>
某一次测试的正确率：</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Iris/result.png)</br>
多次试验表明测试的正确率基本稳定在93.33%和100.00%间。但偶尔也有出现正确率较低的情况，不是非常稳定。</br>
上图展现了一次较为理想的实验结果。</br></br>
### 实现方法2.0</br>
在跟随教程学完DNN与CNN部分后，尝试手工搭建简易框架，并利用框架构建多层神经网络。</br>
网络结构如下。</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Iris_ONNX/onnxmodel.png)</br>
学习率为4e-5，训练代数为2500，批大小为5。</br>
初始化权重矩阵方法为Xavier，优化方法为Adam。</br>
检验集和测试集由训练集随机打乱顺序后分别抽取10%构成。</br>
### 提交文件说明2.0</br>
* iris_DNN - 主程序。</br>
部分程序段解释：</br>
读入数据，标准化，生成检验集和测试集。</br>
```
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
```
</br>设置超参，搭建神经网络。</br>
```
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
    
    f3=FcLayer(f1.output_size,8,params)
    net.add_layer(f3,"f3")
    r3=ActivationLayer(Relu())
    net.add_layer(r3,"r3")

    f4=FcLayer(f3.output_size,num_output,params)
    net.add_layer(f4,"f4")
    s4=ClassificationLayer(Softmax())
    net.add_layer(s4,"s4")
    
    return net
```
</br>
* 迷你框架子目录
  1. ActivatorFunction_3 - 激活层，包含Identity,Sigmoid,Tahn,Relu。</br>
  2. BatchNormLayer_3 - Bn层，进行批量归一化操作。</br>
  3. ClassifierFunction_3 - 分类层，包含Logistic,Softmax。</br>
  4. ConvLayer_3 - 卷积层，前向计算和反向传播分别有numba和im2col方法实现。</br>
  5. ConvWeightsBias_3 - 卷积层权重矩阵与偏移矩阵，含有卷积层特有的权重矩阵翻转180°操作。</br>
  6. DataReader_3 - 数据读取类，从文件中读入数据、标准化、生成检验集和测试集、输出数据、随机打乱顺序。</br>
  7. DropoutLayer_3 - 丢弃层，防止过拟合的一种方法。。</br>
  8. EnumDef_3 - 枚举类，包含网络类型、初始化方法、损失函数图像横坐标、优化方法、训练早停条件、正则化方法、池化类型。</br>
  9. FcLayer_3 - 全连接层。</br>
  10. HyperParameters_3 - 超参数类，学习率、训练代数、批大小、初始化矩阵方法、优化方法等。</br>
  11. Layer_3 - 层类，各个层的父类，规定了各层包含的操作有初始化、训练、预更新、更新、存储参数、加载参数。</br>
  12. LossFunction_3 - 损失函数类，含有MSE,CS2,CS3。</br>
  13. NeuralNet_3 - 神经网络类，初始化、正向、反向、更新、训练、验证、测试等一系列方法。</br>
  14. Optimizer_3 - 优化器类，包含SGD,Momentum,Nag,AdaGrad,AdaDelta,RMSProp,Adam。</br>
  15. PoolingLayer_3 - 池化层，包含最大值和平均值两种方法。</br>
  16. TrainingHistory_3 - 训练记录类，记录训练过程中的损失函数值、验证精度，并绘制图像。</br>
  17. WeightsBias_3 - 权重矩阵与偏移矩阵类，初始化、生成优化、更新、加载数据、保存数据。</br>
  18. jit_utility - 卷积层一系列运算的具体实现过程，包含numba和im2col两种方法。</br>
### 运行结果2.0</br>
训练过程中loss与accuracy的记录：</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Iris_DNN/LossAndAccuracy.png)</br>
某一次测试的正确率：</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Iris_DNN/Running.png)</br>
正确率较高且每次实验结果较稳定。</br>
但考虑到数据集中仅有150条数据，使用该复杂程度的网络结构也许有已经过拟合的可能？</br>
