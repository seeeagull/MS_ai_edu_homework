# MS_ai_edu_homework
以上两个文件夹分别对应两道题目。</br>
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
三维可视化结果</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Mlm/resultMlm.png)
## 2 非线性多分类器</br>
### 任务描述</br>
鸢尾花数据集iris.csv含有150条记录，每条记录包含萼片长度sepal length、萼片宽度sepal width、 花瓣长度petal length和花瓣宽度petal width四个数值型特征，以及它的所属类别class（可能为Iris-setosa,Iris-versicolor,Iris-virginica三者之一）。</br>
**请利用该数据集训练出一个良好的非线性分类器。**</br>
### 实现方法</br>
依照教程手工搭建神经网络</br>
采用双层神经网络，隐层3个神经元。</br>
学习率为0.1，训练代数为5000，批大小为5。</br>
初始化权重矩阵方法为Xavier。</br>
检验集和测试集由训练集随机打乱顺序后分别抽取10%构成。</br>
### 提交文件说明</br>
* nonLinearMultipleClassificationn - 主程序。
* 帮助类子目录
1. DataReader_2 - 数据读取类，从文件中读入数据、标准化、生成检验集和测试集、输出数据、随机打乱顺序。</br>
2. NeuralNet_2 - 神经网络类，初始化、正向、反向、更新、训练、验证、测试等一系列方法。</br>
3. HyperParameters_2 - 超参数类，初始化神经网络的各种超参数。</br>
4. LossFunction_2 - 损失函数类，使用交叉熵函数。</br>
5. ActivatorFunction_2 - 激活函数类，使用Sigmoid函数。</br>
6. ClassifierFunction_2 - 分类函数类，使用Softmax函数。</br>
7. TrainingHistory_2 - 训练记录类，记录训练过程中的损失函数值、验证精度，并绘制图像。</br>
8. WeightsBias_2 - 权重矩阵类，初始化、加载数据、保存数据。</br>
9. EnumDef_2 - 枚举类，枚举了不同初始化方法对应的编号。</br>
### 运行结果</br>
训练过程中loss与accuracy的记录：</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Iris/lossAndAccuracy.png)
某一次测试的正确率：</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Iris/result.png)
实践证明每次测试的正确率基本稳定在93.33%和100.00%间。</br>
