# MS_ai_edu_homework
以上两个文件夹分别对应两道题目。</br>
## 1 线性回归模型</br>
### 任务描述</br>
给定含有1000条记录的数据集mlm.csv，其中每条记录均包含两个自变量x,y和一个因变量z，它们之间存在较为明显的线性关系。</br>
**请对数据进行三维可视化分析，并训练出良好的线性回归模型。**</br>
### 实现方法</br>
依照教程手工搭建神经网络</br>
### 提交文件说明</br>
文件夹中linearRegression.py为主程序。数据读入和处理、超参构建、神经网络前向计算反向传播等在帮助类子目录中。</br>
### 运行结果</br>
三维可视化结果</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Mlm/resultMlm.png)
## 2 非线性多分类器</br>
### 任务描述</br>
鸢尾花数据集iris.csv含有150条记录，每条记录包含萼片长度sepal length、萼片宽度sepal width、 花瓣长度petal length和花瓣宽度petal width四个数值型特征，以及它的所属类别class（可能为Iris-setosa,Iris-versicolor,Iris-virginica三者之一）。</br>
**请利用该数据集训练出一个良好的非线性分类器。**</br>
### 实现方法</br>
依照教程手工搭建神经网络</br>
采用双层神经网络，隐层3个神经元。学习率为0.1，共训练5000代，批大小为5。br>
检验集和测试集由训练集随机打乱顺序后分别抽取10%构成。</br>
### 提交文件说明</br>
文件夹中nonLinearMultipleClassification.py为主程序。</br>
每次调参运行
数据读入和处理、超参构建、神经网络前向计算反向传播等在帮助类子目录中。</br>
### 运行结果</br>
训练过程中loss与accuracy的记录</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Iris/lossAndAccuracy.png)
一次测试的正确率</br>
![avatar](https://github.com/seeeagull/MS_ai_edu_homework/blob/main/Iris/result.png)
每次测试的正确率基本稳定在93.33%和100.00%间</br>
