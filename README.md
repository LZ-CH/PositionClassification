# PositionClassification
人体姿态序列分类任务大作业

此项目包含小组五个成员共5个子项目(对应文件夹分别为ljz,GCN-NAS,human_bps_classification,yzl,yzb)，每一个子项目均可独立完成此次的人体姿态序列分类任务
## 一、ljz项目部分:

by 罗经周19351098

包含:使用ResNet18与LSTM完成此分类任务的所有源程序(默认使用ResNet18，若需检验此部分的LSTM效果,只需将train.py的第57行代码与test.py的第46行代码取消注释)

效果:使用ResNet18准确率为86.32%,使用LSTM的准确率为85.47%
### Requirements
所需环境配置
1. Python 3.8 
2. Pytorch 1.7.1
3. numpy 1.20.2
4. opencv
5. matplotlib
6. torchvision 0.8.2
7. cuda101

### Folder structure
下载整个ljz项目源程序:

该部分的项目文件夹应按以下结构放置:
```
ljz
├── data
│   ├── test
│   └── train
├── test.py # 测试代码
├── train.py # 训练代码
├── model.py # 网络定义代码
├── MyDataset.py
├── result
│   ├── final_weight.pth #  预训练模型 (final_weight.pth)
```
### Test: 
1) 下载训练好的模型文件。<a href="https://pan.baidu.com/s/1MNGYunvonFvSpZJLGCu19Q">baidu cloud [password: 1234]</a>
2) 下载测试集。<a href="https://pan.baidu.com/s/1gfiTziz4RCRHImRrG-EIPw">baidu cloud [password: 1234]</a>
3) cd PositionClassification/ljz.(工作空间设置到该项目的根目录)
4) 在命令行下运行以下代码
```
python test.py 
```

### Train: 
1) 下载训练集与测试集<a href="https://pan.baidu.com/s/1gfiTziz4RCRHImRrG-EIPw">baidu cloud [password: 1234]</a>
2) cd PositionClassification/ljz(工作空间设置到该项目的根目录)
3) 在命令行下运行以下代码
```
python train.py 
```
## 二、GCN-NAS项目部分:

by 张靖宜19351171
### Requirements
- python packages
  - pytorch = 0.4.1
  - torchvision>=0.2.1
  - some other basic packages
  

### Files

  config/train_joint.yaml              配置文件
  
  data/                                             存放预处理后的数据及标签信息等
  
  graph/kinetics.py、tools.py     由关节数据生成所需要的图
  
  model/agcn3.py                          网络搭建
  
  feeders/feeder.py                       对输入数据进行处理
  
  main.py                                        训练过程调用函数
  
  runs/                                             存放模型 

### Model Training 
`python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`

## 三、human_bps_classification项目部分:

by 聂云双
### Requirements
所需环境配置
1. Python 3.8 
2. Pytorch 1.7.1
3. numpy 1.19.2
4. glob2 0.7
5. matplotlib
6. torchvision 0.8.2

### Folder structure
Download the PositionClassification first.(下载整个项目源程序)

The following shows the basic folder structure.(项目文件夹应按以下结构放置)
```
human_bps_classification
├── data
│   ├── test
│   └── train
├── main.py # 网络定义、训练、测试（只需运行该文件）
├── Myfunction.py # 位置编码
├── tool
│   ├── graph.py
│   └── visualise.py
```
###train & test
Run main.py

## 四、yzl项目部分

by 叶泽林19351163
### 环境
* python 3.7
* tensorflow 1.15
### 文件夹架构
```
yzl
├── data
│   ├── test
│   └── train
├── model
│   ├── best.ckpt.data-00000-of-00001
│   ├── best.ckpt.index
│   ├── best.ckpt.meta
│   └── checkpoint
├── main.py 
└── model.py
```
### 训练
将数据集放入data文件夹中，在```main.py```中运行：
```
train_cost, test_cost, train_acc, test_acc = train(epoch=100, batch_size=512, fix_len=32, lr=1e-5)
```
即可得到训练集与测试集的损失与正确率随训练的变化，最佳模型将会保存至model文件夹下。
### 测试
下载预训练模型并解压至model文件夹中，在```main.py```中运行：
```
acc = evaluate(model_path='./model/best.ckpt', data_path='./data/test/')
```
即可得到测试准确率。
预训练模型下载链接```https://pan.baidu.com/s/1FqBvf6KzVU9y3e7t_LH5Hg```, 提取码```gx82```。
### 效果
最高测试准确率为83.76%。

## 五、yzb项目部分

by 杨振邦19351162

### Requirements
所需环境配置
1. Python 3.8 
2. Pytorch 1.7.1
3. numpy 1.20.2
4. opencv
5. cuda11.0


### Folder structure
Download the PositionClassification first.(下载整个项目源程序)

The following shows the basic folder structure.(项目文件夹应按以下结构放置)
```
yzb
├── data
│   ├── test
│   └── train
├── LSTM.py # 模型训练
├── test.py # 读取并展示模型效果
├── LSTM_0_30_68_4t.pth
├── LSTM_03_30_68_4.pth
├── LSTM_07_30_68_4.pth
├── LSTM_0_30_34_2t.pth
├── LSTM_03_30_34_2.pth
├── LSTM_07_30_34_2.pth
