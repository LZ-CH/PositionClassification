## ljz项目部分

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