# PositionClassification
人体姿态序列分类任务大作业
## Requirements
所需环境配置
1. Python 3.8 
2. Pytorch 1.7.1
3. numpy 1.20.2
4. opencv
5. matplotlib
6. torchvision 0.8.2
7. cuda101

### Folder structure
Download the PositionClassification first.(下载整个项目源程序).
The following shows the basic folder structure.(项目文件夹应按以下放置).
```

├── data
│   ├── test
│   └── train
├── test.py # testing code
├── train.py # training code
├── model.py # network
├── MyDataset.py
├── result
│   ├── final_weight.pth #  A pre-trained snapshot (final_weight.pth)
```
### Test: 
1) Download the pre-trained snapshot.(下载训练好的模型文件)<a href="https://pan.baidu.com/s/1MNGYunvonFvSpZJLGCu19Q">baidu cloud [password: 1234]</a>
2) Prepare the test data.(下载测试集)<a href="https://pan.baidu.com/s/1gfiTziz4RCRHImRrG-EIPw">baidu cloud [password: 1234]</a>
3) cd PositionClassification.(工作空间设置到该项目的根目录)
```
python test.py 
```

### Train: 
1) Prepare the train and test data.(下载训练集与测试集)<a href="https://pan.baidu.com/s/1gfiTziz4RCRHImRrG-EIPw">baidu cloud [password: 1234]</a>
2) cd PositionClassification(工作空间设置到该项目的根目录)
```
python train.py 
