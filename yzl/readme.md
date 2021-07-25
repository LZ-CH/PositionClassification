# 人体姿态序列分类任务大作业（一维卷积）
by 叶泽林19351163
## 环境
* python 3.7
* tensorflow 1.15
## 文件夹架构
```
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
## 训练
将数据集放入data文件夹中，在```main.py```中运行：
```
train_cost, test_cost, train_acc, test_acc = train(epoch=100, batch_size=512, fix_len=32, lr=1e-5)
```
即可得到训练集与测试集的损失与正确率随训练的变化，最佳模型将会保存至model文件夹下。
## 测试
下载预训练模型并解压至model文件夹中，在```main.py```中运行：
```
acc = evaluate(model_path='./model/best.ckpt', data_path='./data/test/')
```
即可得到测试准确率。
预训练模型下载链接```https://pan.baidu.com/s/1FqBvf6KzVU9y3e7t_LH5Hg```, 提取码```gx82```。
## 效果
最高测试准确率为83.76%。
