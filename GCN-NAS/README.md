# GCN-NAS

by 张靖宜19351171

## Requirements
- python packages
  - pytorch = 0.4.1
  - torchvision>=0.2.1
  - some other basic packages
  

## Files

  config/train_joint.yaml              配置文件
  data/                                             存放预处理后的数据及标签信息等
  graph/kinetics.py、tools.py     由关节数据生成所需要的图
  model/agcn3.py                          网络搭建
  feeders/feeder.py                       对输入数据进行处理
  main.py                                        训练过程调用函数
  runs/                                             存放模型 

## Model Training 

`python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`



