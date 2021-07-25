#By Luojingzhou

import torch
import numpy as np
import random
from model import LSTM_FC,resnet18
from MyDataset import MyDataset,load_data
import os
from glob import glob
import time
gpu_device='0'#可使用的GPU
os.environ['CUDA_VISIBLE_DEVICES']=gpu_device
use_cuda=torch.cuda.is_available
def test(net,path,k_spilt=48):
    #net 为测试网络,path为测试数据的目录,k_split为每份子数据包含的帧数
    net.eval()
    dir_list=glob(path+'/*')
    dir_list.sort()
    right_count=np.zeros(5)
    group_count=np.zeros(5)
    for label, i in enumerate(dir_list):
        npy_list=glob(i+'/*.npy')
        group_count[label]=len(npy_list)
        for j in npy_list:
            eachdata=np.load(j)
            addx,_=load_data(eachdata,label,k_spilt)
            x=np.array(addx)
            x=torch.tensor(x).float()
            if use_cuda:
                x=x.cuda()
            pre=net(x)
            pre=torch.sum(pre,dim=0)#融合多个分类结果
            pre_label=torch.argmax(pre)
            right_count[label]+=int(pre_label==label)
    group_acc=(right_count/group_count).tolist()#计算各类准确率
    test_acc=(np.sum(right_count)/np.sum(group_count)).item()#计算平均准确率
    print('test_acc:',test_acc,'group_acc:',group_acc)
    return  test_acc,group_acc

if __name__ == '__main__':

    testpath='./data/test'#
    final_weightpath='./result/final_weight.pth'

    net=resnet18()
    # net=LSTM_FC()#若要检验LSTM,需将此行代码取消注释
    if use_cuda:
        net=torch.nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(final_weightpath))
    t0=time.time()
    test(net,testpath)
    t1=time.time()
    dir_list=glob(testpath+'/*')
    dir_list.sort()
    file_count=0
    for i in dir_list:
            npy_list=glob(i+'/*.npy')
            file_count += len(npy_list)
    print('The speed of each file: ',(t1-t0)/file_count,' s')
