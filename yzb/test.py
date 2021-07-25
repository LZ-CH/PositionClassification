# -*- coding: utf-8 -*-
"""
本文件用于展示各个模型的准确率

@author: YZB
"""

import torch
import glob
import numpy as np
import torch.nn as nn
from Lstm import LSTM
from torch.utils.data import Dataset

#############load test_data#############
N=350
label={0:'000',1:'001',2:'002',3:'003',4:'004'}
def test_transform(data):
    data = torch.tensor(data)
    data = data.squeeze()
    data = torch.cat((data[0,:,:,:],data[2,:,:,:]),1)
    data = torch.squeeze(data[:,:,0])
    gap = N-data.size(0)
    while(gap>0):
        data = torch.cat((data[:,:],data[:gap,:]),0)
        gap = N-data.size(0)
    return data


class test_dataset(Dataset):
    def __init__(self, store_path, split):
        self.store_path = store_path
        self.split = split
        self.data_list = []
        self.label_list = []
        for i in range(5) :
            #print(type(label[i]))
            for file in glob.glob(self.store_path + '/' + split + '/' + label[i] + '/*.npy'):
                cur_path = file.replace('\\', '/')
                cur_label = i
                self.data_list.append(cur_path)
                self.label_list.append([cur_label])
 
    def __getitem__(self, item):
        data = np.load(self.data_list[item])
        label = self.label_list[item]
        data = test_transform(data)
        return data, label
 
    def __len__(self):
        return len(self.data_list)    

path='./data'
test_dataset = test_dataset(path, 'test')
test_loader = torch.utils.data.DataLoader(test_dataset, 1)


#############################################
num1=0
num2=0
num3=0


net1 = torch.load('LSTM_0_30_68_4.pth')
net2 = torch.load('LSTM_03_30_68_4.pth')
net3 = torch.load('LSTM_07_30_68_4.pth')


net1.eval()
net2.eval()
net3.eval()

print("网络参数：hidden_size=68,num_layers=4,bidirectional=True,dropout=0.5")
for test_x,test_y in test_loader:
        test_x=test_x.cuda()
        test_x = test_x.float()
        test_y=torch.tensor(test_y)
        test_y=test_y.cuda()
        
        pre1=net1(test_x)
        pre2=net2(test_x)
        pre3=net3(test_x)
        
        pre1 = pre1.cpu().detach().numpy()
        pre2 = pre2.cpu().detach().numpy()
        pre3 = pre3.cpu().detach().numpy()
        
        idx1=np.argmax(pre1)
        idx2=np.argmax(pre2)
        idx3=np.argmax(pre3)
        
        if(idx1==test_y):
            num1=num1+1
        if(idx2==test_y):
            num2=num2+1
        if(idx3==test_y):
            num3=num3+1
          

print('Accuracy_LSTM_0_30_68_4:',num1/test_dataset.__len__())
print('Accuracy_LSTM_03_30_68_4:',num2/test_dataset.__len__())
print('Accuracy_LSTM_07_30_68_4:',num3/test_dataset.__len__())

num1=0
num2=0
num3=0

net1 = torch.load('LSTM_0_30_34_2.pth')
net2 = torch.load('LSTM_03_30_34_2.pth')
net3 = torch.load('LSTM_07_30_34_2.pth')


net1.eval()
net2.eval()
net3.eval()

print("网络参数：hidden_size=30,num_layers=2,bidirectional=True,dropout=0.5")
for test_x,test_y in test_loader:
        test_x=test_x.cuda()
        test_x = test_x.float()
        test_y=torch.tensor(test_y)
        test_y=test_y.cuda()
        
        pre1=net1(test_x)
        pre2=net2(test_x)
        pre3=net3(test_x)
        
        pre1 = pre1.cpu().detach().numpy()
        pre2 = pre2.cpu().detach().numpy()
        pre3 = pre3.cpu().detach().numpy()
        
        idx1=np.argmax(pre1)
        idx2=np.argmax(pre2)
        idx3=np.argmax(pre3)
        
        if(idx1==test_y):
            num1=num1+1
        if(idx2==test_y):
            num2=num2+1
        if(idx3==test_y):
            num3=num3+1
          

print('Accuracy_LSTM_0_30_32_2:',num1/test_dataset.__len__())
print('Accuracy_LSTM_03_30_32_2:',num2/test_dataset.__len__())
print('Accuracy_LSTM_07_30_32_2:',num3/test_dataset.__len__())


    