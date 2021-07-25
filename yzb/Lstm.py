# -*- coding: utf-8 -*-
"""

@author: YZB
"""
import torch
import torch.nn as nn
import glob
from torch.utils.data import Dataset
import numpy as np
N=350#固定输入为350帧，不足的重复
label={0:'000',1:'001',2:'002',3:'003',4:'004'}
#([1, 3, 116, 17, 2])

def train_transform(data):
    data = torch.tensor(data)
    data = data.squeeze()
    data = torch.cat((data[0,:,:,:],data[2,:,:,:]),1)
    #print(data.size())
    data = torch.squeeze(data[:,:,0])
    #print(data.size())
    gap = N-data.size(0)
    while(gap>0):
        data = torch.cat((data[:,:],data[:gap,:]),0)
        gap = N-data.size(0)
    #if np.random.random()<0.7:
        #data==torch.randn(N,34)/100+data
    #print(data.size())
    
    return data


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


class train_dataset(Dataset):
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
        data = train_transform(data)
        #print('data:',self.data_list[item])
        #print(label)
        return data, label
 
    def __len__(self):
        return len(self.data_list)
    
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
        #print('data:',self.data_list[item])
        #print(label)
        return data, label
 
    def __len__(self):
        return len(self.data_list)    

    
########################################## 
    
class LSTM(nn.Module):
    def __init__(self,k=N,num_classes=5,input_size=34,hidden_size=34,num_layers=2,
                 bidirectional=False,batch_first=True,dropout=0.3):
        super(LSTM,self).__init__()
        self.LSTM = nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                            num_layers=num_layers,bidirectional=bidirectional,dropout=dropout,batch_first=batch_first)
        self.fc1=nn.Linear(k*hidden_size*(1+int(bidirectional)),num_classes)
        

    def forward(self,x):
        batch_size = x.size(0)
        output,(_,_) = self.LSTM(x)
        y = output.reshape(batch_size,-1)
        y = self.fc1(y)
        return y
    
##########################################




if __name__ == '__main__':
    path='./data'
    
    train_dataset = train_dataset(path,'train')
    test_dataset = test_dataset(path, 'test')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 1, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 1)
    
    net = LSTM()
    net = net.cuda()

    print("Training")
    net.train()
    epoch_num=30
    learning_rate=1e-4
    celoss=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    for i in range(epoch_num):
        num_right=0
        for x,y in train_loader:
            x=x.cuda()
            x = x.float()
            #print('x:',x)
            y=torch.tensor(y)
            y=y.cuda()
            #print('y:',y)
            optimizer.zero_grad()
            pre=net(x)
            
            loss=celoss(pre,y)
            loss.backward()
            optimizer.step()
            pre = pre.cpu().detach().numpy()
            idx=np.argmax(pre)
            if(idx==y):
                num_right=num_right+1
        print('Epoch:',i,' MSELoss:',loss)
        print('Epoch:',i,' Accuracy:',num_right/train_dataset.__len__(),'%')

    print('Testing')
    net.eval()

    test_loss=[]
    num_right=0
    for test_x,test_y in test_loader:
        test_x=test_x.cuda()
        test_x = test_x.float()
        test_y=torch.tensor(test_y)
        test_y=test_y.cuda()
        test_pre=net(test_x)
        loss=celoss(test_pre,test_y)
        test_loss.append(loss.item())
        test_pre = test_pre.cpu().detach().numpy()
        idx=np.argmax(test_pre)
        if(idx==test_y):
            num_right=num_right+1
    print('test_Loss:',np.mean(test_loss))
    print('Accuracy:',num_right/test_dataset.__len__())
    torch.save(net,'LSTM_0_30_34_2.pth')
