#By Luojingzhou

import numpy as np
import os
from glob import glob
import torch
import cv2

def nomalxy(data):
    #preprocess
    #shape:(1,3,fps,17,2)-->(2,fps,17) because the dataset only have one person.
    data=np.delete(data,1,axis=1)
    data=np.delete(data,1,axis=4)
    xmin=np.min(data[:,0,:,:,:])
    xmax=np.max(data[:,0,:,:,:])
    ymin=np.min(data[:,1,:,:,:])
    ymax=np.max(data[:,1,:,:,:])
    xlen=xmax-xmin
    ylen=ymax-ymin
    # To 0~1
    data[:,0,:,:,:]=(data[:,0,:,:,:]-xmin)/xlen
    data[:,1,:,:,:]=(data[:,1,:,:,:]-ymin)/ylen
    data=data.squeeze()
    return data

def trans_data(data,flip=False):
    #shape:(1,3,fps,17,2)-->(fps,34): [[x0,y0,x1,y1,...],...]

    data=nomalxy(data)#-->(2,fps,17)
    newdata=np.zeros((data.shape[1],34))
    if flip:
        data[0,:,:]=1-data[0,:,:]
    for i in range(17):
        newdata[:,2*i]=data[0,:,i]
        newdata[:,2*i+1]=data[1,:,i]

    return newdata

def load_data(rootdata,label,k):
    #input: data.shape=(1,3,fps,17,2)
    #output: newdata is a list, newdata[i].shape=(fps//k,34),newdata[i] is a numpy;
    #newlabel is a list conclude k same labels
    data=trans_data(rootdata)#data.shape=(fps,34)
    step=data.shape[0]//k
    newdata=[]
    newlabel=[]     
    for i in range (step):
        newdata.append((data[i:i+step*k:step,:]).tolist())
        newlabel.append(label)
    return newdata,newlabel

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,path,k_spilt=48,mode='train'):
        dir_list=glob(path+'/*')
        dir_list.sort()
        self.x=[]
        self.labels=[]
        self.mode=mode
        self.k_spilt=k_spilt
        for label, i in enumerate(dir_list):#获取标签与对应numpy文件
            npy_list=glob(i+'/*.npy')
            for j in npy_list:
                eachdata=np.load(j)
                addx,addlabels=load_data(eachdata,label,k_spilt)
                self.x+=addx
                self.labels+=addlabels
    def __getitem__(self, index):
        a,b=torch.tensor(self.x[index]).float(),torch.tensor(self.labels[index]).long()
        if self.mode=='train':#用于数据增强:坐标随机偏移
            if np.random.random()<0.3:
                a=torch.randn(self.k_spilt,34)/100+a
        return a,b
    
    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    dataset=MyDataset('./data/train')
    for i in range(len(dataset)):
        print (dataset[i][0].shape)