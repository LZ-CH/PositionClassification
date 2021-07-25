#By Luojingzhou

import torch
import numpy as np
import random
from model import LSTM_FC,resnet18
from MyDataset import MyDataset,load_data
import os
from glob import glob
import matplotlib.pyplot as plt
from test import test

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

#获取准确率
def get_acc(pre,label):
    pre_label=torch.argmax(pre,dim=1)
    num=pre_label.shape[0]
    acc=torch.sum((pre_label==label)).item()/num
    return acc

#training parameters:
gpu_device='0'#GPU型号
trainpath='./data/train'#训练数据路径
testpath='./data/test'#测试数据路径
batchsize=48
result_dir='./result'#结果保存路径
epoch_num=200
learning_rate=1e-4
weight_decay=1e-4
k_spilt=48


best_epoch=0
best_acc=0
best_acc_group=0
best_acc_beforevote=0

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

os.environ['CUDA_VISIBLE_DEVICES']=gpu_device#设置可用GPU
use_cuda=torch.cuda.is_available#是否可用cuda加速

#载入数据
train_dataset = MyDataset(trainpath,k_spilt=k_spilt)
test_dataset = MyDataset(testpath,k_spilt=k_spilt,mode ='test')
train_loader = torch.utils.data.DataLoader(train_dataset, batchsize, shuffle=True, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batchsize)
#模型选择
net=resnet18()
# net=LSTM_FC(k_spilt=k_spilt) #若要检验LSTM,需将此行代码取消注释

train_epoch_each,train_acc_each=[],[]
test_epoch_each,test_acc_each=[],[]    
if use_cuda:
    net=torch.nn.DataParallel(net).cuda()
print("Training")
#模型训练
CEloss=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    net.parameters(), lr=learning_rate, weight_decay=weight_decay)
net.train()
for i in range(epoch_num):
    acc_list=[]
    testacc_list=[]
    loss_list=[]
    for x,label in train_loader:
        if use_cuda:
            x=x.cuda()
            label=label.cuda()
        pre=net(x)
        optimizer.zero_grad()
        loss=CEloss(pre,label)
        loss_list.append(loss.item())
        acc_list.append(get_acc(pre,label))
        loss.backward()
        optimizer.step()
    print('Epoch:',i,' CELoss:',np.mean(loss_list),'trainacc:',np.mean(acc_list))
    train_epoch_each.append(i)
    train_acc_each.append(np.mean(acc_list).item())
    #Test普通测试
    net.eval()
    for x,label in test_loader:
        if use_cuda:
            x=x.cuda()
            label=label.cuda()
        pre=net(x)
        testacc_list.append(get_acc(pre,label))
    print('Epoch:',i,' testacc:',np.mean(testacc_list))

    #voting test优化投票融合测试
    test_acc,group_acc=test(net,testpath,k_spilt=k_spilt)
    test_epoch_each.append(i)
    test_acc_each.append(test_acc)
    if best_acc<test_acc:
        best_epoch=i
        best_acc=test_acc
        best_acc_group=group_acc
        best_acc_beforevote=np.mean(testacc_list)
        torch.save(net.state_dict(),result_dir+'/final_weight.pth')
    net.train()
print('Best: epoch:',best_epoch,' acc_beforvote:',best_acc_beforevote,' acc:',best_acc,' acc_group:',best_acc_group)


#plot训练过程可视化
acc_result=plt.figure()
plt.title('Acc of each epoch.(best acc: '+str(round(best_acc,3))+')')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(train_epoch_each,train_acc_each,color='r',label='Train')
plt.plot(test_epoch_each,test_acc_each,color='b',label='Test')
plt.legend()
acc_result.savefig(result_dir+'/Acc_result.png')
