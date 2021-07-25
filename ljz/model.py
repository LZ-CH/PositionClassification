#By Luojingzhou

import torch
import torch.nn as nn
from torchvision import models


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18,self).__init__()
        self.resnet18 = models.resnet18()
        # 修改最后一层全连接层
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 5)
        # 修改第一层卷积层
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    def forward(self,x):        
        x=x.unsqueeze(1)
        y=self.resnet18(x)
        return y




class LSTM_FC(nn.Module):
    def __init__(self,k_spilt=48,numclass=5,input_size=34, hidden_size=68, num_layers=2,
                    bidirectional=True,batch_first=True,dropout=0.5):
        super(LSTM_FC, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                                batch_first=batch_first,bidirectional=bidirectional,dropout=dropout)
        self.lstmoutdim=k_spilt*hidden_size*(1+int(bidirectional))#根据LSTM参数计算输出大小
        self.fc1 = nn.Linear(self.lstmoutdim, numclass)

    def forward(self, x):
        batch_size = x.shape[0]
        lstm_output,(_,_)= self.lstm(x)
        y = lstm_output.reshape(batch_size, -1)
        y = self.fc1(y)
        return y




if __name__ == '__main__':
    net=resnet18()
    x=torch.randn(2,48,34)
    y=net(x)
    print(y.shape)
