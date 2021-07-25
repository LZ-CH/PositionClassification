# coding = utf-8
import torch
import numpy as np
import time
from glob import glob
import torch.nn.functional as F
import matplotlib.pyplot as plt
import MyFunction
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
# torch.manual_seed(1)

import os
import torch.optim as optim
import torch.nn as nn
from tool import graph


traindataset = 'data/train' # 数据集路径D:\NET\data\train


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# GCN
num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward = [
    (10, 8), (8, 6), (9, 7), (7, 5), # arms
    (15, 13), (13, 11), (16, 14), (14, 12), # legs
    (11, 5), (12, 6), (11, 12), (5, 6), # torso
    (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2) # nose, eyes and ears
]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward
A = graph.get_adjacency_matrix(neighbor, num_node)
A_edge = graph.edge2mat(self_link, num_node)
graph_data = A + A_edge
graph_data = torch.tensor(graph_data, dtype=torch.float)
degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  #[N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0
degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵
final_data = torch.mm(degree_matrix, graph_data)
grf = torch.zeros([10, num_node, num_node])
for i in range(10):
    grf[i, :, :] = final_data
print(grf.shape)

class Dataset(Dataset):
    def __init__(self, path):
        self.npypath_list = []
        self.npylabel_list = []
        for i in range(5):
            self.npypath_list.extend(glob(path+'/00'+str(i)+'/*.npy'))
        for n, path in enumerate(self.npypath_list):
            label = path[path.rfind("/00"):][3]
            self.npylabel_list.extend(label)
        self.transforms = transform
    def __len__(self):
        return len(self.npypath_list)
    def __getitem__(self, index):
        np_data = np.load(self.npypath_list[index])
        np_data = np_data[:, :, :, :, 0] # (1, 3, seq, 17)
        np_data = np_data[0, :, :, :]  # (3, seq, 17)
        np_data = np.delete(np_data, 1, 0) # (2, seq, 17)
        len = np_data.shape[1]
        step_size = len // 48
        remainder = len % 48
        data = np.zeros([1, len, 34])
        for i in range(17): # data.shape (2, seq, 34)
            data[:, :, i] = np_data[0, :, i]
            data[:, :, i+17] = np_data[1, :, i]
        start_step = np.random.rand() * remainder # 随机初始位置 < 余数
        start_step = np.int(start_step)  # 取整
        trs_data = np.zeros([1, 48, 34])
        grf_data = np.zeros([2, 48, 17])
        for k in range(48):
            trs_data[:, k, :] = data[:, (start_step+k*step_size), :] # (1, 48, 34)
            grf_data[:, k, :] = np_data[:, (start_step+k*step_size), :] # (2, 48, 17)
        trs_data = self.transforms(trs_data)
        trs_data = torch.transpose(trs_data, 0, 1)
        trs_data = torch.transpose(trs_data, 1, 2)
        grf_data = self.transforms(grf_data)
        grf_data = torch.transpose(grf_data, 0, 1)
        grf_data = torch.transpose(grf_data, 1, 2)
        label = self.npylabel_list[index]
        label = int(label)
        label_ts = torch.tensor(label, dtype=torch.long)
        return trs_data, grf_data, label_ts

train_Dataset=Dataset(traindataset)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gcn = nn.Linear(17, 17)
        self.trans = nn.TransformerEncoderLayer(d_model=34, nhead=17, dropout=0.1, dim_feedforward=256, activation='relu')
        self.lstm = nn.LSTM(34, 17, 2)
        self.gcn_lstm = nn.LSTM(34, 17, 2)
        self.fc1 = nn.Linear(34*48, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, trs_x, grf_x, grf_fig):
        # GCN + LSTM
        bat = grf_x.shape[0]
        grf_x = grf_x.transpose(0, 1)
        grf_x = self.gcn(grf_x)
        grf_x = F.relu(torch.matmul(grf_x, grf_fig))
        temp_x = torch.zeros([bat, 48, 34])
        for i in range(17): # (batch, 48, 34)
            temp_x[:, :, i] = grf_x[0, :, :, i]
            temp_x[:, :, i+17] = grf_x[1, :, :, i]
        grf_x = temp_x
        grf_x = grf_x.transpose(0, 1)
        grf_x, (hn, cn) = self.gcn_lstm(grf_x) # (48, batch, 17)

        # transformerEncoder + LSTM
        trs_x = MyFunction.positional_encoding(trs_x, 34)
        trs_x = trs_x.transpose(0, 1)
        trs_x = self.trans(trs_x)
        trs_x, (hn, cn) = self.lstm(trs_x) # (batch, 48, 17)
        # FCNN
        grf_x = grf_x.transpose(0, 1)
        trs_x = trs_x.transpose(0, 1)
        x = torch.cat((trs_x, grf_x), dim=2)
        b, s, h = x.shape
        x = x.reshape(b, s*h)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x) # (batch, output)
        return x

lstm_model = Model(input_size=17, hidden_size=17, output_size=5)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_model.parameters())

# ______________ train ________________________
trainloader = DataLoader(train_Dataset, batch_size=10, shuffle=True)
max_epoch = 40
max_shape = 0
loss_his = []
for epoch in range(max_epoch):
    print('epoch=', epoch)
    for i, data in enumerate(trainloader):
        trs_inputs, grf_inputs, labels = data
        labels = torch.tensor(labels, dtype=torch.long)
        trs_inputs = trs_inputs[:, 0, :, :]
        trs_inputs = torch.tensor(trs_inputs, dtype=torch.float32)
        grf_inputs = torch.tensor(grf_inputs, dtype=torch.float32)
        output = lstm_model(trs_inputs, grf_inputs, grf)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())
        loss_his.append(loss.item())

# ______________ test ____________________
testdataset = 'data/test' # 数据集路径D:\NET\data\train
test_Dataset=Dataset(testdataset)
testloader = DataLoader(test_Dataset, batch_size=1, shuffle=True)
correct = 0
sum = 0
for i, data in enumerate(testloader):
    sum += 1
    trs_inputs, grf_inputs, labels = data
    labels = torch.tensor(labels, dtype=torch.long)
    trs_inputs = trs_inputs[:, 0, :, :]
    trs_inputs = torch.tensor(trs_inputs, dtype=torch.float32)
    grf_inputs = torch.tensor(grf_inputs, dtype=torch.float32)
    output = lstm_model(trs_inputs, grf_inputs, final_data)
    output = output[0, :]
    output = F.softmax(output, dim=0)
    pred = torch.max(output, 0)[1]
    correct += pred.eq(labels.view_as(pred)).sum().item()
accuracy = correct / sum
print('accuracy=', accuracy)

plt.figure()
plt.plot(loss_his)
plt.show()




# train_directory = os.path.join(dataset, 'train')
# test_directory = os.path.join(dataset, 'test')
# batch_size = 1 #更改
# num_classes = 5
#
# data = {  # 读取数据集
#     'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
#     'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
# }
#
# train_data_size = len(data['train'])
# test_data_size = len(data['test'])
#
# train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)  # DataLoader加载数据
# test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)
#
# print(train_data_size, test_data_size)  # 打印训练集和测试集的大小
#
# resnet18 = models.resnet18(pretrained=False)  # 加载ResNet18的预训练模型
#
# for param in resnet18.parameters():
#     param.requires_grad = True
#
# fc_inputs = resnet18.fc.in_features
# resnet18.fc = nn.Sequential(  # 更改网络结构，适应我们自己的分类需要
#     nn.Linear(fc_inputs, 256),
#     nn.ReLU(),
#     nn.Linear(256, 2)
# )
#
# resnet18 = resnet18.to('cuda:0')
#
# Loss_ = nn.CrossEntropyLoss()  # 定义交叉熵损失
# optimizer_ = optim.Adam(resnet18.parameters(), lr=1e-4)  # 定义Adam优化器
#
#
# def train(model, Loss, optimizer, epoches=50):  # 训练函数
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 调用GPU
#     history = []
#     best_acc = 0.0
#     best_epoch = 0
#
#     for epoch in range(epoches):  # 进入epoch循环
#         epoch_start = time.time()
#         print("Epoch:{}/{}".format(epoch + 1, epoches))
#         model.train()
#
#         train_loss = 0.0  # 记录训练误差及准确率信息
#         train_acc = 0.0
#         test_loss = 0.0
#         test_acc = 0.0
#
#         for i, (inputs, labels) in enumerate(train_data):  # 训练部分
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()
#
#             outputs = model(inputs)  # 前向传播
#             loss = Loss(outputs, labels)  # 计算损失
#             loss.backward()  # 反向传播
#             optimizer.step()
#             train_loss += loss.item() * inputs.size(0)
#
#             ret, predictions = torch.max(outputs.data, 1)
#             correct_counts = predictions.eq(labels.data.view_as(predictions))
#
#             acc = torch.mean(correct_counts.type(torch.FloatTensor))
#
#             train_acc += acc.item() * inputs.size(0)
#
#         with torch.no_grad():
#             model.eval()
#
#             for j, (inputs, labels) in enumerate(test_data):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 outputs = model(inputs)
#                 loss = Loss(outputs, labels)
#                 test_loss += loss.item() * inputs.size(0)
#                 ret, predictions = torch.max(outputs.data, 1)
#                 correct_counts = predictions.eq(labels.data.view_as(predictions))
#                 acc = torch.mean(correct_counts.type(torch.FloatTensor))
#
#                 test_acc += acc.item() * inputs.size(0)
#
#         avg_train_loss = train_loss / train_data_size
#         avg_train_acc = train_acc / train_data_size
#
#         avg_test_loss = test_loss / test_data_size
#         avg_test_acc = test_acc / test_data_size
#
#         history.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])
#
#         if best_acc < avg_test_acc:
#             best_acc = avg_test_acc
#             best_epoch = epoch + 1
#
#         epoch_end = time.time()
#
#         print(
#             "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tTesting: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
#                 epoch + 1, avg_train_loss, avg_train_acc * 100, avg_test_loss, avg_test_acc * 100,
#                 epoch_end - epoch_start
#             ))
#         print("Best Accuracy for test : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
#
#         torch.save(model, 'D:/PyCharm/ResNet/models_2class/' + '_model_' + str(epoch + 1) + '.pt')
#     return model
#
#
# train(resnet18, Loss_, optimizer_, 50)
