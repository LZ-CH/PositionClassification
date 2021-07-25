import torch
import torch.nn as nn
Tensor = torch.Tensor
def positional_encoding(X, num_features, dropout_p=0.1, max_len=48) -> Tensor:
    r'''
        给输入加入位置编码
    参数：
        - num_features: 输入进来的维度
        - dropout_p: dropout的概率，当其为非零时执行dropout
        - max_len: 句子的最大长度，默认512

    形状：
        - 输入： [batch_size, seq_length, num_features]
        - 输出： [batch_size, seq_length, num_features]

    例子：
        - X = torch.randn((2,4,10))
        - X = positional_encoding(X, 10)
        - print(X.shape)
        - torch.Size([2, 4, 10])
    '''

    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1, max_len, num_features))
    X_ = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
        10000,
        torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)
    P[:, :, 0::2] = torch.sin(X_)
    P[:, :, 0::2] = torch.cos(X_)
    X = X + P[:, :X.shape[1], :].to(X.device)
    return dropout(X)

# GCN
# num_node = 17
# self_link = [(i, i) for i in range(num_node)]
# inward = [
#     (10, 8), (8, 6), (9, 7), (7, 5), # arms
#     (15, 13), (13, 11), (16, 14), (14, 12), # legs
#     (11, 5), (12, 6), (11, 12), (5, 6), # torso
#     (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2) # nose, eyes and ears
# ]
# outward = [(j, i) for (i, j) in inward]
# neighbor = inward + outward
# A = graph.get_adjacency_matrix(neighbor, num_node)
# A_edge = graph.edge2mat(self_link, num_node)
# graph_data = A + A_edge
# graph_data = torch.tensor(graph_data, dtype=torch.float)
# degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  #[N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
# degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
# degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0
# degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵
# final_data = torch.mm(degree_matrix, graph_data)
# grf = torch.zeros([10, 34, 34])
# for i in range(10):
#     grf[i, :, :] = final_data
# print(grf.shape)