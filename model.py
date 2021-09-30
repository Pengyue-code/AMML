import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha, dropout, num_classes):  # in_features=72, out_features=8
        super(GraphAttentionLayer, self).__init__()
        self.alpha = alpha
        self.dropout = dropout
        self.num_classes = num_classes
        self.out_features = out_features

        # Add variable
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # a is a single-layer feedforward neural network
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.Classes = torch.nn.Linear(out_features, num_classes)

    def forward(self, h, dist):
        # print(indices.shape)  # (593, 10), h=torch.Size([6, 72])
        # formula 1
        # print("W", self.W.shape)  # torch.Size([72, 9])
        Wh = torch.mm(h, self.W)  # 593*8
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # torch.Size([6, 1])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # torch.Size([6, 1])
        # print(Wh1.shape)  # torch.Size([6, 1])
        # print(Wh2.shape)  # torch.Size([6, 1])
        # broadcast add
        e = Wh1 + Wh2.T
        # print(e.shape)  # torch.Size([6, 6])
        e = self.leakyrelu(e)  # formula 3
        zero_vec = -9e15 * torch.ones_like(e)  # 维度大小与e相同，所有元素都是-9*10的15次方
        attention = torch.where(dist > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)  # formula 2，attention就是公式里的αij

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_update = torch.matmul(attention, Wh)  # formula 4
        h_update = F.elu(h_update)
        h_update = F.dropout(h_update, self.dropout, training=self.training)
        y_pred = self.Classes(h_update)  # 593 * 8
        return y_pred


