import time
import torch
from torch import nn

from measure import HammingLossClass, HammingLoss, Coverage, Average_Precision2, RankingLoss
from model import GraphAttentionLayer
from utils import load, accuracy
from torch.autograd import Variable
import torch.nn.functional as F

dist, features, labels, samples = load()
dist = torch.from_numpy(dist).type(torch.float32)
features = torch.from_numpy(features).type(torch.float32)
labels = torch.from_numpy(labels).type(torch.float32)

# loss_f = torch.nn.BCELoss()

epochs = 100
lr = 1e-1
alpha = 0.15
dropout = 0.5
h_dimension = features.shape[1]
n_hid = 9
n_class = labels.shape[1]
r1 = 0.5
r2 = 0.3
r3 = 0.2

model = GraphAttentionLayer(in_features=h_dimension, out_features=n_hid, alpha=alpha, dropout=dropout, num_classes=n_class)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_f = torch.nn.BCEWithLogitsLoss()

Sigmoid_fun = nn.Sigmoid()

num_train = int(samples * r1)
num_test = samples - int(samples * r3)

idx_train = range(num_train)  # 0~139
idx_val = range(num_train, num_test)  # 200~499
idx_test = range(num_test, samples)  # 500~1499
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


def compute_test():
    model.eval()
    output = model(data, dist)
    loss_test = loss_f(output[idx_test], labels[idx_test])
    acc_test = HammingLossClass(target[idx_test], Sigmoid_fun(output[idx_test]))
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "hamming loss= {:.4f}".format(acc_test))


for epoch in range(epochs):
    t = time.time()
    # model.train()
    data, target = Variable(features), Variable(labels)
    # print("data", data.shape)  # torch.Size([6, 72])
    optimizer.zero_grad()
    output = model(data, dist)
    loss_train = loss_f(output[idx_train], target[idx_train])
    # precision1, recall = calculate_acuracy_mode_one(Sigmoid_fun(output), target)
    acc1 = HammingLossClass(target[idx_train], Sigmoid_fun(output[idx_train]))

    loss_train.backward()
    optimizer.step()

    with torch.no_grad():
        data, target = Variable(features), Variable(labels)
        output = model(data, dist)
        loss_test = loss_f(Sigmoid_fun(output[idx_val]), target[idx_val])

        # precision2, recall = calculate_acuracy_mode_one(Sigmoid_fun(output), target)
        precision2 = HammingLossClass(target[idx_val], Sigmoid_fun(output[idx_val]))
        # precision3 = Coverage(target, Sigmoid_fun(output))
        # precision4 = Average_Precision2(Sigmoid_fun(output), target)
        # precision5 = RankingLoss(target, Sigmoid_fun(output))
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc1),
          'loss_val: {:.4f}'.format(loss_test.item()),
          'acc_val: {:.4f}'.format(precision2),
          'time: {:.4f}s'.format(time.time() - t))

# Testing
compute_test()
