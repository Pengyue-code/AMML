import arff
import numpy
import numpy as np
import os
from sklearn import preprocessing
from torch.utils.data import TensorDataset


def load():
    file = open(os.path.join("datasets", "emotions.arff"), "r")

    decoder = arff.ArffDecoder()
    draft = decoder.decode(file)

    file.close()

    data = numpy.array(draft['data'], dtype=numpy.float32)

    x = data[:, 0: 72]
    y = data[:, 72: 78]

    # flags数据集
    # x = data[:, 0: 10]  # print(features.shape)  (593, 72)
    # y = data[:, 10: 17]  # print(labels.shape)  (593, 6)

    x = preprocessing.scale(x)

    """
    计算pairwise距离
    :param x: 矩阵matrix
    :return: 距离
    """
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)

    sortedIndices = dist.argsort()   # 排序，得到排序后的下标
    indices = sortedIndices[:, 1:10+1]  # 取最小的k个
    for i in range(len(dist)):
        temp = indices[i]
        for j in range(len(dist)):
            if j in temp:
                dist[i][j] = 1
            else:
                dist[i][j] = 0
    # print("第二步，邻接矩阵构建完成\n")
    # print("sum(dist[20]):", dist[20])
    return dist, x, y

'''
accuracy输入为分类的最后一层输出，与label进行比较，然后计算准确率
'''
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# if __name__ == '__main__':
#     indices, features, labels = load()
#     print("第1步，邻接矩阵构建成功\n")
