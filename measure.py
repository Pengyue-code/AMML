import numpy as np

from sklearn.metrics import hamming_loss

from sklearn.metrics import hamming_loss


def HammingLossClass(y_true, y_pred):
    y_hot = np.array(y_pred > 0.5, dtype=float)
    HammingLoss = []
    for i in range(y_true.shape[0]):
        H = hamming_loss(y_true[i, :], y_hot[i, :])
        HammingLoss.append(H)
    return np.mean(HammingLoss)


def HammingLoss(label, predict):
    # label: (N, D)
    D = len(label[0])  # the number of label
    N = len(label)
    tmp = 0
    for i in range(N):
        tmp = tmp + np.sum(label[i] ^ predict[i])
    hamming_loss = tmp / N / D
    return hamming_loss


def Coverage(label, logit):
    D = len(label[0])
    N = len(label)
    label_index = []
    for i in range(N):
        index = np.where(label[i] == 1)[0]
        label_index.append(index)
    cover = 0
    for i in range(N):
        # 从大到小排序
        index = np.argsort(-logit[i]).tolist()
        tmp = 0
        for item in label_index[i]:
            tmp = max(tmp, index.index(item) + 1)
        cover += tmp
    coverage = cover * 1.0 / N
    return coverage


def One_error(label, logit):
    logit = np.array(logit > 0.5, dtype=float)
    label = label.tolist()
    logit = logit.tolist()

    N = len(label)
    for i in range(N):
        if max(label[i]) == 0:
            print("该条数据哪一类都不是")
    label_index = []
    for i in range(N):
        index = np.where(label[i] == 1)[0]
        label_index.append(index)
    OneError = 0
    for i in range(N):
        if np.argmax(logit[i]) not in label_index[i]:
            OneError += 1
    OneError = OneError * 1.0 / N
    return OneError


def compute_rank(y_prob):
    rank = np.zeros(y_prob.shape)
    for i in range(len(y_prob)):
        temp = y_prob[i, :].argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(y_prob[i, :]))
        rank[i, :] = ranks
    return y_prob.shape[1] - rank


def Average_Precision2(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    precision = 0
    for i in range(num_samples):
        positive = np.sum(label[i, :] > 0.5)
        rank_i = rank[i, label[i, :] > 0.5]
        temp = rank_i.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(rank_i))
        ranks = ranks + 1
        ans = ranks * 1.0 / rank_i
        if positive > 0:
            precision += np.sum(ans) * 1.0 / positive
    return precision / num_samples


def Average_Precision(label, logit):
    N = len(label)
    for i in range(N):
        if max(label[i]) == 0 or min(label[i]) == 1:
            print("该条数据哪一类都不是或者全都是")
    precision = 0
    for i in range(N):
        index = np.where(label[i] == 1)[0]
        score = logit[i][index]
        score = sorted(score)
        score_all = sorted(logit[i])
        precision_tmp = 0
        for item in score:
            tmp1 = score.index(item)
            tmp1 = len(score) - tmp1
            tmp2 = score_all.index(item)
            tmp2 = len(score_all) - tmp2
            precision_tmp += tmp1 / tmp2
        precision += precision_tmp / len(score)
    Average_Precision = precision / N
    return Average_Precision


def RankingLoss(label, logit):
    N = len(label)
    for i in range(N):
        if max(label[i]) == 0 or min(label[i]) == 1:
            print("该条数据哪一类都不是或者全都是")
    rankloss = 0
    for i in range(N):
        index1 = np.where(label[i] == 1)[0]
        index0 = np.where(label[i] == 0)[0]
        tmp = 0
        for j in index1:
            for k in index0:
                if logit[i][j] <= logit[i][k]:
                    tmp += 1
        rankloss += tmp * 1.0 / ((len(index1)) * len(index0))
    rankloss = rankloss / N
    return rankloss


logit = np.array([[0.3, 0.4, 0.5, 0.1, 0.15]])
label = np.array([[1, 0, 1, 0, 0]])
pred = np.array([[0, 1, 1, 0, 0]])

# print(HammingLoss(label, pred))