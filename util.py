import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score


def transfer_letter_to_num(data):
    res = []
    for seq in data:
        tmp = deepcopy(seq)
        tmp = tmp.replace('A', '0')
        tmp = tmp.replace('G', '1')
        tmp = tmp.replace('C', '2')
        tmp = tmp.replace('T', '3')
        tmp = np.array([int(s) for s in list(tmp)[:-1]])
        one_hot = np.zeros(shape=(len(tmp), 4))
        one_hot[np.arange(len(tmp)), tmp] = 1
        res.append(one_hot.reshape((1, -1, 4)))
    res = np.concatenate(res)
    return res


def cal_auc(pred, label):
    res = .0
    res_list = []
    for idx in range(len(label)):
        tmp = roc_auc_score(label[idx], pred[idx])
        res += tmp
        res_list.append(tmp)
    res /= len(label)
    return res_list, res
