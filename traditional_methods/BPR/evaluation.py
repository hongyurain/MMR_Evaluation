
import math
import numpy as np
import pdb


def precision(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set)) / float(N)
    elif isinstance(N, list):
        return np.array([precision(actual, predicted, n) for n in N])


def recall(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set)) / float(len(set(actual)))
    elif isinstance(N, list):
        return np.array([recall(actual, predicted, n) for n in N])


def rmse(test_data, U, V):
    ii, jj, rr = test_data["users"], test_data["items"], test_data["ratings"]
    v = np.sum(U[ii, :] * V[jj, :], axis=1)
    return np.sqrt(np.linalg.norm(rr - v)**2 / test_data.shape[0])


def nDCG_LY(Tr, topK, num=None):
    if num is None:
        num = len(topK)
    dcg, vec = 0, []
    for i in range(num):
        if topK[i] in Tr:
            dcg += 1 / math.log(i + 2, 2)
            vec.append(1)
        else:
            vec.append(0)
    vec.sort(reverse=True)
    idcg = sum([vec[i] / math.log(i + 2, 2) for i in xrange(num)])
    if idcg > 0:
        return dcg / idcg
    else:
        return idcg


def nDCG_new(Tr, topK, num=None):
    if num is None:
        num = len(topK)
    vec = [1.0 if v in Tr else 0 for v in topK]
    dcg = sum([vec[i] / math.log(i + 2, 2) for i in xrange(num)])
    vec.sort(reverse=True)
    idcg = sum([vec[i] / math.log(i + 2, 2) for i in xrange(num)])
    if idcg > 0:
        return dcg / idcg
    else:
        return idcg


def nDCG(Tr, topK, num=None):
    # modified by Guimei
    if num is None:
        num = len(topK)
    dcg, vec = 0, []
    for i in range(num):
        if topK[i] in Tr:
            dcg += 1.0 / math.log(i + 2, 2)
            vec.append(1.0)
        else:
            vec.append(0)
    vec.sort(reverse=True)
    # idcg = sum([vec[i]/math.log(i+2, 2) for i in xrange(num)])
    # idcg = sum([1.0 / math.log(i + 2, 2) for i in range(0, min(len(Tr), len(topK)))])
    idcg = sum([1.0 / math.log(i + 2, 2) for i in range(0, min(len(Tr), num))])
    if idcg > 0:
        return dcg / idcg
    else:
        return idcg


def mean_average_precision():
    pass


def AUC():
    pass


def nDCGAtN(correct_items, ranked_items, N=10):
    dcg, num_pos = 0, 0
    correct_set = set(list(correct_items))
    for i in range(len(ranked_items)):
        if ranked_items[i] in correct_set:
            dcg += 1 / math.log(i + 2, 2)
            num_pos += 1
    if num_pos > 0:
        idcg = sum([1.0 / math.log(i + 2, 2) for i in range(num_pos)])
        return dcg / idcg
    else:
        return 0.0
