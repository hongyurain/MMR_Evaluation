import os
import pdb
import BPR
# import time
import math
import random
import xlwt
from scipy import stats
import numpy as np
from collections import defaultdict
from evaluation import precision, recall, nDCG #nDCG_LGM
from tqdm import tqdm
from scipy import spatial
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from datetime import datetime
from time import time


def NDCG(actual, predicted, topk):
    if isinstance(topk, int):
        res = 0
        k = min(topk, len(actual))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[j] in set(actual)) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
        return res / float(len(actual))
    elif isinstance(topk, list):
        return np.array([NDCG(actual, predicted, n) for n in topk])


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def precision(actual, predicted):
    inter_set = set(actual) & set(predicted)
    return float(len(inter_set)) / float(N)



def recall(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set)) / float(len(set(actual)))
    elif isinstance(N, list):
        return np.array([recall(actual, predicted, n) for n in N])



class BPRMF:

    def __init__(self, num_factors=100, reg_u=0.015, reg_i=0.015, theta=0.05,
                 max_iter=100, seed=123, outf=1):
        self.d = num_factors
        self.theta = theta
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.max_iter = max_iter
        self.seed = seed
        self.outf = outf
        self.rng = np.random.RandomState(self.seed)
        random.seed(self.seed)

    def random_sample_triples(self, train_data, Tr_neg):
        pair_arr = [(u, v, BPR.uniform_random_id(Tr_neg[u]["items"].astype(np.int32), Tr_neg[u]["num"])) for u, v in train_data]
        return np.array(pair_arr, dtype=np.int32)

    def sample_validation_data(self, Tr, Tr_neg, ratio=0.05):
        validation = []
        num = max(int(np.ceil(ratio*self.num_users)), 100)  # TODO yinan initial 100
        print('self.num_users = ', self.num_users)
        sub_set = random.sample(Tr.keys(), num)
        for u in sub_set:
            # pdb.set_trace()
            validation.extend([(u, i, BPR.uniform_random_id(Tr_neg[u]["items"].astype(np.int32), Tr_neg[u]["num"])) for i in Tr[u]])
        return np.array(validation)

    def fix_model(self, train_data, Tr, Tr_neg, item_sim=None):
        self.num_users = np.max(train_data[:, 0])+1
        self.num_items = np.max(train_data[:, 1])+1
        self.users = set(range(self.num_users))
        self.items = set(range(self.num_items))
        self.U = self.rng.rand(self.num_users, self.d)
        self.V = self.rng.rand(self.num_items, self.d)
        # print('self.U = ', self.U)
        # print('self.U = ', self.U.shape)
        # print('self.U = ', self.U.flags)
        # # TODO add here yinan
        #
        # self.U = np.load('../dataset/movielens/ml_1m/ml_1m_user64_k1000symjaccard_right.npy')
        # self.V = np.load('../dataset/movielens/ml_1m/ml_1m_item64_k1000symjaccard_right.npy')
        # self.U = np.ascontiguousarray(self.U)
        # self.V = np.ascontiguousarray(self.V)

        if item_sim is not None:
            self.item_sim = item_sim

        valid_data = self.sample_validation_data(Tr, Tr_neg)
        num_valid = valid_data.shape[0]
        curr_theta = self.theta
        last_loss = BPR.compute_loss(
            valid_data, self.U, self.V, self.reg_u, self.reg_i,
            num_valid, self.num_users, self.num_items, self.d)
        for e in tqdm(range(self.max_iter)):
            # tic = time.process_time()
            pairwise_data = self.random_sample_triples(train_data, Tr_neg)
            pairwise_data = pairwise_data.astype(np.int64)
            #pdb.set_trace()
            num_pairs = pairwise_data.shape[0]
            ii = self.rng.permutation(num_pairs)
            BPR.gradient_update(
                pairwise_data[ii, :], self.U, self.V, curr_theta, self.reg_u,
                self.reg_i, num_pairs, self.d)
            if np.isnan(np.linalg.norm(self.U, 'fro')) or \
               np.isnan(np.linalg.norm(self.V, 'fro')):
                print("early stop")
                break
            curr_loss = BPR.compute_loss(
                valid_data, self.U, self.V, self.reg_u, self.reg_i,
                num_valid, self.num_users, self.num_items,
                self.d)
            delta_loss = (curr_loss-last_loss)/last_loss
            # if self.outf > 0:
            #     print("epoch:%d, CurrLoss:%.6f, DeltaLoss:%.6f, Time:%.6f" % (
            #         e, curr_loss, delta_loss, time.process_time()-tic))
            if abs(delta_loss) < 1e-5:
                break
            last_loss, curr_theta = curr_loss, 0.9*curr_theta

    def predict_individual(self, u, inx, k):
        val = np.dot(self.U[u, :], self.V[inx, :].transpose())
        ii = np.argsort(val)[::-1][:k]
        return inx[ii]

    def evaluation(self, Tr_neg, Te, positions=[5, 10]):
        from evaluation import precision
        from evaluation import recall
        from evaluation import nDCG
        prec = np.zeros(len(positions))
        rec = np.zeros(len(positions))
        ndcg = np.zeros(len(positions))
        div = 0.0
        for u in Te:
            val = np.dot(self.U[u, :], self.V.transpose())
            inx = Tr_neg[u]["items"]
            A = set(Te[u])
            B = set(inx) - A
            # compute precision and recall
            ii = np.argsort(val[inx])[::-1][:max(positions)]
            prec += precision(Te[u], inx[ii], positions)
            rec += recall(Te[u], inx[ii], positions)
            ndcg += np.array([nDCG(Te[u], inx[ii], p) for p in positions])
            div += self.diversity(inx[ii])
        return ndcg, prec, rec, div / len(Te.keys())

    def evaluation_new(self, Tr_neg, Te, step_size, steps=1):
        prec = np.zeros(steps)
        rec = np.zeros(steps)
        hr = np.zeros(steps)
        ndcg = np.zeros(steps)
        num = len(Te.keys())
        predict = {}
        recommend = []
        for u in tqdm(Te):
            # pdb.set_trace()
            val = np.dot(self.U[u, :], self.V.transpose())
            inx = Tr_neg[u]["items"]
            # A = set(Te[u])
            # B = set(inx) - A
            ii = np.argsort(val[inx])[::-1]

            for i in range(steps):
                s1 = inx[ii][: (i+1) * step_size]
                predict[u] = list(s1)
                recommend += list(s1)
                # pdb.set_trace()
                inter_set = set(Te[u]).intersection(set(s1))
                # print('s1 = ', s1)
                # print('Te[u] = ', Te[u])
                # input('debug')
                if len(inter_set) >= 1:
                    hr[i] += 1
                prec[i] += float(len(inter_set)) / float(len(s1))
                rec[i] += float(len(inter_set)) / float(len(set(Te[u])))
                ndcg[i] += NDCG(Te[u], list(s1), i)
                # div[i] += self.diversity_3(s1)

                # pdb.set_trace()

                # div1 = self.diversity(s1)
                # div2 = self.diversity_1(s1, self.V)
                # div3 = self.diversity_2(self.V[s1, :])

        result_df = pd.DataFrame({"precision": prec/num, "recall": rec/num, "hr": hr/num, "ndcg": ndcg/num})
        print('result_df = ', result_df)
        return result_df, predict, recommend

    def diversity(self, s_inx):
        num = len(s_inx)
        s = []
        for i in range(num):
            for j in range(i+1, num):
                s.append(spatial.distance.cosine(self.V[s_inx[i], :], self.V[s_inx[j], :]))
        return sum(s) / len(s)

    def diversity_1(self, s_inx, X):
        """
            s_inx: the selected item set, a numpy vector
            X: the item feature matrix, shape (d, m)
            The similarities between items are measured using cosine similarity.
        """
        S = cosine_similarity(X[s_inx, :])
        ii, jj = np.triu_indices(len(s_inx), k=1)
        vec = S[ii, jj]
        div = 1 - np.mean(vec)
        return div

    def diversity_2(self, emb_action):
        div = []
        for i in range(len(emb_action)):
            for j in range(i+1, len(emb_action)):
                div.append(distance.cosine(emb_action[i], emb_action[j]))
        div = np.array(div)
        return np.mean(div)

    def diversity_3(self, s_inx):
        num = len(s_inx)
        s = []
        # pdb.set_trace()
        for i in range(num):
            for j in range(i+1, num):
                s.append(1 - self.item_sim[s_inx[i], s_inx[j]])
        # print(s)
        return sum(s) / len(s)


    def __str__(self):
        return "Recommender: bprmf, num_factors:%s, reg_u:%s, reg_i:%s, theta:%s, max_iter:%s, seed:%s" % (self.d, self.reg_u, self.reg_i, self.theta, self.max_iter, self.seed)


def data_process(train_file, test_file):
    import data_io
    train_data = data_io.load_history_array(train_file)
    print('train size = ', train_data.shape)
    Te = data_io.load_history_dict(test_file)
    Tr = defaultdict(lambda: defaultdict(int))
    items = []
    for u, i in tqdm(train_data):
        Tr[u][i] += 1
        items.append(i)
    items = set(items)
    Tr_neg = {}
    for u in tqdm(Tr):
        # x = list(items-set(Tr[u].keys()))
        x = list(items)
        # pdb.set_trace()
        Tr_neg[u] = {"items": np.array(x), "num": len(x)}
    return train_data, Tr, Tr_neg, Te


def normalization(x):
    # x = x / np.linalg.norm(x)
    x = stats.zscore(x)
    x = np.real(x)
    print('mean = ', np.mean(x))
    print('val = ', np.std(x))
    return x


if __name__ == '__main__':

    DATA_FOLDER = "../../../data/CGN_processed_dataset/douban_books_douban_movies/"  # books, movies_and_tv, toys_games
    DATASET = "douban_movies"

    train_file = DATA_FOLDER + DATASET + "_train.txt"
    test_file = DATA_FOLDER + DATASET + "_test.txt"

    dim = 10

    train_data, Tr, Tr_neg, Te = data_process(train_file, test_file)
    positions = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    optimal_para, optimal_results, max_prec = '', '',  0
    optimal_model = None

    rec_steps, rec_size = 20, 1
    print('start')

    for x in np.arange(-6, -5):
        for y in np.arange(-6, -5):
            # t1 = time.process_time()
            x, y = -2, -2
            print(DATASET)
            model = BPRMF(num_factors=dim, reg_u=10.0**(x), reg_i=10.0**(x), theta=2.0**(y), max_iter=200, seed=123, outf=0)  # TODO initial max_iter=200
            cmd_str = str(model)
            print(cmd_str)
            now = datetime.now()
            print('now = ', now)
            t0 = time()
            model.fix_model(train_data, Tr, Tr_neg)
            t2 = time()
            now = datetime.now()
            print('now = ', now)
            print('t1 - t0 = ', t2 - t0)

            result_df, predict, recommend = model.evaluation_new(Tr_neg, Te, rec_size, rec_steps)
            # print('recommend = ', recommend)
            # input('debug')
            # write to xlwt
            # wb = xlwt.Workbook()
            # ws = wb.add_sheet('recommendation')
            # ws.write(0, 0, 'bpr')
            # for i in range(len(recommend)):
            #     ws.write(i+1, 0, int(recommend[i]))
            # wb.save('rec result bpr.xls')
            # write to xlwt

            # result_df.to_csv("%s_lmf_dim%s_at%s_test.csv" % (DATASET, dim, rec_size), index=False, mode="w")
            # pdb.set_trace()

            # f = open(DATA_FOLDER + DATASET + '_predict_bpr.txt', 'w')
            # f.write(str(predict))
            # f.close()

            # ndcg, prec, rec, div = model.evaluation(Tr_neg, Te, positions)
            #
            # results = ' '.join(['P@%d:%.6f' % (positions[i], prec[i]/len(Te.keys())) for i in range(len(positions))])+' '
            # results += ' '.join(['R@%d:%.6f' % (positions[i], rec[i]/len(Te.keys())) for i in range(len(positions))])+' '
            # # results += ' '.join(['div@%d:%.6f' % (positions[i], div[i]/len(Te.keys())) for i in range(len(positions))])
            # print(results)
            # t2 = time.process_time()
            # print("total time: %s" % t2)
            #
            # if prec[0] > max_prec:
            #     optimal_para = str(model)
            #     optimal_results = results
            #     max_prec = prec[0]
            #     optimal_model = model

    # print("\nthe optimal parameters and results are as follows:\n%s\n%s" % (optimal_para, optimal_results))
