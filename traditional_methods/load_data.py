import numpy as np
import random as rd
import pandas as pd


def load_file_dict(file_name):
    df = pd.read_csv(file_name, sep=',', names=['users','items','time'], skiprows=1)  # "users", "items" , "ratings", "time", "xlable" 
    # print('df = ', df)
    print('size = ', df.shape)
    test = {}
    user_dict = {}
    all_items = []
    for name, g in df.groupby('users'):
        items = []
        test[name] = g.sort_values(by=["time"], ascending=[1])["items"].values
        # test[name] = g.sort_values(by=["items"], ascending=[1])["items"].values
        for i in test[name]:
            all_items.append(i)
            if i not in items:
                items.append(i)
        user_dict[name] = items
    # print('user_dict = ', user_dict)
    return user_dict, set(all_items)


class Data(object):
    def __init__(self, train_file, validation_file, test_file):
        # get number of users and items
        # self.train_items, self.n_items = load_file_dict(train_file)
        self.train_items, set_item_train_jing = load_file_dict(train_file)
        self.test_set, set_item_test_jing = load_file_dict(test_file)
        self.validation_items, set_item_valid_jing = load_file_dict(validation_file)
        self.n_items = len(set_item_train_jing|set_item_test_jing|set_item_valid_jing)
        self.n_users = len(self.train_items.keys())
        print('self.n_users = ', self.n_users)
        print('self.n_items = ', self.n_items)
        self.R = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        self.V = np.zeros((self.n_users, self.n_items), dtype=np.float32)

        for u in self.train_items.keys():
            for i in self.train_items[u]:
                self.R[u][i] = 1

        for u in self.validation_items.keys():
            for i in self.validation_items[u]:
                self.V[u][i] = 1

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(range(self.n_users), self.batch_size)
        else:
            users = [rd.choice(range(self.n_users)) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]#np.nonzero(self.graph[u,:])[0].tolist()
            if len(pos_items) >= num:
                return rd.sample(pos_items, num)
            else:
                return [rd.choice(pos_items) for _ in range(num)]

        def sample_neg_items_for_u(u, num):
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))#np.nonzero(self.graph[u,:] == 0)[0].tolist()
                        
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items
