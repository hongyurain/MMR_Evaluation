
import pdb
import pandas as pd
import numpy as np
from math import ceil
from collections import defaultdict
import json
from operator import itemgetter
import random


def dict2txt(file_name, data):
    with open(file_name, "w") as outf:
        for u in data:
            for v in data[u]:
                outf.write("\t".join([str(u), str(v), str(data[u][v])]) + "\n")


def user_filter_old(df, user_thr=10):
    users = defaultdict(lambda: defaultdict(list))
    for i, (u, v, r, t) in df.iterrows():
        users[u][v].append((r, t))
    train_users, train_items = [], []

    for u in users:
        if len(users[u]) >= user_thr:
            train_users.append(u)

    train_pairs = []
    for u in train_users:
        for v in users[u]:
            for r, t in users[u][v]:
                train_pairs.append((u, v, r, t))
            train_items.append(v)

    return train_pairs, set(train_users), set(train_items)


def ml100k_user_filter(df, item_cate, user_thr=10):
    users = defaultdict(lambda: defaultdict(list))
    for i, (u, v, r, t) in df.iterrows():
        # pdb.set_trace()
        if v in item_cate:
            users[u][v].append((r, t))
    train_users, train_items = [], []

    for u in users:
        if len(users[u]) >= user_thr:
            train_users.append(u)

    train_pairs = []
    for u in train_users:
        for v in users[u]:
            for r, t in users[u][v]:
                train_pairs.append((u, v, r, t))
            train_items.append(v)

    return train_pairs, set(train_users), set(train_items)


def user_filter(df, user_thr=10, top_item_num=5000):
    users = defaultdict(lambda: defaultdict(list))
    items = defaultdict(lambda: defaultdict(list))

    for i, (u, v, r, t) in df.iterrows():
        users[int(u)][int(v)].append((r, t))
        items[int(v)][int(u)].append((r, t))

    train_items = [(i, len(items[i])) for i in items]
    train_items.sort(key=itemgetter(1))
    train_items = train_items[::-1][:top_item_num]
    
    train_items = set([i for i, v in train_items])
    train_users = []

    for u in users:
        s1 = set(users[u].keys()).intersection(train_items)
        if len(s1) >= user_thr:
            train_users.append(u)

    train_users = set(train_users)
    train_pairs = []

    for u in train_users:
        for v in users[u]:
            if v in train_items:
                for r, t in users[u][v]:
                    train_pairs.append((u, v, r, t))
    return train_pairs, set(train_users), set(train_items)


def user_filter_10m(df, user_thr=10, random_user_num=5000, top_item_num=5000):
    users = defaultdict(lambda: defaultdict(list))
    items = defaultdict(lambda: defaultdict(list))

    for i, (u, v, r, t) in df.iterrows():
        users[u][v].append((r, t))
        items[v][u].append((r, t))

    # pdb.set_trace()

    train_items = [(i, len(items[i])) for i in items]
    train_items.sort(key=itemgetter(1))
    train_items = train_items[::-1][:top_item_num]
    
    train_items = set([i for i, v in train_items])
    train_users = []

    for u in users:
        s1 = set(users[u].keys()).intersection(train_items)
        if len(s1) >= user_thr:
            train_users.append(u)

    # pdb.set_trace()

    # if len(train_users) > random_user_num:
    #     train_users = set(random.sample(list(set(train_users)), random_user_num))

    train_pairs = []

    for u in train_users:
        for v in users[u]:
            if v in train_items:
                for r, t in users[u][v]:
                    train_pairs.append((u, v, r, t))
    # pdb.set_trace()
    return train_pairs, set(train_users), set(train_items)


def txt2dict(file_name):
    data = defaultdict(lambda: defaultdict(float))
    users, items = [], []
    with open(file_name, "r") as inf:
        for line in inf:
            u, v, w, t = line.strip("\n").split(",")
            data[u][v] += float(w)
            users.append(u)
            items.append(v)
    users = set(users)
    items = set(items)
    return data, users, items


def occf(train_file, test_file):
    train_data, train_users, train_items = txt2dict(train_file)
    test_data, _, _ = txt2dict(test_file)
    user_map = dict(zip(train_users, range(len(train_users))))
    item_map = dict(zip(train_items, range(len(train_items))))
    new_file = train_file.split(".")[0] + "_occf.txt"
    with open(new_file, "w") as outf:
        for u in train_data:
            for v in train_data[u]:
                outf.write("\t".join([str(user_map[u]), str(item_map[v]), str(train_data[u][v])]) + "\n")
    new_file = test_file.split(".")[0] + "_occf.txt"
    with open(new_file, "w") as outf:
        for u in test_data:
            if u in user_map:
                for v in test_data[u]:
                    if v in item_map:
                        outf.write("\t".join([str(user_map[u]), str(item_map[v]), str(test_data[u][v])]) + "\n")


def data_partition_temporal_ml(inter_file_name, item_file_name=None, train_ratio=0.8, rating_thr=4, num_thr=10):
    df = pd.read_csv(inter_file_name, sep="\t", header=None, names=["users", "items", "ratings", "time"])
    df = df.sort_values(["time"], ascending=[1])

    df = df[df["time"] > 800000000]
    df = df[df["ratings"] >= rating_thr]

    pdb.set_trace()

    df = df.sort_values(["time", "users", "items"], ascending=[1, 1, 1])
    num = int(ceil(train_ratio * df.shape[0]))

    train_pairs, train_users, train_items = user_filter_10m(df[:num], num_thr, 5000, 5000)
    # train_pairs, train_users, train_items = user_filter_old(df[:num], num_thr)

    print(len(train_pairs), len(train_users), len(train_items))

    pdb.set_trace()

    user_map = dict(zip(train_users, range(len(train_users))))
    item_map = dict(zip(train_items, range(len(train_items))))

    train_dict = defaultdict(lambda: defaultdict(float))
    test_dict = defaultdict(lambda: defaultdict(float))

    with open("datasets/train.txt", "w") as outf:
        # for i, (u, v, r, t) in df[:num].iterrows():
        for u, v, r, t in train_pairs:
            train_dict[user_map[int(u)]][item_map[int(v)]] = float(r)
            outf.write("\t".join([str(user_map[int(u)]), str(item_map[int(v)]), str(r), str(int(t))]) + "\n")

    with open("datasets/test.txt", "w") as outf:
        for i, (u, v, r, t) in df[num:].iterrows():
            if u in user_map and v in item_map:
                test_dict[user_map[int(u)]][item_map[int(v)]] = float(r)
                outf.write("\t".join([str(user_map[int(u)]), str(item_map[int(v)]), str(r), str(int(t))]) + "\n")

    dict2txt("datasets/train_occf.txt", train_dict)
    dict2txt("datasets/test_occf.txt", test_dict)
    np.save("datasets/user_mapping.npy", user_map)
    np.save("datasets/item_mapping.npy", item_map)

    # the category of items
    if item_file_name is not None:
        item_cate = item_categories(item_file_name, item_map)
        # item_cate = ml100k_item_categories(item_file_name, item_map)
        item_sim = item_similarities(item_cate, len(item_map))
        np.save("datasets/item_sim.npy", item_sim)
        np.save("datasets/item_categories.npy", item_cate)


def item_similarities(item_cate, num_items):
    S = np.zeros((num_items, num_items))
    for i in range(num_items):
        for j in range(i+1, num_items):
            s1 = set(item_cate[i]).intersection(set(item_cate[j]))
            s2 = set(item_cate[i]).union(set(item_cate[j]))
            # pdb.set_trace()
            S[i, j] = float(len(s1)) / float(len(s2))
            S[j, i] = S[i, j]
    return S


def item_categories(item_file_name, item_map):
    item_df = pd.read_csv(item_file_name, sep="::", header=None, names=["items", "item_names", "categories"])
    item_df["categories"] = item_df["categories"].apply(lambda x: x.split("|"))
    item_cate = defaultdict(list)
    
    for index, row in item_df.iterrows():
        i, n, c = row
        if i in item_map:
            item_cate[item_map[i]] = c
    return item_cate


def ml100k_item_categories(item_file_name):
    item_cate = defaultdict(list)

    with open(item_file_name) as inf:
        for line in inf:
            meta_data = line.strip("\n").split("|")
            item_id = int(meta_data[0])
            cate_vec = meta_data[-19:]
            vec1 = [i for i, v in enumerate(cate_vec) if i > 0 and int(v) > 0]
            if len(vec1) > 0:
                item_cate[item_id] =  vec1

    return item_cate


def ml100k_data_partition_temporal(inter_file_name, item_file_name=None, train_ratio=0.8, rating_thr=4, num_thr=10):
    df = pd.read_csv(inter_file_name, sep="\t", header=None, names=["users", "items", "ratings", "time"])
    df = df.sort_values(["time"], ascending=[1])

    df = df[df["time"] > 800000000]
    df = df[df["ratings"] >= rating_thr]

    df = df.sort_values(["time", "users", "items"], ascending=[1, 1, 1])
    num = int(ceil(train_ratio * df.shape[0]))

    item_cate = ml100k_item_categories(item_file_name)
    train_pairs, train_users, train_items = ml100k_user_filter(df[:num], item_cate, num_thr)


    print(len(train_pairs), len(train_users), len(train_items))

    pdb.set_trace()

    user_map = dict(zip(train_users, range(len(train_users))))
    item_map = dict(zip(train_items, range(len(train_items))))

    train_dict = defaultdict(lambda: defaultdict(float))
    test_dict = defaultdict(lambda: defaultdict(float))

    with open("datasets/train.txt", "w") as outf:
        # for i, (u, v, r, t) in df[:num].iterrows():
        for u, v, r, t in train_pairs:
            train_dict[user_map[int(u)]][item_map[int(v)]] = float(r)
            outf.write("\t".join([str(user_map[int(u)]), str(item_map[int(v)]), str(r), str(int(t))]) + "\n")

    with open("datasets/test.txt", "w") as outf:
        for i, (u, v, r, t) in df[num:].iterrows():
            if u in user_map and v in item_map:
                test_dict[user_map[int(u)]][item_map[int(v)]] = float(r)
                outf.write("\t".join([str(user_map[int(u)]), str(item_map[int(v)]), str(r), str(int(t))]) + "\n")

    dict2txt("datasets/train_occf.txt", train_dict)
    dict2txt("datasets/test_occf.txt", test_dict)
    np.save("datasets/user_mapping.npy", user_map)
    np.save("datasets/item_mapping.npy", item_map)

    new_item_cate = {}
    for i in item_map:
        new_item_cate[item_map[i]] = item_cate[i]

    # the category of items
    if item_file_name is not None:
        item_sim = item_similarities(new_item_cate, len(item_map))
        np.save("datasets/item_sim.npy", item_sim)
        np.save("datasets/item_categories.npy", item_cate)


def data_partition_user(file_name, train_ratio, thr=4, sep="::"):
    df = pd.read_csv(file_name, sep=sep, header=None, names=["users", "items", "ratings", "time"])
    df = df[df["ratings"] >= thr]

    pdb.set_trace()

    user_set = list(set(df["users"]))
    num = int(ceil(train_ratio * len(user_set)))
    train_users = set(np.random.choice(user_set, num, replace=False))

    train_data = []
    train_items = []
    test_data  = []

    for i, (u, v, r, t) in df.iterrows():
        if u in train_users:
            train_data.append((u, v, r))
            train_items.append(v)
        else:
            test_data.append((u, v, r))

    train_items = set(train_items)

    train_user_map = dict(zip(list(train_users), range(len(train_users))))
    train_item_map = dict(zip(list(train_items), range(len(train_items))))

    print("num_train_users:%s, num_train_items:%s, num_train_data:%s" %(len(train_users), len(train_items), len(train_data)))
    print("num_test_data:%s" % len(test_data))

    # mapping training data
    train_data = [(str(train_user_map[u]), str(train_item_map[v]), r) for u, v, r in train_data]

    # filtering the testing data
    test_data = [(u, str(train_item_map[v]), r) for u, v, r in test_data if v in train_items]
    print("num_test_data:%s" % len(test_data))

    return np.array(train_data), np.array(test_data)


def data_temporal_partition_user(file_name, train_ratio, rating_thr=4, train_pos_thr=5, sep="::"):
    """
        temporally split the data into training and testing
    """
    df = pd.read_csv(file_name, sep=sep, header=None, engine="python", names=["users", "items", "ratings", "time"])

    df = df[df["ratings"] >= rating_thr]

    grouped = df.groupby(["users"])

    train_data, train_users, train_items = [], [], []
    test_data_dict = defaultdict(list)
    test_data = []

    for name, group in grouped:
        sorted_g = group.sort_values(["time"], ascending=True)
        num  = int(ceil(len(sorted_g) * train_ratio))
        if num >= train_pos_thr:
            train_data.extend(sorted_g[:num].values.tolist())
            train_items.extend(sorted_g[:num]["items"].values.tolist())
            train_users.append(name)
            test_data_dict[name] = sorted_g[num:].values.tolist()

    train_users = set(train_users)
    train_items = set(train_items)

    train_user_map = dict(zip(list(train_users), range(len(train_users))))
    train_item_map = dict(zip(list(train_items), range(len(train_items))))

    print("num_train_users:%s, num_train_items:%s, num_train_data:%s" %(len(train_users), len(train_items), len(train_data)))

    # mapping training data
    train_data = [(str(train_user_map[u]), str(train_item_map[v]), str(r), str(t)) for u, v, r, t in train_data]

    for key in test_data_dict:
        vec = [(str(train_user_map[u]), str(train_item_map[v]), str(r), str(t)) for u, v, r, t in test_data_dict[key] if (v in train_items and u in train_users)]
        test_data.extend(vec)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    test_users = set(test_data[:, 0])
    test_items = set(test_data[:, 1])

    print("num_test_users:%s, num_test_items:%s, num_test_data:%s" % (len(test_users), len(test_items), test_data.shape[0]))

    return train_data, test_data, train_user_map, train_item_map


def data_partition_temporal_gowalla(inter_file_name, train_ratio=0.8, user_num_thr=10, item_num_thr=10):
    """
        function used for processing the location based data
    """
    df = pd.read_csv(inter_file_name)[["UserID", "LocationID", "Time"]]
    df = df.sort_values(["Time"], ascending=[1])
    df = df.sort_values(["Time", "UserID", "LocationID"], ascending=[1, 1, 1])
    df["Rating"] = 1

    df = df[["UserID", "LocationID", "Rating", "Time"]]
    num = int(ceil(train_ratio * df.shape[0]))

    train_pairs, train_users, train_items = user_filter(df[:num], user_num_thr, item_num_thr)

    print(len(train_pairs), len(train_users), len(train_items))

    user_map = dict(zip(train_users, range(len(train_users))))
    item_map = dict(zip(train_items, range(len(train_items))))

    train_dict = defaultdict(lambda: defaultdict(float))
    test_dict = defaultdict(lambda: defaultdict(float))

    with open("datasets/train.txt", "w") as outf:
        for u, v, r, t in train_pairs:
            train_dict[user_map[int(u)]][item_map[int(v)]] = float(r)
            outf.write("\t".join([str(user_map[int(u)]), str(item_map[int(v)]), str(r), str(t)]) + "\n")

    with open("datasets/test.txt", "w") as outf:
        for i, (u, v, r, t) in df[num:].iterrows():
            if u in user_map and v in item_map:
                test_dict[user_map[int(u)]][item_map[int(v)]] = float(r)
                outf.write("\t".join([str(user_map[int(u)]), str(item_map[int(v)]), str(r), str(t)]) + "\n")

    dict2txt("datasets/train_occf.txt", train_dict)
    dict2txt("datasets/test_occf.txt", test_dict)
    np.save("datasets/user_mapping.npy", user_map)
    np.save("datasets/item_mapping.npy", item_map)


def item_category_similarities(in_cate_file, out_sim_file):
    item_cate = np.load(in_cate_file).item()
    num_items = len(item_cate)
    S = np.zeros((num_items, num_items))

    for i in range(num_items):
        for j in range(i+1, num_items):
            s1 = set(item_cate[i]).intersection(set(item_cate[j]))
            s2 = set(item_cate[i]).union(set(item_cate[j]))
            S[i, j] = float(len(s1)) / float(len(s2))
            S[j, i] = S[i, j]

    np.save(out_sim_file, S)


def yelp_data_temporal(rating_file, item_category_file, train_ratio=0.8, rating_thr=4, num_thr=10):
    ratings = []
    item_cate = {}

    with open(item_category_file, encoding='utf-8') as inf:
        for line in inf:
            x = json.loads(line)
            item_cate[x["business_id"]] = x["categories"]

    with open(rating_file, encoding='utf-8') as inf:
        for line in inf:
            x = json.loads(line)
            if (item_cate[x["business_id"]] is not None) and (len(item_cate[x["business_id"]]) > 0):
                ratings.append((x["user_id"], x["business_id"], x["stars"], x["date"]))

    df = pd.DataFrame(ratings, columns=["user_id", "item_id", "rating", "date"])
    # pdb.set_trace()

    df = df[df["rating"] >= rating_thr]
    df["date"] = pd.to_datetime(df["date"])

    # pdb.set_trace()

    df = df[df["date"] > pd.to_datetime("20141115", format="%Y%m%d")]

    # pdb.set_trace()

    df = df.sort_values(["date"], ascending=[1])
    num = int(ceil(train_ratio * df.shape[0]))

    train_pairs, train_users, train_items = user_filter_10m(df[:num], num_thr, random_user_num=5000, top_item_num=3000)

    print(len(train_pairs), len(train_users), len(train_items))

    # pdb.set_trace()

    user_map = dict(zip(train_users, range(len(train_users))))
    item_map = dict(zip(train_items, range(len(train_items))))

    train_dict = defaultdict(lambda: defaultdict(float))
    test_dict = defaultdict(lambda: defaultdict(float))

    with open("datasets/train.txt", "w") as outf:
        for u, v, r, t in train_pairs:
            train_dict[user_map[u]][item_map[v]] = float(r)
            outf.write("\t".join([str(user_map[u]), str(item_map[v]), str(r), str(t)]) + "\n")

    with open("datasets/test.txt", "w") as outf:
        for i, (u, v, r, t) in df[num:].iterrows():
            if u in user_map and v in item_map:
                test_dict[user_map[u]][item_map[v]] = float(r)
                outf.write("\t".join([str(user_map[u]), str(item_map[v]), str(r), str(t)]) + "\n")

    np.save("datasets/user_mapping.npy", user_map)
    np.save("datasets/item_mapping.npy", item_map)

    # the category of items
    if item_category_file is not None: 
        item_cate_new = {}
        for i in item_map:
            item_cate_new[item_map[i]] = item_cate[i]

        item_sim = item_similarities(item_cate_new, len(item_map))
        # pdb.set_trace()
        np.save("datasets/item_sim.npy", item_sim)
        np.save("datasets/item_categories.npy", item_cate)


def anime_data_temporal(rating_file, item_category_file, train_ratio=0.8, rating_thr=4, num_thr=10):

    df = pd.read_csv(rating_file)[["user_id", "anime_id", "rating"]]
    df = df.sort_values(["Time"], ascending=[1])
    df = df.sort_values(["Time", "UserID", "LocationID"], ascending=[1, 1, 1])
    df["Rating"] = 1

    with open(item_category_file, encoding='utf-8') as inf:
        for line in inf:
            x = json.loads(line)
            item_cate[x["business_id"]] = x["categories"]

    with open(rating_file, encoding='utf-8') as inf:
        for line in inf:
            x = json.loads(line)
            if (item_cate[x["business_id"]] is not None) and (len(item_cate[x["business_id"]]) > 0):
                ratings.append((x["user_id"], x["business_id"], x["stars"], x["date"]))

    df = pd.DataFrame(ratings, columns=["user_id", "item_id", "rating", "date"])
    # pdb.set_trace()

    df = df[df["rating"] >= rating_thr]
    df["date"] = pd.to_datetime(df["date"])

    # pdb.set_trace()

    df = df[df["date"] > pd.to_datetime("20141115", format="%Y%m%d")]

    # pdb.set_trace()

    df = df.sort_values(["date"], ascending=[1])
    num = int(ceil(train_ratio * df.shape[0]))

    train_pairs, train_users, train_items = user_filter_10m(df[:num], num_thr, random_user_num=5000, top_item_num=3000)

    print(len(train_pairs), len(train_users), len(train_items))

    # pdb.set_trace()

    user_map = dict(zip(train_users, range(len(train_users))))
    item_map = dict(zip(train_items, range(len(train_items))))

    train_dict = defaultdict(lambda: defaultdict(float))
    test_dict = defaultdict(lambda: defaultdict(float))

    with open("datasets/train.txt", "w") as outf:
        for u, v, r, t in train_pairs:
            train_dict[user_map[u]][item_map[v]] = float(r)
            outf.write("\t".join([str(user_map[u]), str(item_map[v]), str(r), str(t)]) + "\n")

    with open("datasets/test.txt", "w") as outf:
        for i, (u, v, r, t) in df[num:].iterrows():
            if u in user_map and v in item_map:
                test_dict[user_map[u]][item_map[v]] = float(r)
                outf.write("\t".join([str(user_map[u]), str(item_map[v]), str(r), str(t)]) + "\n")

    np.save("datasets/user_mapping.npy", user_map)
    np.save("datasets/item_mapping.npy", item_map)

    # the category of items
    if item_category_file is not None: 
        item_cate_new = {}
        for i in item_map:
            item_cate_new[item_map[i]] = item_cate[i]

        item_sim = item_similarities(item_cate_new, len(item_map))
        # pdb.set_trace()
        np.save("datasets/item_sim.npy", item_sim)
        np.save("datasets/item_categories.npy", item_cate)


if __name__ == "__main__":

    ml100k_data_partition_temporal("datasets/ml-100k/u.data", "datasets/ml-100k/u.item", 0.8, rating_thr=4, num_thr=10)

    # data_partition_temporal_gowalla("datasets/Chicago/ChicagoSubCheckins.csv", 0.8, 10, 10)

    # item_category_similarities("datasets/ml-1m/ml-1m_tmp_0.8_10_item_categories.npy", "datasets/ml-1m/ml-1m_tmp_0.8_10_item_similarities.npy")

    # yelp_data_temporal("datasets/yelp/review.json", "datasets/yelp/business.json")

    # file_name = "../datasets/ml-100k/ratings.txt"
    # train_ratio = 0.8

    # train_data, test_data, uMap, iMap = data_temporal_partition_user(file_name, train_ratio, rating_thr=4, train_pos_thr=5, sep="\t")
    # np.savetxt("../datasets/ml-100k/user_temporal_train.txt", train_data, delimiter='\t', fmt="%s")
    # np.savetxt("../datasets/ml-100k/user_temporal_test.txt", test_data, delimiter='\t', fmt="%s")

    # np.save("../datasets/ml-100k/user_temporal_user_id_map.npy", uMap)
    # np.save("../datasets/ml-100k/user_temporal_item_id_map.npy", iMap)
