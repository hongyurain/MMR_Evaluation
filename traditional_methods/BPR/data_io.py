
import json
import numpy as np
from collections import defaultdict


def load_history_array(file_name):
    with open(file_name, "r") as inf:
        # arr = np.array([line.split()[:2] for line in inf], dtype=np.int32) - 1
        arr = np.array([line.split(' ')[:2] for line in inf], dtype=np.int64)
    return arr


def load_history_dict(file_name):
    user_history = defaultdict(list)
    with open(file_name, 'r') as inf:
        for line in inf:
            # u, i, t = line.split("\t")
            data = line.split(" ")
            u, i = data[0], data[1]
            # user_history[int(u)-1].append(int(i)-1)
            user_history[int(u)].append(int(i))
    return user_history


# def load_history_array_one(file_name):
#     pairs, ratings = [], []
#     with open(file_name, "r") as inf:
#         # Mydtype = [('users', np.int32), ('items', np.int32), ('ratings', np.float32)]
#         for line in inf:
#             u, i, r = line.strip('\n').split('$$')
#             pairs.append([u, i])
#             ratings.append(r)
#     return np.array(pairs, dtype=np.int32), np.array(ratings, dtype=np.float64)
