
import time
import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "bpr.h":
    void gradient_update_func(long int *pairwise_data, double *U, double *V, double theta, double reg_u, double reg_i, long int num_pairs, int d)

    double compute_loss_func(long int *pairwise_data, double *U, double *V, double reg_u,
        double reg_i, long int num_pairs, int m, int n, int d)

    void boost_gradient_update_func(long int *pairwise_data, double *weights, double *U,
        double *V, double theta, double lmbda, long int num_pairs, int loss_code, int d, int flag)

    double boost_compute_loss_func(long int *pairwise_data, double * weights, double *U, double *V, long int num_pairs, int loss_code, int d)

    double boost_compute_loss_one_func(long int *pairwise_data, double * weights, double *U,
        double *V, double *UM, double *VM, double lmbda1, double lmbda2, long int num_pairs, int m, int n, int d)

    void boost_gradient_update_one_func(long int *pairwise_data, double * weights, double *U,
        double *V, double *UM, double * VM,double theta, double lmbda1, double lmbda2, long int num_pairs, int m, int d)

    long int auc_computation_func(long int *pos, long int *neg, double *val, long int num_pos,
        long int num_neg)

    void compute_auc_list_func(long int *pos, long int *neg, double *val, double *measure,
        long int num_pos, long int num_neg)

    void compute_auc_at_N_list_func(long int *pos, long int *neg, double *val, double *measure,
        long int num_pos, long int num_neg, int N)

    void compute_map_list_func(long int *pos_sort, long int *neg_sort, double *val, double *measure, long int num_pos, long int num_neg)

    double mean_average_precision_func(double *pos_val, double *neg_val, long int num_pos,
        long int num_neg)

    void instance_boost_gradient_update_func(long int *pairwise_data, double * weights, double *U, double *V, double theta, double lmbda, long int num_pairs, int loss_code, int d, int flag)

    double instance_boost_compute_loss_func(long int *pairwise_data, double * weights, double *U, double *V, 
        long int num_pairs, int loss_code, int d)

    int random_id(int *list, int size)

    void sample_without_replacement_func(long int *population, int *flag, long int num, long int*list, long int size)

    void adamf_gradient_update_func(long int *data, double *ratings, double * weights, double *U, double *V, double theta, double lmbda, long int num_pairs, int m, int d)

    double tfmap_approx_loss_fast(double *predict_val, long int *pos_inx, int num_pos)

    void tfmap_fast_update_user(long int *buffer_set, double *predict_val, int buffer_size, long int uid, double *U,
        double *V, double theta, double lmbda, int d)

    void tfmap_fast_update_item(long int *buffer, double *predict_val, int buffer_size, long int uid, double *U,
        double *V, double theta, double lmbda, int d)

    void pointwise_gradient_update_func(long int *data, double *ratings, double * weights, double *U, double *V, double theta, double lmbda, long int num_pairs, int d)

    void gbpr_gradient_update(double *U, double *V, long int uid, long int pid1, long int pid2, long int *item_users, int size, int group_size, int num_factors, double theta, double reg_u, double reg_i, double rho)

    void gbpr_fast_update(long int *train_data, long int *index, long int **neg_items, int *neg_size, long int **item_users, int *size, double *U, double *V, int group_size, long int num_logs, int num_factors, double theta, double reg_u, double reg_i, double rho)

    void mf_sgd_optimize(long int *pairs, double *ratings, long int *index, double *W,double *U, double *V, long int num, int num_factors, double theta, double reg_u, double reg_i)

    double mf_least_square_loss(long int *pair_data, double *rating_data, double *W, double *U, double *V, long int num_pairs, long int num_users, long int num_items, int num_factors, double reg_u, double reg_i)

    void warp_fast_update(long int *train_data, long int *index, long int **neg_items, int *neg_size, double *U, double *V, long int num_pairs, int num_factors, int gamma, double theta, double reg_u, double reg_i)

    void warp_sample_update(long int **pos_items, int *pos_size, long int **neg_items, int *neg_size, double *U, double *V, int num_users, int num_factors, int gamma, double theta, double reg_u, double reg_i)

    void socf_bpr_first_order(long int *pairwise_pt, long int *index_pt, long int **neg_items, int *neg_size, double *U, double *V, double *rho_U, double *rho_V, double *hU, double *hV, double theta, double reg_u, double reg_i, double tau_u, double tau_i, long int num_pairs, int num_factors)

    void socf_bpr_second_order(long int *pairwise_pt, long int *index_pt, long int **neg_items, int *neg_size, double *U, double *V, double *rho_U, double *rho_V, double *hU, double *hV, double *SigMu, double *SigMv, double lmbda, double tau_u, double tau_i, double eta, long int num_pairs, int num_factors)

    void adawarp_fast_update(long int *train_data, long int *index, long int **neg_items, int *neg_size, double *U, double *V, double *W, long int num_pairs, int num_factors, int gamma, double theta, double reg_u, double reg_i)

    void adawrmf_fast_train(long int *index_pt, long int *pair_pt, double *R_pt, double *W_pt, double *U_pt, double *V_pt, long int num_pairs, int num_factors,double theta, double lmbda)

    double adawrmf_train_loss(long int *pair_pt, double *R_pt, double *W_pt, double *U_pt, double *V_pt, long int num_pairs, long int num_users, long int num_items, int num_factors, double lmbda)

def gradient_update(np.ndarray[long int, ndim=2, mode="c"] pairwise_data not None,
                    np.ndarray[double, ndim=2, mode="c"] U not None,
                    np.ndarray[double, ndim=2, mode="c"] V not None,
                    theta, reg_u, reg_i, num_pairs, d):
    gradient_update_func(<long int*>np.PyArray_DATA(pairwise_data),<double*>np.PyArray_DATA(U), <double*>np.PyArray_DATA(V), theta, reg_u, reg_i, num_pairs, d)

def compute_loss(np.ndarray[long int, ndim=2, mode="c"] pairwise_data not None,
                    np.ndarray[double, ndim=2, mode="c"] U not None,
                    np.ndarray[double, ndim=2, mode="c"] V not None,
                    reg_u, reg_i, num_pairs, m, n, d):
    return compute_loss_func(<long int*>np.PyArray_DATA(pairwise_data),<double*>np.PyArray_DATA(U), <double*>np.PyArray_DATA(V), reg_u, reg_i, num_pairs, m, n, d)

def boost_gradient_update(np.ndarray[long int, ndim=2, mode="c"] pairwise_data not None,
                    np.ndarray[double, ndim=1, mode="c"] weights not None,
                    np.ndarray[double, ndim=2, mode="c"] U not None,
                    np.ndarray[double, ndim=2, mode="c"] V not None,
                    theta, lmbda, num_pairs, loss_code, d, flag):
    boost_gradient_update_func(<long int*>np.PyArray_DATA(pairwise_data), <double*>np.PyArray_DATA(
        weights),<double*>np.PyArray_DATA(U),<double*>np.PyArray_DATA(V), theta, lmbda,
        num_pairs, loss_code, d, flag)

def boost_compute_loss(np.ndarray[long int, ndim=2, mode="c"] pairwise_data not None,
                    np.ndarray[double, ndim=1, mode="c"] weights not None,
                    np.ndarray[double, ndim=2, mode="c"] U not None,
                    np.ndarray[double, ndim=2, mode="c"] V not None,
                    num_pairs, loss_code, d):
    return boost_compute_loss_func(<long int*>np.PyArray_DATA(pairwise_data), <double*>np.PyArray_DATA(weights), <double*>np.PyArray_DATA(U), <double*>np.PyArray_DATA(V), num_pairs, loss_code, d)

def boost_gradient_update_one(np.ndarray[long int, ndim=2, mode="c"] pairwise_data not None,
                    np.ndarray[double, ndim=1, mode="c"] weights not None,
                    np.ndarray[double, ndim=2, mode="c"] U not None,
                    np.ndarray[double, ndim=2, mode="c"] V not None,
                    np.ndarray[double, ndim=2, mode="c"] UM not None,
                    np.ndarray[double, ndim=2, mode="c"] VM not None,
                    theta, lmbda1, lmbda2, num_pairs, m, d):
    boost_gradient_update_one_func(<long int*>np.PyArray_DATA(pairwise_data),
                                <double*>np.PyArray_DATA(weights),
                                <double*>np.PyArray_DATA(U),
                                <double*>np.PyArray_DATA(V),
                                <double*>np.PyArray_DATA(UM),
                                <double*>np.PyArray_DATA(VM),
                                theta, lmbda1, lmbda2, num_pairs, m, d)

def boost_compute_loss_one(np.ndarray[long int, ndim=2, mode="c"] pairwise_data not None,
                    np.ndarray[double, ndim=1, mode="c"] weights not None,
                    np.ndarray[double, ndim=2, mode="c"] U not None,
                    np.ndarray[double, ndim=2, mode="c"] V not None,
                    np.ndarray[double, ndim=2, mode="c"] UM not None,
                    np.ndarray[double, ndim=2, mode="c"] VM not None,
                    lmbda1, lmbda2, num_pairs, m, n, d):
    return boost_compute_loss_one_func(<long int*>np.PyArray_DATA(pairwise_data),
                    <double*>np.PyArray_DATA(weights),<double*>np.PyArray_DATA(U),<double*>np.PyArray_DATA(V),<double*>np.PyArray_DATA(UM),<double*>np.PyArray_DATA(VM),lmbda1, lmbda2, num_pairs,m, n, d)

def auc_computation(np.ndarray[long int, ndim=1, mode="c"] pos_inx not None,
                    np.ndarray[long int, ndim=1, mode="c"] neg_inx not None,
                    np.ndarray[double, ndim=1, mode="c"] val not None):
    num_pos, num_neg = len(pos_inx), len(neg_inx)
    return auc_computation_func(<long int*>np.PyArray_DATA(pos_inx), <long int*>np.PyArray_DATA(
                    neg_inx), <double *>np.PyArray_DATA(val), num_pos, num_neg)/float(num_pos*num_neg)

def compute_auc_list(pos_inx, neg_inx, val, measure):
    num_pos, num_neg = len(pos_inx), len(neg_inx)
    compute_auc_list_func(<long int*>np.PyArray_DATA(pos_inx), <long int*>np.PyArray_DATA(
                    neg_inx), <double *>np.PyArray_DATA(val), <double *>np.PyArray_DATA(measure), num_pos, num_neg)

def compute_auc_at_N_list(pos_inx, neg_inx, val, measure, N):
    num_pos, num_neg = len(pos_inx), len(neg_inx)
    compute_auc_at_N_list_func(<long int*>np.PyArray_DATA(pos_inx), <long int*>np.PyArray_DATA(
                    neg_inx), <double *>np.PyArray_DATA(val), <double *>np.PyArray_DATA(measure), num_pos, num_neg, N)

def compute_map_list(pos_sort, neg_sort, val, measure, num_pos, num_neg):
    compute_map_list_func(<long int*>np.PyArray_DATA(pos_sort), <long int*>np.PyArray_DATA(
                    neg_sort), <double *>np.PyArray_DATA(val), <double *>np.PyArray_DATA(measure), num_pos, num_neg)

def mean_average_precision(pos_inx, neg_inx, val):
    pos_val, neg_val = val[pos_inx], val[neg_inx]
    ii = np.argsort(pos_val)[::-1]
    jj = np.argsort(neg_val)[::-1]
    num_pos, num_neg = len(pos_inx), len(neg_inx)
    pos_sort, neg_sort = pos_val[ii], neg_val[jj]
    return mean_average_precision_func(<double *>np.PyArray_DATA(pos_sort),
                    <double *>np.PyArray_DATA(neg_sort), num_pos, num_neg)

def instance_boost_compute_loss(np.ndarray[long int, ndim=2, mode="c"] pairwise_data not None,
                    np.ndarray[double, ndim=1, mode="c"] weights not None,
                    np.ndarray[double, ndim=2, mode="c"] U not None,
                    np.ndarray[double, ndim=2, mode="c"] V not None,
                    num_pairs, loss_code, d):
    return instance_boost_compute_loss_func(<long int*>np.PyArray_DATA(pairwise_data),
                    <double*>np.PyArray_DATA(weights),<double*>np.PyArray_DATA(U),<double*>np.PyArray_DATA(V),num_pairs, loss_code, d)

def instance_boost_gradient_update(np.ndarray[long int, ndim=2, mode="c"] pairwise_data not None,
                    np.ndarray[double, ndim=1, mode="c"] weights not None,
                    np.ndarray[double, ndim=2, mode="c"] U not None,
                    np.ndarray[double, ndim=2, mode="c"] V not None,
                    theta, lmbda, num_pairs, loss_code, d, flag):
    instance_boost_gradient_update_func(<long int*>np.PyArray_DATA(pairwise_data), <double*>np.PyArray_DATA(weights),<double*>np.PyArray_DATA(U),<double*>np.PyArray_DATA(V), theta, lmbda, num_pairs, loss_code, d, flag)

def uniform_random_id(np.ndarray[int, ndim=1, mode="c"] X not None, num):
    return random_id(<int*>np.PyArray_DATA(X), num)

def sample_without_replacement(np.ndarray[int, ndim=1, mode="c"] population not None, num, size):
    flag = np.ones(num, dtype=np.int8)
    list = np.empty(num, dtype=np.int32)
    sample_without_replacement_func(<long int*>np.PyArray_DATA(population),
                                    <int*>np.PyArray_DATA(flag), num, <long int*>np.PyArray_DATA(list), size)
    return list

def adamf_gradient_update(np.ndarray[long int, ndim=2, mode="c"] data not None,
                        np.ndarray[double, ndim=1, mode="c"] ratings not None,
                        np.ndarray[double, ndim=1, mode="c"] weights not None,
                        np.ndarray[double, ndim=2, mode="c"] U not None,
                        np.ndarray[double, ndim=2, mode="c"] V not None,
                        theta, lmbda, num_pairs, m, d):
    adamf_gradient_update_func(<long int*>np.PyArray_DATA(data), <double *>np.PyArray_DATA(ratings), <double *>np.PyArray_DATA(weights), <double *>np.PyArray_DATA(U), <double *>np.PyArray_DATA(V), theta, lmbda, num_pairs, m, d)

def pointwise_gradient_update(np.ndarray[long int, ndim=2, mode="c"] data not None,
                        np.ndarray[double, ndim=1, mode="c"] ratings not None,
                        np.ndarray[double, ndim=1, mode="c"] weights not None,
                        np.ndarray[double, ndim=2, mode="c"] U not None,
                        np.ndarray[double, ndim=2, mode="c"] V not None,
                        theta, lmbda, num_pairs, d):
    pointwise_gradient_update_func(<long int*>np.PyArray_DATA(data), <double *>np.PyArray_DATA(ratings), <double *>np.PyArray_DATA(weights), <double *>np.PyArray_DATA(U), <double *>np.PyArray_DATA(V), theta, lmbda, num_pairs, d)


ctypedef np.float32_t REAL_t
DEF MAX_ITEM_NUM = 10000
DEF MAX_USER_NUM = 30000

def gbpr_optimize(model, pairwise_data):
    cdef int num_users = model.num_users
    cdef int num_items = model.num_items
    cdef int num_pairs = pairwise_data.shape[0]
    cdef int group_size = model.group_size
    cdef int num_factors = model.d
    cdef double reg_u = model.reg_u
    cdef double reg_i = model.reg_i
    cdef double rho = model.rho
    cdef double theta = model.theta

    cdef double *U = <double *>(np.PyArray_DATA(model.U))
    cdef double *V = <double *>(np.PyArray_DATA(model.V))
    #cdef long int *data = <long int*>np.PyArray_DATA(pairwise_data)

    cdef int size[MAX_ITEM_NUM]
    cdef long int *item_users[MAX_ITEM_NUM]

    for i in xrange(num_items):
        size[i] = <int>len(model.item_dict[i])
        item_users[i] = <long int *> np.PyArray_DATA(model.item_dict[i])

    inx = np.random.permutation(num_pairs)
    for u, i, j in pairwise_data[inx]:
        gbpr_gradient_update(U, V, u, i, j, item_users[i], size[i], group_size, num_factors, theta, reg_u, reg_i, rho)

def gbpr_sgd_train(model, train_data, Tr_neg):
    cdef int num_users = model.num_users
    cdef int num_items = model.num_items
    cdef int group_size = model.group_size
    cdef int num_factors = model.d
    cdef int max_iter = model.max_iter
    cdef double reg_u = model.reg_u
    cdef double reg_i = model.reg_i
    cdef double rho = model.rho
    cdef double theta = model.theta

    cdef double *U = <double *>(np.PyArray_DATA(model.U))
    cdef double *V = <double *>(np.PyArray_DATA(model.V))

    cdef long int num_pairs = train_data.shape[0]
    cdef long int *pair_data = <long int*>np.PyArray_DATA(train_data)

    cdef int size[MAX_ITEM_NUM]
    cdef long int *item_users[MAX_ITEM_NUM]
    cdef int neg_size[MAX_USER_NUM]
    cdef long int *neg_items[MAX_USER_NUM]
    cdef long int *index

    for i in xrange(num_items):
        size[i] = <int>len(model.item_dict[i])
        item_users[i] = <long int *> np.PyArray_DATA(model.item_dict[i])

    for u in xrange(num_users):
        neg_size[u] = <int> Tr_neg[u]["num"]
        neg_items[u] = <long int *> np.PyArray_DATA(Tr_neg[u]["items"])

    for t in xrange(max_iter):
        inx = np.random.permutation(num_pairs)
        index = <long int *>np.PyArray_DATA(inx)
        gbpr_fast_update(pair_data, index, neg_items, neg_size, item_users, size, U, V, group_size, num_pairs, num_factors, theta, reg_u, reg_i, rho)
        theta *=0.9

def adamf_sgd_optimization(model, ccf_index, pairs, ratings):
    weights = model.D*model.num_users
    cdef int num_factors = model.d
    cdef double reg_u = model.reg_u
    cdef double reg_i = model.reg_i
    cdef double theta = model.theta
    cdef int max_iter = model.max_iter
    cdef long int num_pairs = pairs.shape[0]
    cdef long int num_users = model.num_users
    cdef long int num_items = model.num_items

    cdef long int *pair_data = <long int*>(np.PyArray_DATA(pairs))
    cdef double *rating_data = <double *>(np.PyArray_DATA(ratings))

    cdef double *U = <double *>(np.PyArray_DATA(model.U[ccf_index]))
    cdef double *V = <double *>(np.PyArray_DATA(model.V[ccf_index]))
    cdef double *W = <double *>(np.PyArray_DATA(weights))
    cdef long int *index
    last_loss = mf_least_square_loss(pair_data, rating_data, W, U, V, num_pairs, num_users, num_items, num_factors, reg_u, reg_i)

    for t in xrange(max_iter):
        inx = np.random.permutation(num_pairs)
        index = <long int *>np.PyArray_DATA(inx)
        mf_sgd_optimize(pair_data, rating_data, index, W, U, V, num_pairs, num_factors, theta, reg_u, reg_i)
        curr_loss = mf_least_square_loss(pair_data, rating_data, W, U, V, num_pairs, num_users, num_items, num_factors, reg_u, reg_i)
        delta_loss = (curr_loss-last_loss)/last_loss
        if np.abs(delta_loss)<1e-12:
            break
        #print "epoch:%s, delta_loss:%s" %(t, delta_loss)
        last_loss = curr_loss
        theta *= 0.9

def warp_truncated_train(model, train_data, Tr, Tr_neg):
    cdef int num_users = model.num_users
    cdef int num_factors = model.num_factors
    cdef int max_iter = model.max_iter
    cdef int gamma = model.gamma
    cdef long int num_pairs = train_data.shape[0]
    cdef double theta = model.theta
    cdef double reg_u = model.reg_u
    cdef double reg_i = model.reg_i

    cdef double *U = <double*>(np.PyArray_DATA(model.U))
    cdef double *V = <double*>(np.PyArray_DATA(model.V))
    cdef long int *pair_data = <long int*>np.PyArray_DATA(train_data)
    cdef int pos_size[MAX_USER_NUM]
    cdef long int *pos_items[MAX_USER_NUM]
    cdef int neg_size[MAX_USER_NUM]
    cdef long int *neg_items[MAX_USER_NUM]
    cdef long int *index_pt

    for u in xrange(num_users):
        if u in Tr:
            pos_size[u] = <int>(Tr[u]['num'])
            neg_size[u] = <int>(Tr_neg[u]['num'])
            pos_items[u] = <long int *> np.PyArray_DATA(Tr[u]['items'])
            neg_items[u] = <long int *> np.PyArray_DATA(Tr_neg[u]['items'])

    UK, VK = model.U.copy(), model.V.copy()
    num_valid_users = max(int(np.floor(len(Tr.keys())*1)), 100)
    valid_users = np.random.choice(Tr.keys(), num_valid_users, replace=False)
    obj = 0.0
    for t in xrange(max_iter):
        inx = np.random.permutation(num_pairs)
        index_pt = <long int *>np.PyArray_DATA(inx)
        warp_fast_update(pair_data, index_pt, neg_items, neg_size, U, V, num_pairs, num_factors, gamma, theta, reg_u, reg_i)
        theta *= 0.9
        #warp_sample_update(pos_items, pos_size, neg_items, neg_size, U, V, num_users, num_factors, gamma, theta, reg_u, reg_i)
        predict_val = np.dot(model.U[valid_users, :], model.V.transpose())
        train_map=sum([mean_average_precision(Tr[valid_users[i]]['items'], Tr_neg[valid_users[i]]['items'], predict_val[i]) for i in xrange(num_valid_users)])/num_valid_users
        #print "MAP on valid data:%s" %train_map
        if train_map>obj:
            obj = train_map
            UK, VK = model.U.copy(), model.V.copy()
        else:
            model.U, model.V = UK, VK
            break     


def tfmap_construct_buffer(predict_val, Tr, Tr_neg, max_size, fastmode):
    buffer_set, lastrelpos = {}, {}
    for u in Tr:
        X = predict_val[u]
        pos_inx = Tr[u]["items"]
        neg_inx = Tr_neg[u]["items"]
        ii = np.where(X[neg_inx]>=np.min(X[pos_inx]))
        num = neg_inx[ii].shape[0]
        if num == 0:
            buffer_set[u] = pos_inx
            lastrelpos[u] = Tr[u]["num"]
        else:
            if num+Tr[u]["num"]>max_size:
                while(1):
                    n1 = np.random.randint(max_size)
                    n2 = max_size-n1
                    if n1<=num and n2<=Tr[u]["num"]:
                        break
                select_pos = np.random.choice(pos_inx, n2, replace=False)
                select_neg_num = n1
                select_pos_num = n2
            else:
                select_pos = pos_inx
                select_neg_num = num
                select_pos_num = Tr[u]["num"]
            if select_neg_num>0:
                if fastmode>0:
                    if fastmode<=select_pos_num:
                        jj=np.argsort(X[select_pos])[:fastmode]
                        inditem = select_pos[jj]
                    else:
                        inditem = select_pos
                else:
                    inditem = select_pos
                buffer_set[u] = inditem
                lastrelpos[u] = num+Tr[u]["num"]
            else:
                buffer_set[u] = pos_inx
                lastrelpos[u] = Tr[u]["num"]
    return buffer_set, lastrelpos

def tfmap_approx_loss(predict_val, Tr, U, V, reg_u, reg_i):
    loss = 0
    for u in Tr:
        pos_inx = Tr[u]["items"]
        num_pos = Tr[u]["num"]
        X = predict_val[u]
        loss += tfmap_approx_loss_fast(<double *>np.PyArray_DATA(X), <long int *>np.PyArray_DATA(pos_inx), num_pos)
    #loss = loss - 0.5*reg_u*np.linalg.norm(U, 'fro')**(2)-0.5*reg_i*np.linalg.norm(V, 'fro')**(2)
    return loss

def tfmap_sgd_optimizer(model, train_data, Tr, Tr_neg):
    cdef int num_users = model.num_users
    cdef int num_factors = model.num_factors
    cdef int max_size = model.max_size
    cdef int fastmode = model.fastmode
    cdef int max_iter = model.max_iter
    cdef double theta = model.theta
    cdef double reg_u = model.reg_u
    cdef double reg_i = model.reg_i

    cdef double *U = <double *>np.PyArray_DATA(model.U)
    cdef double *V = <double *>np.PyArray_DATA(model.V)
    predict_val = np.dot(model.U, model.V.transpose())
    last_loss = tfmap_approx_loss(predict_val, Tr, model.U, model.V, reg_u, reg_i)

    for t in xrange(max_iter):
        buffer_set, lastrelpos = tfmap_construct_buffer(predict_val, Tr, Tr_neg, max_size, fastmode)
        for u in buffer_set:
            if lastrelpos[u]>Tr[u]["num"]:
                tfmap_fast_update_item(<long int *>np.PyArray_DATA(buffer_set[u]),
                                       <double *>np.PyArray_DATA(predict_val[u]),
                                       buffer_set[u].shape[0], u, U, V, theta, reg_i, num_factors)

                tfmap_fast_update_user(<long int *>np.PyArray_DATA(buffer_set[u]),
                                       <double *>np.PyArray_DATA(predict_val[u]),
                                       buffer_set[u].shape[0], u, U, V, theta, reg_u, num_factors)

        predict_val = np.dot(model.U, model.V.transpose())
        curr_loss = tfmap_approx_loss(predict_val, Tr, model.U, model.V, reg_u, reg_i)

        train_map=sum([mean_average_precision(Tr[u]['items'], Tr_neg[u]['items'], predict_val[u])
                     for u in xrange(num_users)])/num_users
        
        delta_loss = (curr_loss-last_loss)/last_loss
        if (delta_loss>=0) and delta_loss<=1e-12:
            break
        #else:
        #    if delta_loss<0:
        #        theta = 0.5*model.theta
        #    else:
        #        theta = model.theta
        theta = 0.9* theta
        print "epoch:%s, curr_loss:%s, train map: %s" %(t+1, curr_loss, train_map)
        last_loss = curr_loss

def socf_bpr_train(model, valid_data, train_data, Tr, Tr_neg):
    
    cdef double tau_u = model.tau_u
    cdef double tau_i = model.tau_i
    cdef double reg_u = model.reg_u
    cdef double reg_i = model.reg_i
    cdef double theta = model.theta
    cdef int num_users = model.num_users
    cdef int num_items = model.num_items
    cdef int num_factors = model.d
    cdef int first_order = model.flag
    
    cdef long int num_pairs = train_data.shape[0]
    cdef long int* pair_data = <long int*>np.PyArray_DATA(train_data)
    cdef long int* valid_pt = <long int*>np.PyArray_DATA(valid_data)
    cdef long int num_valid = <long int>valid_data.shape[0]

    cdef double* U_pt = <double *> np.PyArray_DATA(model.U)
    cdef double* V_pt = <double *> np.PyArray_DATA(model.V)
    cdef double* rho_U = <double *>np.PyArray_DATA(model.rho_U)
    cdef double* rho_V = <double *>np.PyArray_DATA(model.rho_V)
    cdef double* hU = <double *>np.PyArray_DATA(model.hU)
    cdef double* hV = <double *>np.PyArray_DATA(model.hV)
    
    cdef int neg_size[MAX_USER_NUM]
    cdef long int *neg_items[MAX_USER_NUM]
    cdef long int *index_pt
    cdef double eta, *SigMu, *SigMv
    if model.flag == 2:
        eta = model.eta
        SigMu = <double *>np.PyArray_DATA(model.SigMu)
        SigMv = <double *>np.PyArray_DATA(model.SigMv)

    for u in xrange(num_users):
        neg_size[u] = <int> Tr_neg[u]["num"]
        neg_items[u] = <long int *> np.PyArray_DATA(Tr_neg[u]["items"])

    last_loss = compute_loss_func(valid_pt, U_pt, V_pt, reg_u, reg_i, num_valid, num_users,
                                  num_items, num_factors)

    for t in xrange(model.max_iter):
        tic = time.clock()
        inx = np.random.permutation(num_pairs)
        index_pt = <long int *>np.PyArray_DATA(inx)
        if first_order==1:
            socf_bpr_first_order(pair_data, index_pt, neg_items, neg_size, U_pt, V_pt, rho_U, rho_V, hU, hV, theta, reg_u, reg_i, tau_u, tau_i, num_pairs, num_factors)

        elif first_order==2:
            socf_bpr_second_order(pair_data, index_pt, neg_items, neg_size, U_pt, V_pt, rho_U, rho_V, hU, hV, SigMu, SigMv, reg_u, tau_u, tau_i, eta, num_pairs, num_factors)

        curr_loss = compute_loss_func(valid_pt, U_pt, V_pt, reg_u, reg_i, num_valid, num_users,
                                  num_items, num_factors)

        delta_loss = (curr_loss-last_loss)/last_loss
        print "epoch:%d, CurrLoss:%.6f, DeltaLoss:%.6f, Time:%.6f" % (
                    t+1, curr_loss, delta_loss, time.clock()-tic)
        if np.abs(delta_loss)<1e-5:
            break
        last_loss, theta = curr_loss, 0.9*theta
    print "complete the learning of socf bpr model"

def adawarp_component_train(model, ccf_inx, train_data, Tr, Tr_neg):
    cdef int num_users = model.num_users
    cdef int num_factors = model.d
    cdef int max_iter = model.max_iter
    cdef int gamma = model.gamma
    cdef long int num_pairs = model.num_logs
    cdef double theta = model.theta
    cdef double reg_u = model.lmbda
    cdef double reg_i = model.lmbda

    cdef double *U_pt = <double*>(np.PyArray_DATA(model.U[ccf_inx]))
    cdef double *V_pt = <double*>(np.PyArray_DATA(model.V[ccf_inx]))
    cdef double *W_pt = <double*>(np.PyArray_DATA(model.D))
    cdef long int *pair_data = <long int*>np.PyArray_DATA(train_data)
    cdef int pos_size[MAX_USER_NUM]
    cdef long int *pos_items[MAX_USER_NUM]
    cdef int neg_size[MAX_USER_NUM]
    cdef long int *neg_items[MAX_USER_NUM]
    cdef long int *index_pt

    for u in xrange(num_users):
        pos_size[u] = <int>(Tr[u]['num'])
        neg_size[u] = <int>(Tr_neg[u]['num'])
        pos_items[u] = <long int *> np.PyArray_DATA(Tr[u]['items'])
        neg_items[u] = <long int *> np.PyArray_DATA(Tr_neg[u]['items'])

    UK, VK = model.U[ccf_inx].copy(), model.V[ccf_inx].copy()
    num_valid_users = max(int(np.floor(num_users*1)), 100)
    valid_users = np.random.choice(num_users, num_valid_users, replace=False)
    obj = 0.0
    for t in xrange(max_iter):
        inx = np.random.permutation(num_pairs)
        index_pt = <long int *>np.PyArray_DATA(inx)
        adawarp_fast_update(pair_data, index_pt, neg_items, neg_size, U_pt, V_pt, W_pt, num_pairs, num_factors, gamma, theta, reg_u, reg_i)
        theta *= 0.9
        predict_val = np.dot(model.U[ccf_inx][valid_users, :], model.V[ccf_inx].transpose())
        train_map=sum([mean_average_precision(Tr[valid_users[i]]['items'], Tr_neg[valid_users[i]]['items'], predict_val[i]) for i in xrange(num_valid_users)])/num_valid_users
        #print "MAP on valid data:%s" %train_map
        if train_map>obj:
            obj = train_map
            UK, VK = model.U[ccf_inx].copy(), model.V[ccf_inx].copy()
        else:
            model.U[ccf_inx], model.V[ccf_inx] = UK, VK
            break


def adawrmf_sample_negative(Tr, Tr_neg, rho=1):
    data = []
    for u in Tr:
        data.extend([(u, i) for i in np.random.choice(Tr_neg[u]['items'], len(Tr[u].keys())*rho)])
    return data

def adawrmf_component_train(model, ccf_inx, pair_data, ratings, weights):
    
    cdef long int num_users = model.num_users
    cdef long int num_items = model.num_items
    cdef int max_iter = model.max_iter
    cdef int num_factors = model.d
    cdef double theta = model.theta
    cdef double lmbda = model.lmbda
    cdef long int num_pairs = pair_data.shape[0]

    cdef double *R_pt = <double*>(np.PyArray_DATA(ratings))
    cdef double *W_pt = <double*>(np.PyArray_DATA(weights))
    cdef double *U_pt = <double*>(np.PyArray_DATA(model.U[ccf_inx]))
    cdef double *V_pt = <double*>(np.PyArray_DATA(model.V[ccf_inx]))
    cdef long int *pair_pt = <long int*>np.PyArray_DATA(pair_data)
    cdef long int *index_pt

    last_loss = adawrmf_train_loss(pair_pt, R_pt, W_pt, U_pt, V_pt, num_pairs, num_users, num_items, num_factors, lmbda)

    for t in xrange(max_iter):
        inx = np.random.permutation(num_pairs)
        index_pt = <long int *>np.PyArray_DATA(inx)
        adawrmf_fast_train(index_pt, pair_pt, R_pt, W_pt, U_pt, V_pt, num_pairs, num_factors, theta, lmbda)
        curr_loss = adawrmf_train_loss(pair_pt, R_pt, W_pt, U_pt, V_pt, num_pairs, num_users, num_items, num_factors, lmbda)
        delta_loss = (curr_loss-last_loss)/last_loss
        #print " epoch: %s, Curr_loss:%s, Delta_loss:%s" %(t+1, curr_loss, delta_loss)
        if np.abs(delta_loss)<1e-5:
            break
        theta *= 0.9
        last_loss = curr_loss
