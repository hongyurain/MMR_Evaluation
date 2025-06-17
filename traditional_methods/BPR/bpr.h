
void gradient_update_func(long int *pairwise_data, double *U, double *V, double theta, double reg_u, double reg_i, long int num_pairs, int d);

double compute_loss_func(long int *pairwise_data, double *U, double *V, double reg_u, double reg_i, long int num_pairs, int m, int n, int d);

void boost_gradient_update_func(long int *pairwise_data, double * weights, double *U, double *V, double theta, double lmbda, long int num_pairs, int loss_code, int d, int flag);

double boost_compute_loss_func(long int *pairwise_data, double * weights, double *U, double *V, long int num_pairs, int loss_code, int d);

double boost_compute_loss_one_func(long int *pairwise_data, double * weights, double *U, double *V, double *UM, double *VM, double lmbda1, double lmbda2, long int num_pairs, int m, int n, int d);

void boost_gradient_update_one_func(long int *pairwise_data, double * weights, double *U, double *V, double *UM, double * VM,double theta, double lmbda1, double lmbda2, long int num_pairs, int m, int d);

long int auc_computation_func(long int *pos, long int *neg, double *val, long int num_pos, long int num_neg);

double mean_average_precision_func(double *pos_val, double *neg_val, long int num_pos, long int num_neg);

void compute_auc_list_func(long int *pos, long int *neg, double *val, double *measure, long int num_pos, long int num_neg);

void compute_auc_at_N_list_func(long int *pos, long int *neg, double *val, double *measure, long int num_pos, long int num_neg, int N);

void compute_map_list_func(long int *pos, long int *neg, double *val, double *measure, long int num_pos, long int num_neg);

void instance_boost_gradient_update_func(long int *pairwise_data, double * weights, double *U, double *V, double theta, double lmbda, long int num_pairs, int loss_code, int d, int flag);

double instance_boost_compute_loss_func(long int *pairwise_data, double * weights, double *U, double *V, long int num_pairs, int loss_code, int d);

int random_id(int *list, int size);

void sample_without_replacement_func(int *population, int *flag, int num, int*list, int size);

void adamf_gradient_update_func(long int *data, double *ratings, double * weights, double *U, double *V, double theta, double lmbda, long int num_pairs, int m, int d);

double sigmoid_func(double x);

double gradient_sigmoid_func(double x);

double tfmap_approx_loss_fast(double *predict_val, long int *pos_inx, int num_pos);

void tfmap_fast_update_user(long int *buffer_set, double *predict_val, int buffer_size, long int uid, double *U, double *V, double theta, double lmbda, int d);

void tfmap_fast_update_item(long int *buffer, double *predict_val, int buffer_size, long int uid, double *U, double *V, double theta, double lmbda, int d);

double warp_loss_func(long int *pos, long int *neg, double *val, long int num_pos, long int num_neg);

void warp_sgd_optimizer(long int *pair_data, long int *buffer, double *U, double *V, long int num_pair, int buffer_size, double lmbda, double theta);

void pointwise_gradient_update_func(long int *data, double *ratings, double * weights, double *U, double *V, double theta, double lmbda, long int num_pairs, int d);

void gbpr_gradient_update(double *U, double *V, long int uid, long int pid1, long int pid2, long int *item_users, int size, int group_size, int num_factors, double theta, double reg_u, double reg_i, double rho);

void gbpr_fast_update(long int *train_data, long int *index, long int **neg_items, int *neg_size, long int **item_users, int *size, double *U, double *V, int group_size, long int num_logs, int num_factors, double theta, double reg_u, double reg_i, double rho);

void mf_sgd_optimize(long int *pairs, double *ratings, long int *index, double *W, double *U, double *V, long int num, int num_factors, double theta, double reg_u, double reg_i);
double mf_least_square_loss(long int *pair_data, double *rating_data, double *W, double *U, double *V, long int num_pairs, long int num_users, long int num_items, int num_factors, double reg_u, double reg_i);

long int random_sample(long int *list, int size);
double warp_compute_rank_loss(float num, float t);
void warp_fast_update(long int *train_data, long int *index, long int **neg_items, int *neg_size, double *U, double *V, long int num_pairs, int num_factors, int gamma, double theta, double reg_u, double reg_i);

void warp_sample_update(long int **pos_items, int *pos_size, long int **neg_items, int *neg_size, double *U, double *V, int num_users, int num_factors, int gamma, double theta, double reg_u, double reg_i);

void adawarp_fast_update(long int *train_data, long int *index, long int **neg_items, int *neg_size, double *U, double *V, double *W, long int num_pairs, int num_factors, int gamma, double theta, double reg_u, double reg_i);

void socf_bpr_first_order(long int *pairwise_pt, long int *index_pt, long int **neg_items, int *neg_size, double *U, double *V, double *rho_U, double *rho_V, double *hU, double *hV, double theta, double reg_u, double reg_i, double tau_u, double tau_i, long int num_pairs, int num_factors);

void socf_bpr_second_order(long int *pairwise_pt, long int *index_pt, long int **neg_items, int *neg_size, double *U, double *V, double *rho_U, double *rho_V, double *hU, double *hV, double *SigMu, double *SigMv, double lmbda, double tau_u, double tau_i, double eta, long int num_pairs, int num_factors);

void adawrmf_fast_train(long int *index_pt, long int *pair_pt, double *R_pt, double *W_pt, double *U_pt, double *V_pt, long int num_pairs, int num_factors,double theta, double lmbda);

double adawrmf_train_loss(long int *pair_pt, double *R_pt, double *W_pt, double *U_pt, double *V_pt, long int num_pairs, long int num_users, long int num_items, int num_factors, double lmbda);