
#define min(x, y) (((x) < (y)) ? (x) : (y))
#define max(x, y) (((x) > (y)) ? (x) : (y))

#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

void gradient_update_func(long int *pairwise_data, double *U, double *V, double theta, double reg_u, double reg_i, long int num_pairs, int d){
    double val, uk, vk1, vk2;
    long int k, u, i, j; int r;
    for(k=0; k<num_pairs; k++){
        u=pairwise_data[k*3];
        i=pairwise_data[k*3+1]; 
        j=pairwise_data[k*3+2];
        val=0;        
        for(r=0; r<d; r++)
            val += U[u*d+r]*(V[i*d+r]-V[j*d+r]);
        //update the model parameters
        val = 1.0/(1.0+exp(val));

        for(r=0; r<d; r++){
            uk=U[u*d+r]; vk1=V[i*d+r]; vk2=V[j*d+r];
            U[u*d+r] += theta*(val*(vk1-vk2) - reg_u*uk);
            V[i*d+r] += theta*(val*uk - reg_i*vk1);
            V[j*d+r] += theta*(-val*uk - reg_i*vk2);
        }
    }
}

double compute_loss_func(long int *pairwise_data, double *U, double *V, double reg_u, double reg_i, long int num_pairs, int m, int n, int d){

    double val, loss=0;
    long int u, i, j, k; int r;

    for(k=0; k<num_pairs; k++){
        u=pairwise_data[k*3]; i=pairwise_data[k*3+1]; j=pairwise_data[k*3+2]; val=0;
        for(r=0; r<d; r++)
            val += U[u*d+r]*(V[i*d+r]-V[j*d+r]);
        loss += log(1.0+exp(-val));
    }
    val=0;
    for(u=0; u<m; u++){
        for(r=0; r<d; r++){
            val += U[u*d+r]*U[u*d+r];
        }
    }
    loss += 0.5*reg_u*val; val=0;

    for(i=0; i<n; i++){     
        for(r=0; r<d; r++){
            val += V[i*d+r]*V[i*d+r];
        }
    }
    loss += 0.5*reg_i*val;
    return loss;
}

void boost_gradient_update_func(long int *pairwise_data, double * weights, double *U, double *V, double theta, double lmbda, long int num_pairs, int loss_code, int d, int flag){

    double val, uk, vk1, vk2, w;
    long int k, u, i, j; int r;

    for(k=0; k<num_pairs; k++){
        u=pairwise_data[k*3]; i=pairwise_data[k*3+1]; j=pairwise_data[k*3+2]; val=0;        
        for(r=0; r<d; r++)
            val += U[u*d+r]*(V[i*d+r]-V[j*d+r]);
        w = weights[u];
        if (loss_code == 1)
            val = -1.0/(1.0+exp(val));
        else if(loss_code==2){
            val = exp(val);
            val = -val/pow((1+val), 2);
        }
        else if (loss_code==3){
            val = (val<1.0) ? -1.0 : 0.0;
        }
        else if(loss_code==4){
            val = -exp(-val);
        }
        else if(loss_code==5){
            val =val-1.0;
        }
        if(flag==0){
            for(r=0; r<d; r++){
                uk=U[u*d+r]; vk1=V[i*d+r]; vk2=V[j*d+r];
                U[u*d+r] -= theta*(val*w*(vk1-vk2)+lmbda*uk);
                V[i*d+r] -= theta*(val*w*uk+lmbda*vk1);
                V[j*d+r] -= theta*(-val*w*uk+lmbda*vk2);
            }
        }
        else if(flag==1){
            for(r=0; r<d; r++){
                uk=U[u*d+r]; vk1=V[i*d+r]; vk2=V[j*d+r];
                U[u*d+r] -= theta*(val*w*(vk1-vk2)+lmbda*uk);
            }
        }
        else if(flag==2){
            for(r=0; r<d; r++){
                uk=U[u*d+r]; vk1=V[i*d+r]; vk2=V[j*d+r];
                V[i*d+r] -= theta*(val*w*uk+lmbda*vk1);
                V[j*d+r] -= theta*(-val*w*uk+lmbda*vk2);
            }
        }
    }
}

void boost_gradient_update_one_func(long int *pairwise_data, double * weights, double *U, double *V, double *UM, double * VM,double theta, double lmbda1, double lmbda2, long int num_pairs, int m, int d){

    double val, uk, vk1, vk2, ukm, vkm1, vkm2, w;
    long int k, u, i, j; int r;

    for(k=0; k<num_pairs; k++){
        u=pairwise_data[k*3]; i=pairwise_data[k*3+1]; j=pairwise_data[k*3+2]; val=0;        
        for(r=0; r<d; r++)
            val += U[u*d+r]*(V[i*d+r]-V[j*d+r]);
        w=weights[u]*m;
        val=-1.0/(1.0+exp(w*val));
        // printf("%g\n", w);
        for(r=0; r<d; r++){
            uk=U[u*d+r]; vk1=V[i*d+r]; vk2=V[j*d+r];
            ukm=UM[u*d+r]; vkm1=VM[i*d+r]; vkm2=VM[j*d+r];
            U[u*d+r] -= theta*(val*w*(vk1-vk2)+lmbda1*uk+lmbda2*(uk-ukm));
            V[i*d+r] -= theta*(val*w*uk+lmbda1*vk1+lmbda2*(vk1-vkm1));
            V[j*d+r] -= theta*(-val*w*uk+lmbda1*vk2+lmbda2*(vk2-vkm2));
        }
    }
}

double boost_compute_loss_func(long int *pairwise_data, double * weights, double *U, double *V, long int num_pairs, int loss_code, int d){

    double val, w, loss=0;
    long int u, i, j, k; int r;

    for(k=0; k<num_pairs; k++){
        u=pairwise_data[k*3]; i=pairwise_data[k*3+1]; j=pairwise_data[k*3+2]; val=0;
        for(r=0; r<d; r++)
            val += U[u*d+r]*(V[j*d+r]-V[i*d+r]);
        w=weights[u];
        if (loss_code==1)
            loss += w*log(1.0+exp(val));
        else if (loss_code==2)
            loss += -w*(1.0/(1.0+exp(val)));
        else if (loss_code==3){
            val += 1.0;
            loss += w*((val>0.0)? val : 0.0);
        }
        else if(loss_code==4){
            loss += w*exp(val);
        }
        else if(loss_code==5){
            loss += 0.5*w*pow((1.0+val), 2);
        }
    }
    // val=0;
    // for(u=0; u<m; u++){
    //     for(r=0; r<d; r++){
    //         val += U[u*d+r]*U[u*d+r];
    //     }
    // }
    // loss += 0.5*lmbda*val; val=0;

    // for(i=0; i<n; i++){     
    //     for(r=0; r<d; r++){
    //         val += V[i*d+r]*V[i*d+r];
    //     }
    // }
    // loss += 0.5*lmbda*val;
    return loss;
}

double boost_compute_loss_one_func(long int *pairwise_data, double * weights, double *U, double *V, double *UM, double *VM, double lmbda1, double lmbda2, long int num_pairs, int m, int n, int d){

    double val, w, loss=0;
    long int u, i, j, k; int r;

    for(k=0; k<num_pairs; k++){
        u=pairwise_data[k*3]; i=pairwise_data[k*3+1]; j=pairwise_data[k*3+2]; val=0;
        for(r=0; r<d; r++)
            val += U[u*d+r]*(V[j*d+r]-V[i*d+r]);
        w=weights[u]*m;
        loss += log(1.0+exp(w*val));
    }
    val=0;
    for(u=0; u<m; u++){
        for(r=0; r<d; r++){
            // val += U[u*d+r]*U[u*d+r];
            val += 0.5*lmbda1*pow(U[u*d+r], 2)+0.5*lmbda2*pow(U[u*d+r]-UM[u*d+r], 2);
        }
    }
    loss += val; val=0;

    for(i=0; i<n; i++){     
        for(r=0; r<d; r++){
            val += 0.5*lmbda1*pow(V[i*d+r],2)+0.5*lmbda2*pow(V[i*d+r]-VM[i*d+r], 2);
        }
    }
    loss += val;
    return loss;
}


long int auc_computation_func(long int *pos, long int *neg, double *val, long int num_pos, long int num_neg){
    long int i, j, num=0;
    for(i=0; i<num_pos; i++){
        for(j=0; j<num_neg; j++){
            if(val[pos[i]]>val[neg[j]])
                num++;
        }
    }
    return num;
}

double mean_average_precision_func(double *pos_val, double *neg_val, long int num_pos, long int num_neg){
    long int i, j;
    double num, map=0.0;

    for(i=0; i<num_pos; i++){
        num = 0.0;
        for(j=0; j<num_neg;j++){
            if(pos_val[i]<=neg_val[j])
                num += 1.0;
        }
        map += (i+1)/(i+num+1);
    }

    return (map/num_pos);
}

void compute_auc_list_func(long int *pos, long int *neg, double *val, double *measure, long int num_pos, long int num_neg){
    
    long int i, j;
    double num;
    for(i=0; i<num_pos; i++){
        num = 0.0;
        for(j=0; j<num_neg; j++){
            if(val[pos[i]]>val[neg[j]])
                num+=1.0;
            // printf("%g,%g\n", val[pos[i]], val[neg[j]]);
        }
        measure[i]=num/num_neg;
    }
}

void compute_auc_at_N_list_func(long int *pos, long int *neg, double *val, double *measure, long int num_pos, long int num_neg, int N){
    
    long int i, j;
    double num;
    for(i=0; i<num_pos; i++){
        num = 0.0;
        for(j=0; j<num_neg; j++){
            if(val[pos[i]]<=val[neg[j]])
                num+=1.0;
            // printf("%g,%g\n", val[pos[i]], val[neg[j]]);
        }
        if(num>=N)
            measure[i]=0.0;
        else
            measure[i]=1.0 - num/N;
    }
}


void compute_map_list_func(long int *pos, long int *neg, double *val, double *measure, long int num_pos, long int num_neg){
    
    long int i, j;
    double num;
    for(i=0; i<num_pos; i++){
        num = 0.0;
        for(j=0; j<num_neg; j++){
            if(val[pos[i]]<=val[neg[j]])
                num += 1.0;
            // printf("%g,%g\n", val[pos[i]], val[neg[j]]);
        }
        measure[i]=(i+1)/(i+num+1);
    }
}

double instance_boost_compute_loss_func(long int *pairwise_data, double * weights, double *U, double *V, long int num_pairs, int loss_code, int d){

    double val, w, loss=0;
    long int u, i, j, k; int r;

    for(k=0; k<num_pairs; k++){
        u=pairwise_data[k*3]; i=pairwise_data[k*3+1]; j=pairwise_data[k*3+2]; val=0;
        for(r=0; r<d; r++)
            val += U[u*d+r]*(V[j*d+r]-V[i*d+r]);
        w=weights[k];
        if (loss_code==1)
            loss += w*log(1.0+exp(val));
        else if (loss_code==2)
            loss += -w*(1.0/(1.0+exp(val)));
        else if (loss_code==3){
            val += 1.0;
            loss += w*((val>0.0)? val : 0.0);
        }
        else if(loss_code==4){
            loss += w*exp(val);
        }
        else if(loss_code==5){
            loss += 0.5*w*pow((1.0+val), 2);
        }
    }
    return loss;
}

void instance_boost_gradient_update_func(long int *pairwise_data, double * weights, double *U, double *V, double theta, double lmbda, long int num_pairs, int loss_code, int d, int flag){

    double val, uk, vk1, vk2, w;
    long int k, u, i, j; int r;

    for(k=0; k<num_pairs; k++){
        u=pairwise_data[k*3]; i=pairwise_data[k*3+1]; j=pairwise_data[k*3+2]; val=0;        
        for(r=0; r<d; r++)
            val += U[u*d+r]*(V[i*d+r]-V[j*d+r]);
        w = weights[k];
        if (loss_code == 1)
            val = -1.0/(1.0+exp(val));
        else if(loss_code==2){
            val = exp(val);
            val = -val/pow((1+val), 2);
        }
        else if (loss_code==3){
            val = (val<1.0) ? -1.0 : 0.0;
        }
        else if(loss_code==4){
            val = -exp(-val);
        }
        else if(loss_code==5){
            val =val-1.0;
        }

        if(flag==0){
            for(r=0; r<d; r++){
                uk=U[u*d+r]; vk1=V[i*d+r]; vk2=V[j*d+r];
                U[u*d+r] -= theta*(val*w*(vk1-vk2)+lmbda*uk);
                V[i*d+r] -= theta*(val*w*uk+lmbda*vk1);
                V[j*d+r] -= theta*(-val*w*uk+lmbda*vk2);
            }
        }
        else if(flag==1){
            for(r=0; r<d; r++){
                uk=U[u*d+r]; vk1=V[i*d+r]; vk2=V[j*d+r];
                U[u*d+r] -= theta*(val*w*(vk1-vk2)+lmbda*uk);
            }
        }
        else if(flag==2){
            for(r=0; r<d; r++){
                uk=U[u*d+r]; vk1=V[i*d+r]; vk2=V[j*d+r];
                V[i*d+r] -= theta*(val*w*uk+lmbda*vk1);
                V[j*d+r] -= theta*(-val*w*uk+lmbda*vk2);
            }
        }
    }
}

int random_id(int *list, int size){
    int i;
    i = (rand() % size);
    return list[i];
}

void sample_without_replacement_func(long int *population, int *flag, long int num, long int*list, long int size){
    long int i, m=0;
    while(m<size){
        i = (rand() % num);
        if(flag[i]){
            list[m] = population[i];
            flag[i] = 0;
            m++;
        }
    }
}

void adamf_gradient_update_func(long int *data, double *ratings, double * weights, double *U, double *V, double theta, double lmbda, long int num_pairs, int m, int d){

    double val, uk, vk, w, rat;
    long int k, u, i; int r;

    for(k=0; k<num_pairs; k++){
        u=data[k*2]; i=data[k*2+1];
        rat=ratings[k]; val=0;        
        for(r=0; r<d; r++)
            val += U[u*d+r]*V[i*d+r];
        w=weights[u]*m;
        val=rat-val;
        for(r=0; r<d; r++){
            uk=U[u*d+r]; vk=V[i*d+r];
            U[u*d+r] += theta*(val*w*vk-lmbda*uk);
            V[i*d+r] += theta*(val*w*uk-lmbda*vk);
        }
    }
}

double sigmoid_func(double x){
    double val = exp(-x);
    return (1.0/(1.0+val));
}

double gradient_sigmoid_func(double x){
    double val=exp(x);
    return val/pow((1.0+val), 2);
}

void tfmap_fast_update_user(long int *buffer_set, double *predict_val, int buffer_size, long int uid, double *U, double *V, double theta, double lmbda, int d){
    long int pid1, pid2; int r, i, j;
    double v1, v2, v3, val1, val2, vk;

    double *worker1=(double *) malloc(sizeof(double)*buffer_size);

    for(i=0; i<buffer_size; i++){
        worker1[i] = 0.0;
    }
    for(i=0; i<buffer_size; i++){
        pid1 = buffer_set[i];
        v1 = sigmoid_func(predict_val[pid1]);
        v2 = gradient_sigmoid_func(predict_val[pid1]);
        for(j=0; j<buffer_size; j++){
            pid2 = buffer_set[j];
            v3 = predict_val[pid2]-predict_val[pid1];
            val1 = v2*sigmoid_func(v3);
            val2 = v1*gradient_sigmoid_func(v3);
            worker1[i] += val1-val2;
            worker1[j] += val2;
        }
    }
    for(r=0; r<d; r++){
        vk = 0;
        for(i=0; i<buffer_size;i++){
            pid1 = buffer_set[i];
            vk += worker1[i]*V[pid1*d+r];
        }
        U[uid*d+r] += theta*((1.0/(double) buffer_size)*vk - lmbda*U[uid*d+r]);
    }
    free(worker1);
}

void tfmap_fast_update_item(long int *buffer, double *predict_val, int buffer_size, long int uid, double *U, double *V, double theta, double lmbda, int d){

    long int item1, item2;
    int i, j, r;
    double val1, val2, val3, sum;

    for(i=0; i<buffer_size; i++){
        sum=0.0;
        item1 = buffer[i];
        val1 = predict_val[item1];
        for(j=0; j<buffer_size; j++){
            item2 = buffer[j];
            val2 = predict_val[item2];
            val3 = val2-val1;
            sum += sigmoid_func(val3)*gradient_sigmoid_func(val1)+(sigmoid_func(val2)-sigmoid_func(val1))*gradient_sigmoid_func(val3);
        }
        for(r=0; r<d; r++)
            V[item1*d+r] += theta*(U[uid*d+r]*sum/((double) buffer_size)-lmbda*V[item1*d+r]);
    }
}

double tfmap_approx_loss_fast(double *predict_val, long int *pos_inx, int num_pos){
    int i, j; double val, loss=0;
    long int pid1, pid2;

    for(i=0; i<num_pos; i++){
        pid1 = pos_inx[i]; val = 0.0;
        for(j=0; j<num_pos; j++){
            pid2 = pos_inx[j];
            val += sigmoid_func(predict_val[pid2]-predict_val[pid1]);
        }
        loss += val*sigmoid_func(predict_val[pid1]);
    }
    return loss/num_pos;
}


double warp_loss_func(long int *pos, long int *neg, double *val, long int num_pos, long int num_neg){

    long int i, j, item1, item2, num;
    double loss=0.0;
    for(i=0; i<num_pos; i++){
        num = 0;
        item1 = pos[i];
        for(j=0; j<num_neg; j++){
            item2 = neg[j];
            if(val[item2] > val[item1])
                num++;
        }
        for(j=0; j<num; j++)
            loss += 1.0/(1.0+num);        
    }
    return loss;
}

void warp_sgd_optimizer(long int *pair_data, long int *buffer, double *U, double *V, long int num_pair, int buffer_size, double lmbda, double theta){
    
    //long int u, i, j;
}

void pointwise_gradient_update_func(long int *data, double *ratings, double * weights, double *U, double *V, double theta, double lmbda, long int num_pairs, int d){

    double val, uk, vk, g;
    long int k, u, i; int r;

    for(k=0; k<num_pairs; k++){
        u=data[k*2]; i=data[k*2+1]; val=0;        
        for(r=0; r<d; r++)
            val += U[u*d+r]*V[i*d+r];
        g = exp(val);
        g = (ratings[k]*(1.0+g)*g-g*g)/pow((1.0+g), 3);
        for(r=0; r<d; r++){
            uk=U[u*d+r]; vk=V[i*d+r];
            U[u*d+r] -= theta*(weights[u]*g*vk+lmbda*uk);
            V[i*d+r] -= theta*(weights[u]*g*uk+lmbda*vk);
        }
    }
}


void gbpr_gradient_update(double *U, double *V, long int uid, long int pid1, long int pid2, long int *item_users, int size, int group_size, int num_factors, double theta, double reg_u, double reg_i, double rho){

    double val, val1, val2, uk, uk1, uk2, vk1, vk2;
    int r, m, _gsize, i; int *label;
    long int u, *group;

    // random smapling a group of users
    if (size<=group_size){
        _gsize = size;
        group = item_users;
    }
    else{
        label = (int *) malloc(sizeof(int)*size);
        for(i=0; i<size; i++)
            label[i]=0; // initialize
        group = (long int *) malloc(sizeof(long int)*_gsize);
        _gsize = group_size;
        group[0] = uid;
        m = 0;
        while(m<group_size-1){
            i = (rand() % size);
            if((item_users[i] != uid)&&(label[i]==0)){
                m++;
                group[m] = item_users[i];
                label[i] = 1;
            }
        }
    }
    val = 0; val1 = 0; val2 = 0;
    for(r=0; r<num_factors; r++){
        for(m=0; m<_gsize; m++){
            u = group[m];
            val += U[u*num_factors+r]*V[pid1*num_factors+r];
        }
        val1 += U[uid*num_factors+r]*V[pid1*num_factors+r];
        val2 += U[uid*num_factors+r]*V[pid2*num_factors+r];
    }

    val = rho*val/_gsize + (1-rho)*val1 - val2;
    val = 1.0/(exp(val)+1.0);
    
    for(r=0; r<num_factors; r++){
        uk = U[uid*num_factors+r]; uk1 = uk;
        vk1 = V[pid1*num_factors+r]; vk2 = V[pid2*num_factors+r];
        U[uid*num_factors+r] += theta*(val*(rho*vk1/_gsize+(1-rho)*vk1-vk2)-reg_u*uk);
        for(m=1; m<_gsize; m++){
            u = group[m];
            uk2 = U[u*num_factors+r];
            U[u*num_factors+r] += theta*(val*rho*vk1/_gsize-reg_i*uk2);
            uk1 += uk2;
        }
        V[pid1*num_factors+r] += theta*(val*((1-rho)*uk+rho*uk1/_gsize)-reg_i*vk1);
        V[pid2*num_factors+r] += theta*(-val*uk-reg_i*vk2);
    }
    if(size>group_size){
        free(label);
        free(group);
    }    
}

void gbpr_fast_update(long int *train_data, long int *index, long int **neg_items, int *neg_size, long int **item_users, int *size, double *U, double *V, int group_size, long int num_logs, int num_factors, double theta, double reg_u, double reg_i, double rho){

    long int uid, pid1, pid2, k1, k2, t;

    for(t=0; t< num_logs; t++){
        k1 = index[t];
        uid = train_data[k1*2];
        pid1 = train_data[k1*2+1];
        k2 = (rand() % neg_size[uid]);
        pid2 = neg_items[uid][k2];
        gbpr_gradient_update(U, V, uid, pid1, pid2, item_users[pid1], size[pid1], group_size,num_factors, theta, reg_u, reg_i, rho);
        
        // printf("%d, %d, %d\n", uid, pid1, pid2);
        // for(k1=0; k1<neg_size[uid]; k1++)
        //     printf("%d ", neg_items[uid][k1]);
        // printf("%d \n", neg_size[uid]);
    }
}

void mf_sgd_optimize(long int *pairs, double *ratings, long int *index, double *W,double *U, double *V, long int num, int num_factors, double theta, double reg_u, double reg_i){

    long int u, i, k, inx; int d;
    double r, val, uk, vk;

    for(k=0; k<num; k++){
        inx = index[k];
        u = pairs[2*inx]; i = pairs[2*inx+1]; r = ratings[inx];
        // printf("%d, %d, %f, %d\n", u, i, r, inx);
        // break;
        val = 0;
        for(d=0; d<num_factors; d++)
            val += U[u*num_factors+d]*V[i*num_factors+d];
        val = r - val;

        for(d=0; d<num_factors; d++){
            uk = U[u*num_factors+d]; vk = V[i*num_factors+d];
            U[u*num_factors+d] += theta*(W[u]*val*vk - reg_u*uk);
            V[i*num_factors+d] += theta*(W[u]*val*uk - reg_i*vk);
        }
    }
}


double mf_least_square_loss(long int *pair_data, double *rating_data, double *W, double *U, double *V, long int num_pairs, long int num_users, long int num_items, int num_factors, double reg_u, double reg_i){

    long int u, i, k; int d;
    double r, val, loss=0;
    for(k=0; k<num_pairs; k++){
        u = pair_data[k*2]; i=pair_data[k*2+1]; r = rating_data[k];
        val = 0;
        for(d=0; d<num_factors; d++)
            val += U[u*num_factors+d]*V[i*num_factors+d];
        val = r-val;
        loss += 0.5*pow(val, 2)*W[u];
    }
    val = 0;
    for(u=0; u<num_users; u++){
        for(d=0; d<num_factors; d++)
            val += pow(U[u*num_factors+d], 2);
    }
    loss += 0.5*reg_u*val;
    val =0;
    for(i=0; i<num_items; i++){
        for(d=0; d<num_factors; d++)
            val += pow(V[i*num_factors+d], 2);
    }
    loss += 0.5*reg_i*val;
    return (loss);
}

long int random_sample(long int *list, int size){
    int i;
    i = (rand() % size);
    return list[i];
}

double warp_compute_rank_loss(float num, float t){
    int i; double loss=0;
    int rank =(int) floor((num-1)/t);
    for(i=1; i<rank+1; i++)
        loss += 1.0/i;
    return loss;
}

void warp_fast_update(long int *train_data, long int *index, long int **neg_items, int *neg_size, double *U, double *V, long int num_pairs, int num_factors, int gamma, double theta, double reg_u, double reg_i){

    long int uid, pid1, pid2, k, k1;
    double val1, val2, loss, uk, vk1, vk2, rank, sg1, sg2, g1, g2;
    int d, t; int flag;

    for(k=0; k<num_pairs; k++){
        k1 = index[k];
        uid = train_data[2*k1]; pid1 = train_data[2*k1+1];
        val1=0;
        for(d=0; d<num_factors; d++)
            val1 += U[uid*num_factors+d]*V[pid1*num_factors+d];
        t = 0; flag = 0;
        while(t<neg_size[uid]/gamma){
            t++;
            k1 = (rand() % neg_size[uid]);
            pid2 = neg_items[uid][k1];            
            val2 = 0;
            for(d=0; d<num_factors; d++)
                val2 += U[uid*num_factors+d]*V[pid2*num_factors+d];
            if(val1<val2+1){
                flag=1;
                break;
            }
        }
        // update the model parameters
        if(flag){
            loss = warp_compute_rank_loss((float) neg_size[uid], (float) t);
            // rank = floor((neg_size[uid]-1)/t);
            // sg1 = sigmoid_func(val1);
            // sg2 = sigmoid_func(val2-val1);
            // g1 = gradient_sigmoid_func(val1);
            // g2 = gradient_sigmoid_func(val2-val1);

            for(d=0; d<num_factors; d++){
                uk = loss*(V[pid2*num_factors+d]-V[pid1*num_factors+d])+reg_u*U[uid*num_factors+d];
                vk1 = -loss*U[uid*num_factors+d] + reg_i*V[pid1*num_factors+d];
                vk2 = loss*U[uid*num_factors+d] + reg_i*V[pid2*num_factors+d];

                // uk = g1*V[pid1*num_factors+d]*rank*sg2 + rank*sg1*g2*(V[pid2*num_factors+d]-V[pid1*num_factors+d])+reg_u*U[uid*num_factors+d];
                // vk1 = g1*U[uid*num_factors+d]*rank*sg2 - rank*sg1*g2*U[uid*num_factors+d] + reg_i*V[pid1*num_factors+d];
                // vk2 = rank*sg1*g2*U[uid*num_factors+d] + reg_i*V[pid2*num_factors+d];

                U[uid*num_factors+d] -= theta*uk;
                V[pid1*num_factors+d] -= theta*vk1;
                V[pid2*num_factors+d] -= theta*vk2;

            }
        }
        else{
            for(d=0; d<num_factors; d++){
                uk = reg_u*U[uid*num_factors+d];
                vk1 = reg_i*V[pid1*num_factors+d];
                U[uid*num_factors+d] -= theta*uk;
                V[pid1*num_factors+d] -= theta*vk1;
            }
        }
    }
}

void adawarp_fast_update(long int *train_data, long int *index, long int **neg_items, int *neg_size, double *U, double *V, double *W, long int num_pairs, int num_factors, int gamma, double theta, double reg_u, double reg_i){

    long int uid, pid1, pid2, k, k1;
    double w, val1, val2, loss, uk, vk1, vk2;
    int d, t; int flag;

    for(k=0; k<num_pairs; k++){
        k1 = index[k];
        uid = train_data[2*k1]; pid1 = train_data[2*k1+1]; w=W[k1];
        val1=0;
        for(d=0; d<num_factors; d++)
            val1 += U[uid*num_factors+d]*V[pid1*num_factors+d];
        t = 0; flag = 0;
        while(t<neg_size[uid]/gamma){
            t++;
            k1 = (rand() % neg_size[uid]);
            pid2 = neg_items[uid][k1];            
            val2 = 0;
            for(d=0; d<num_factors; d++)
                val2 += U[uid*num_factors+d]*V[pid2*num_factors+d];
            if(val1<val2+1){
                flag=1;
                break;
            }
        }
        // update the model parameters
        if(flag){
            loss = warp_compute_rank_loss((float) neg_size[uid], (float) t);
            // loss = 1.0;
            for(d=0; d<num_factors; d++){
                uk = w*loss*(V[pid2*num_factors+d]-V[pid1*num_factors+d])+reg_u*U[uid*num_factors+d];
                vk1 = -w*loss*U[uid*num_factors+d] + reg_i*V[pid1*num_factors+d];
                vk2 = w*loss*U[uid*num_factors+d] + reg_i*V[pid2*num_factors+d];
                U[uid*num_factors+d] -= theta*uk;
                V[pid1*num_factors+d] -= theta*vk1;
                V[pid2*num_factors+d] -= theta*vk2;
            }
        }
        else{
            for(d=0; d<num_factors; d++){
                uk = reg_u*U[uid*num_factors+d];
                vk1 = reg_i*V[pid1*num_factors+d];
                U[uid*num_factors+d] -= theta*uk;
                V[pid1*num_factors+d] -= theta*vk1;
            }
        }
    }
}

void warp_sample_update(long int **pos_items, int *pos_size, long int **neg_items, int *neg_size, double *U, double *V, int num_users, int num_factors, int gamma, double theta, double reg_u, double reg_i){

    long int uid, pid1, pid2;
    double val1, val2, loss, uk, vk1, vk2;
    int i, t, k, d, flag;
    
    for(i=0; i<num_users; i++){
        // random sampling a positive interaction pair
        uid = (rand() % num_users);
        k = (rand() % pos_size[uid]);
        pid1 = pos_items[uid][k];
        val1 = 0;
        for(d=0; d<num_factors; d++)
            val1 += U[uid*num_factors+d]*V[pid1*num_factors+d];
        
        t = 0; flag = 0;
        while(t<(neg_size[uid]/gamma)){
            t++;
            k = (rand() % neg_size[uid]);
            pid2 = neg_items[uid][k];
            val2 = 0;
            for(d=0; d<num_factors; d++)
                val2 += U[uid*num_factors+d]*V[pid2*num_factors+d];
            if(val1<val2+1){
                flag = 1;
                break;
            }
        }
        if(flag){
            loss = warp_compute_rank_loss((float) neg_size[uid], (float) t);
            for(d=0; d<num_factors; d++){
                uk = loss*(V[pid2*num_factors+d]-V[pid1*num_factors+d]) + reg_u*U[uid*num_factors+d];
                vk1 = -loss*U[uid*num_factors+d] + reg_i*V[pid1*num_factors+d];
                vk2 = loss*U[uid*num_factors+d] + reg_i*V[pid2*num_factors+d];
                U[uid*num_factors+d] -= theta*uk;
                V[pid1*num_factors+d] -= theta*vk1;
                V[pid2*num_factors+d] -= theta*vk2;
            }
        }
        // else{
        //     for(d=0; d<num_factors; d++){
        //         uk = reg_u*U[uid*num_factors+d];
        //         vk1 = reg_i*V[pid1*num_factors+d];
        //         U[uid*num_factors+d] -= theta*uk;
        //         V[pid1*num_factors+d] -= theta*vk1;
        //     }
        // }
    }
}


void socf_bpr_first_order(long int *pairwise_pt, long int *index_pt, long int **neg_items, int *neg_size, double *U, double *V, double *rho_U, double *rho_V, double *hU, double *hV, double theta, double reg_u, double reg_i, double tau_u, double tau_i, long int num_pairs, int num_factors){

    long int t, k, uid, pid1, pid2; int d;
    double val=0, x, uk, vk1, vk2;

    for(t=0; t<num_pairs; t++){
        k = index_pt[t];
        uid = pairwise_pt[2*k];
        pid1 = pairwise_pt[2*k+1];
        k = (rand() % neg_size[uid]);
        pid2 = neg_items[uid][k];
        val = 0;
        for(d=0; d<num_factors; d++)
            val += U[uid*num_factors+d]*(V[pid1*num_factors+d]-V[pid2*num_factors+d]);
        // val = (val<1.0) ? -1.0 : 0.0;
        val = -1.0/(1.0+exp(val));
        // update the model parameters
        for(d=0; d<num_factors; d++){
            uk = U[uid*num_factors+d];
            vk1 = V[pid1*num_factors+d];
            vk2 = V[pid2*num_factors+d];
            if (tau_u==0){
                U[uid*num_factors+d] = uk - theta*(val*(vk1-vk2)+reg_u*uk);
            }
            else{
                if(uk!=0){
                    rho_U[uid*num_factors+d] += theta*tau_u;
                    x = uk - theta*(val*(vk1-vk2)+reg_u*uk);
                    if(x>0){
                        // y = x-(rho_U[uid*num_factors+d]+hU[uid*num_factors+d]);
                        // U[uid*num_factors+d] = (y>=0? y : 0);
                        U[uid*num_factors+d] = max(0, x-(rho_U[uid*num_factors+d]+hU[uid*num_factors+d]));
                    }
                    else if(x<0){
                        // y = x+(rho_U[uid*num_factors+d]-hU[uid*num_factors+d]);
                        // U[uid*num_factors+d] = (y<=0? y : 0);
                        U[uid*num_factors+d] = min(0, x+(rho_U[uid*num_factors+d]-hU[uid*num_factors+d]));
                    }
                    hU[uid*num_factors+d] += U[uid*num_factors+d]-x;
                }
            }
            if (tau_i==0){
                V[pid1*num_factors+d] = vk1-theta*(val*uk+reg_i*vk1);
                V[pid2*num_factors+d] = vk2-theta*(-val*uk+reg_i*vk2);
            }
            else{
                if(vk1!=0){
                    rho_V[pid1*num_factors+d] += theta*tau_i;
                    x = vk1-theta*(val*uk+reg_i*vk1);
                    if(x>0){
                        // y = x-(rho_V[pid1*num_factors+d]+hV[pid1*num_factors+d]);
                        // V[pid1*num_factors+d] = (y>=0? y : 0);
                        V[pid1*num_factors+d] = max(0, x-(rho_V[pid1*num_factors+d]+hV[pid1*num_factors+d]));
                    }
                    else if(x<0){
                        // y = x+(rho_V[pid1*num_factors+d]-hV[pid1*num_factors+d]);
                        // V[pid1*num_factors+d] = (y<=0? y : 0);
                        V[pid1*num_factors+d] = max(0, x+(rho_V[pid1*num_factors+d]-hV[pid1*num_factors+d]));
                    }
                    hV[pid1*num_factors+d] += V[pid1*num_factors+d]-x;
                }

                if(vk2!=0){
                    rho_V[pid2*num_factors+d] += theta*tau_i;
                    x = vk2-theta*(-val*uk+reg_i*vk2);
                    if(x>0){
                        // y = x-(rho_V[pid2*num_factors+d]+hV[pid2*num_factors+d]);
                        // V[pid2*num_factors+d] = (y>=0? y : 0);
                        V[pid2*num_factors+d] = max(0, x-(rho_V[pid2*num_factors+d]+hV[pid2*num_factors+d]));
                    }
                    else if(x<0){
                        // y = x+(rho_V[pid2*num_factors+d]-hV[pid2*num_factors+d]);
                        // V[pid2*num_factors+d] = (y<=0? y : 0);
                        V[pid2*num_factors+d] = min(0, x+(rho_V[pid2*num_factors+d]-hV[pid2*num_factors+d]));
                    }
                    hV[pid2*num_factors+d] += V[pid2*num_factors+d]-x;
                }
            }
        }
    }
}

void socf_bpr_second_order(long int *pairwise_pt, long int *index_pt, long int **neg_items, int *neg_size, double *U, double *V, double *rho_U, double *rho_V, double *hU, double *hV, double *SigMu, double *SigMv, double lmbda, double tau_u, double tau_i, double eta, long int num_pairs, int num_factors){

    long int t, k, uid, pid1, pid2; int d;
    double val=0, x, y, uk, vk1, vk2, mu, mv1, mv2, val1, val2, val3;

    for(t=0; t<num_pairs; t++){
        k = index_pt[t];
        uid = pairwise_pt[2*k];
        pid1 = pairwise_pt[2*k+1];
        k = (rand() % neg_size[uid]);
        pid2 = neg_items[uid][k];
        val = 0;
        for(d=0; d<num_factors; d++)
            val += U[uid*num_factors+d]*(V[pid1*num_factors+d]-V[pid2*num_factors+d]);
        // val = (val<1.0) ? -1.0 : 0.0;
        val = -1.0/(1.0+exp(val));

        // update the mean value
        val1=0; val2=0;
        for(d=0; d<num_factors; d++){
            uk = U[uid*num_factors+d];
            vk1 = V[pid1*num_factors+d];
            vk2 = V[pid2*num_factors+d];
            mu = SigMu[uid*num_factors+d];
            mv1 = SigMv[pid1*num_factors+d];
            mv2 = SigMv[pid2*num_factors+d];
            U[uid*num_factors+d] = uk - eta*val*(vk1-vk2)*mu; 
            V[pid1*num_factors+d] = vk1-eta*val*uk*mv1;
            V[pid2*num_factors+d] = vk2-eta*(-val*uk)*mv2;           
            val1 += mu*pow((V[pid1*num_factors+d]-V[pid2*num_factors+d]), 2);
            val2 += mv1*pow(U[uid*num_factors+d], 2);
            val3 += mv2*pow(U[uid*num_factors+d], 2);
        }
        // upadte the covariance matrix
        for(d=0; d<num_factors; d++){
            SigMu[uid*num_factors+d] -= pow(SigMu[uid*num_factors+d]*(V[pid1*num_factors+d]-V[pid2*num_factors+d]), 2)/(1/lmbda+val1);
            SigMv[pid1*num_factors+d] -= pow(SigMu[uid*num_factors+d]*V[pid1*num_factors+d], 2)/(1/lmbda+val2);
            SigMv[pid2*num_factors+d] -= pow(SigMu[uid*num_factors+d]*V[pid2*num_factors+d], 2)/(1/lmbda+val3);
        }
    }
}

void adawrmf_sample_negative(long int *output_pairs, double *R, double *SD, double *D, long int *input_pairs, long int **neg_items, int *neg_size, int *pos_size,long int num_input_pairs, long int num_users){

    long int uid, pid, k, init_pos;
    int i, j, num_sample;
    int t, *label; 

    for(k=0; k<num_input_pairs; k++){
        output_pairs[2*k] = input_pairs[2*k];
        output_pairs[2*k+1] = input_pairs[2*k+1];
        R[k] = 1;
        SD[k] = 1+D[k];
    }

    for(uid=0; uid<num_users; uid++){
        num_sample = pos_size[uid];
        label = (int *) malloc(sizeof(int)*neg_size[uid]);
        for(i=0; i<neg_size[uid]; i++)
            label[i] = 0;
        t = 0;
        while(t<pos_size[uid]){
            i = (rand() % neg_size[uid]);
        }
    }

}


void adawrmf_fast_train(long int *index_pt, long int *pair_pt, double *R_pt, double *W_pt, double *U_pt, double *V_pt, long int num_pairs, int num_factors,double theta, double lmbda){

    long int k1, k2, uid, pid;
    double val, uk, vk;
    int d;

    for(k1=0; k1<num_pairs; k1++){
        k2 = index_pt[k1];
        uid = pair_pt[2*k2];
        pid = pair_pt[2*k2+1];
        val = 0;
        for(d=0; d<num_factors; d++)
            val += U_pt[uid*num_factors+d]*V_pt[pid*num_factors+d];
        val = R_pt[k2]-val;
        for(d=0; d<num_factors; d++){
            uk = U_pt[uid*num_factors+d];
            vk = V_pt[pid*num_factors+d];
            U_pt[uid*num_factors+d] += theta*(W_pt[k2]*val*vk - lmbda*uk);
            V_pt[pid*num_factors+d] += theta*(W_pt[k2]*val*uk - lmbda*vk); 
        }
    }
}

double adawrmf_train_loss(long int *pair_pt, double *R_pt, double *W_pt, double *U_pt, double *V_pt, long int num_pairs, long int num_users, long int num_items, int num_factors, double lmbda){

    long int k, uid, pid;
    int d;
    double val, loss=0;

    for(k=0; k<num_pairs; k++){
        uid = pair_pt[2*k]; pid = pair_pt[2*k+1];
        val = 0;
        for(d=0; d<num_factors; d++)
            val += U_pt[uid*num_factors+d]*V_pt[pid*num_factors+d];
        val = val - R_pt[k];
        loss += 0.5*W_pt[k]*pow(val, 2);
    }

    for(uid=0; uid<num_users; uid++){
        val = 0;
        for(d=0; d<num_factors; d++)
            val += U_pt[uid*num_factors+d]*U_pt[uid*num_factors+d];
        loss += 0.5*lmbda*val;
    }

    for(pid=0; pid<num_items; pid++){
        val = 0;
        for(d=0; d<num_factors; d++)
            val += V_pt[pid*num_factors+d]*V_pt[pid*num_factors+d];
        loss += 0.5*lmbda*val;
    }

    return loss;
}