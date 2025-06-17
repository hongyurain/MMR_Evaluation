import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, build_knn_normalized_graph


class MICRO(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MICRO, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        #self.feat_embed_dim = config['feat_embed_dim']
        self.weight_size = config['weight_size']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.sparse = 1 
        self.norm_type = config['norm_type']
        self.loss_ratio = config['loss_ratio']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_adj_mat()
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if config['cf_model'] == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            dropout_list = config['mess_dropout']
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}.pt'.format(self.knn_k))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k , is_sparse=self.sparse, norm_type=self.norm_type)
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k , is_sparse=self.sparse, norm_type=self.norm_type)
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        #self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

        self.query = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.tau = 0.5

    def pre_epoch_processing(self):
        self.build_item_graph = True

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            #print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def mm(self, x, y):
        if self.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
                   
        loss_vec = torch.cat(losses)
        return loss_vec.mean()

#未完成modality missing部分
    def forward(self, adj, build_item_graph=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        #print('build_item_graph',build_item_graph)
        if build_item_graph:
            #weight = self.softmax(self.modal_weight)

            if self.v_feat is not None:
                self.image_adj = build_sim(image_feats)
                self.image_adj = build_knn_normalized_graph(self.image_adj, topk=self.knn_k , is_sparse=self.sparse, norm_type=self.norm_type)
                self.image_adj = (1 - self.lambda_coeff) * self.image_adj + self.lambda_coeff * self.image_original_adj
                
            if self.t_feat is not None:
                self.text_adj = build_sim(text_feats)
                self.text_adj = build_knn_normalized_graph(self.text_adj, topk=self.knn_k , is_sparse=self.sparse, norm_type=self.norm_type)
                self.text_adj = (1 - self.lambda_coeff) * self.text_adj + self.lambda_coeff * self.text_original_adj
                

            # if self.v_feat is not None:
            #     self.image_adj = build_sim(image_feats)
            #     #self.image_adj = build_knn_neighbourhood(self.image_adj, topk=self.knn_k)
            #     self.image_adj = build_knn_normalized_graph(self.image_adj, topk=self.knn_k , is_sparse=self.sparse, norm_type=self.norm_type)
            #     self.image_adj = (1 - self.lambda_coeff) * self.image_adj + self.lambda_coeff * self.image_original_adj
            #     #self.image_adj = self.lambda_coeff * self.image_original_adj + (1 - self.lambda_coeff) * self.image_adj
            #     image_item_embeds = self.item_id_embedding.weight

            #     for i in range(self.n_layers):
            #         image_item_embeds = self.mm(self.image_adj, image_item_embeds)

            #     # learned_adj = self.image_adj
            #     # original_adj = self.image_original_adj
            # if self.t_feat is not None:
            #     self.text_adj = build_sim(text_feats)
            #     #self.text_adj = build_knn_neighbourhood(self.text_adj, topk=self.knn_k)
            #     self.text_adj = build_knn_normalized_graph(self.text_adj, topk=self.knn_k , is_sparse=self.sparse, norm_type=self.norm_type)
            #     self.text_adj = (1 - self.lambda_coeff) * self.text_adj + self.lambda_coeff * self.text_original_adj
            #     #self.text_adj = self.lambda_coeff * self.text_original_adj + (1 - self.lambda_coeff) * self.text_adj
            #     text_item_embeds = self.item_id_embedding.weight

            #     for i in range(self.n_layers):
            #         text_item_embeds = self.mm(self.text_adj, text_item_embeds) 

                # learned_adj = self.text_adj
                # original_adj = self.text_original_adj
            # if self.v_feat is not None and self.t_feat is not None:
            #     att = torch.cat([self.query(image_item_embeds), self.query(text_item_embeds)], dim=-1)
            #     weight = self.softmax(att)
            #     h = weight[:, 0].unsqueeze(dim=1) * image_item_embeds + weight[:, 1].unsqueeze(dim=1) * text_item_embeds

            # learned_adj = compute_normalized_laplacian(learned_adj)
            # #print(self.lambda_coeff)
            # #print(learned_adj.shape)
            # #print(original_adj.shape)
            # self.item_adj = (1 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj
        else:
            if self.v_feat is not None:
                self.image_adj = self.image_adj.detach()
            if self.t_feat is not None:
                self.text_adj = self.text_adj.detach()

        image_item_embeds = self.item_id_embedding.weight
        text_item_embeds = self.item_id_embedding.weight


        if self.v_feat is not None:
            for i in range(self.n_layers):
                image_item_embeds = self.mm(self.image_adj, image_item_embeds)

        if self.t_feat is not None:
            for i in range(self.n_layers):
                text_item_embeds = self.mm(self.text_adj, text_item_embeds)

        if self.v_feat is not None:
            att = self.query(image_item_embeds)
            weight = self.softmax(att)
            h = weight * image_item_embeds
        if self.t_feat is not None:
            att = self.query(text_item_embeds)
            weight = self.softmax(att)
            h = weight * text_item_embeds
        if self.v_feat is not None and self.t_feat is not None:
            att = torch.cat([self.query(image_item_embeds), self.query(text_item_embeds)], dim=-1)
            weight = self.softmax(att)
            h = weight[:, 0].unsqueeze(dim=1) * image_item_embeds + weight[:, 1].unsqueeze(dim=1) * text_item_embeds

        
        if self.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings, image_item_embeds, text_item_embeds, h
        elif self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings, image_item_embeds, text_item_embeds, h
        elif self.cf_model == 'mf':
            return self.user_embedding.weight, self.item_id_embedding.weight + F.normalize(h, p=2, dim=1), image_item_embeds, text_item_embeds, h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, image_item_embeds, text_item_embeds, fusion_embed = self.forward(self.norm_adj, build_item_graph=self.build_item_graph)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)
        
        batch_contrastive_loss = 0
        if self.v_feat is not None:
            batch_contrastive_loss += self.batched_contrastive_loss(image_item_embeds,fusion_embed)
        if self.t_feat is not None:
            batch_contrastive_loss += self.batched_contrastive_loss(text_item_embeds,fusion_embed)

        batch_contrastive_loss *= self.loss_ratio



        return batch_mf_loss + batch_emb_loss + batch_reg_loss + batch_contrastive_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e, *rest = self.forward(self.norm_adj, build_item_graph=True)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

