'''import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class DCCF_crv(BaseModel):
    def __init__(self, data_handler):
        super(DCCF_crv, self).__init__(data_handler)
        # Original collaborative graph parameters
        self.adj = data_handler.torch_adj
        self.semantic_adj = data_handler.semantic_adj
        self.keep_rate = configs['model']['keep_rate']
        self.edge_dropper = SpAdjEdgeDrop()

        # User and item parameters
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']

        # Hyperparameters
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.contrast_weight = self.hyper_config['contrastive_weight']
        self.contrast_temp = self.hyper_config['contrast_temp']

        # Collaborative channel embedding
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        # Semantic channel embedding and MLP
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.semantic_mlp = nn.Sequential(
            nn.Linear(1536, 768),
            nn.LeakyReLU(),
            nn.Linear(768, self.embedding_size)
        )

        # GAT layers for semantic channel
        self.gat_layers = nn.ModuleList([
            GATConv(
                self.embedding_size,
                self.embedding_size,
                heads=self.hyper_config['gat_heads'],
                concat=False,
                add_self_loops=False,
                edge_dim=1  # Adding edge weights support
            )
            for _ in range(self.hyper_config['layer_num'])
        ])

        # Gating mechanism for fusion of channels
        self.gate = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        # Custom weight initialization for GAT layers
        for gat in self.gat_layers:
            init(gat.att_src)
            init(gat.att_dst)
            if hasattr(gat, 'lin'):
                init(gat.lin.weight)
            elif hasattr(gat, 'lin_src') and gat.lin_src is not None:
                init(gat.lin_src.weight)
                init(gat.lin_dst.weight)

    def _propagate_co(self, adj, embeds):
        """Collaborative channel propagation"""
        embeds_list = [embeds]
        for _ in range(configs['model']['layer_num']):
            embeds = t.spmm(adj, embeds_list[-1])
            embeds_list.append(embeds)
        return sum(embeds_list)  # Sum over all layers

    def _propagate_sem(self, embeds):
        """Semantic channel propagation using GAT"""
        edge_index = self.semantic_adj.indices()
        edge_weight = self.semantic_adj.values()
        for gat in self.gat_layers:
            embeds = gat(embeds, edge_index, edge_attr=edge_weight)
            embeds = F.leaky_relu(embeds)
        return embeds

    def forward(self, keep_rate=1.0):
        # Collaborative channel embedding propagation
        co_embeds = t.cat([self.user_embeds, self.item_embeds], 0)
        dropped_adj = self.edge_dropper(self.adj, keep_rate)  # Drop edges for regularization
        co_embeds = self._propagate_co(dropped_adj, co_embeds)

        # Semantic channel embedding propagation (semantic graph)
        sem_embeds = t.cat([
            self.semantic_mlp(self.usrprf_embeds),
            self.semantic_mlp(self.itmprf_embeds)
        ], 0)
        sem_embeds = self._propagate_sem(sem_embeds)

        # Fusion via gating mechanism
        gate_input = t.cat([co_embeds, sem_embeds], dim=1)
        gate = self.sigmoid(self.gate(gate_input))
        fused_embeds = gate * co_embeds + (1 - gate) * sem_embeds

        return fused_embeds[:self.user_num], fused_embeds[self.user_num:]

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds_struct, item_embeds_struct = self.forward(self.keep_rate)
        user_embeds_sem, item_embeds_sem = self.forward(1.0)

        # Contrastive loss computation
        anc_embeds_struct, pos_embeds_struct, _ = self._pick_embeds(user_embeds_struct, item_embeds_struct, batch_data)
        anc_embeds_sem, pos_embeds_sem, _ = self._pick_embeds(user_embeds_sem, item_embeds_sem, batch_data)

        user_contrast_loss = cal_infonce_loss(anc_embeds_struct, anc_embeds_sem, anc_embeds_sem, self.contrast_temp)
        item_contrast_loss = cal_infonce_loss(pos_embeds_struct, pos_embeds_sem, pos_embeds_sem, self.contrast_temp)
        contrast_loss = (user_contrast_loss + item_contrast_loss) * self.contrast_weight

        # BPR loss
        user_embeds, item_embeds = self.forward(self.keep_rate)
        anc, pos, neg = batch_data
        anc_embeds = user_embeds[anc]
        pos_embeds = item_embeds[pos]
        neg_embeds = item_embeds[neg]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)

        # Regularization
        reg_loss = self.reg_weight * (
                t.norm(self.user_embeds) +
                t.norm(self.item_embeds) +
                sum(t.norm(p) for p in self.semantic_mlp.parameters()) +
                sum(t.norm(p) for p in self.gat_layers.parameters())
        )

        # Knowledge Distillation loss
        usrprf_embeds = self.semantic_mlp(self.usrprf_embeds)
        itmprf_embeds = self.semantic_mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        # Total loss
        loss = bpr_loss + reg_loss + kd_loss + contrast_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss
        }
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds, item_embeds = self.forward(1.0)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        # Predict full scores
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T

        # Mask out training interactions
        if train_mask is not None:
            full_preds = self._mask_predict(full_preds, train_mask)

        return full_preds'''


import torch as t
import numpy as np
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class DCCF_crv(BaseModel):
    def __init__(self, data_handler):
        super(DCCF_crv, self).__init__(data_handler)

        # 协同通道参数（DCCF原始参数）
        # prepare adjacency matrix for DCCF
        rows = data_handler.trn_mat.tocoo().row
        cols = data_handler.trn_mat.tocoo().col
        new_rows = np.concatenate([rows, cols + self.user_num], axis=0)
        new_cols = np.concatenate([cols + self.user_num, rows], axis=0)
        plain_adj = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)),
                                  shape=[self.user_num + self.item_num, self.user_num + self.item_num]).tocsr().tocoo()
        self.all_h_list = list(plain_adj.row)
        self.all_t_list = list(plain_adj.col)
        self.A_in_shape = plain_adj.shape
        self.A_indices = t.tensor([self.all_h_list, self.all_t_list], dtype=t.long).cuda()
        self.D_indices = t.tensor([list(range(self.user_num + self.item_num)),
                                   list(range(self.user_num + self.item_num))], dtype=t.long).cuda()
        self.all_h_list = t.LongTensor(self.all_h_list).cuda()
        self.all_t_list = t.LongTensor(self.all_t_list).cuda()
        self.G_indices, self.G_values = self._cal_sparse_adj()

        # 原始协同图参数和边丢弃
        self.adj = data_handler.torch_adj
        self.semantic_adj = data_handler.semantic_adj
        self.keep_rate = configs['model']['keep_rate']
        self.edge_dropper = SpAdjEdgeDrop()

        # 超参数
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']
        self.intent_num = configs['model']['intent_num']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.contrast_weight = self.hyper_config['contrastive_weight']
        self.contrast_temp = self.hyper_config['contrast_temp']
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']

        # DCCF模型参数
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.user_intent = t.nn.Parameter(init(t.empty(self.embedding_size, self.intent_num)), requires_grad=True)
        self.item_intent = t.nn.Parameter(init(t.empty(self.embedding_size, self.intent_num)), requires_grad=True)

        # 语义通道参数
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.semantic_mlp = nn.Sequential(
            nn.Linear(1536, 768),
            nn.LeakyReLU(),
            nn.Linear(768, self.embedding_size)
        )

        # GAT参数
        self.gat_layers = nn.ModuleList([
            GATConv(
                self.embedding_size,
                self.embedding_size,
                heads=self.hyper_config['gat_heads'],
                concat=False,
                add_self_loops=False,
                edge_dim=1  # 支持边权重
            )
            for _ in range(self.layer_num)
        ])

        # 门控融合参数
        self.gate = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.sigmoid = nn.Sigmoid()

        # 训练/测试状态
        self.is_training = True
        self.final_embeds = None

        self._init_weights()

    def _init_weights(self):
        # 初始化GAT参数
        for gat in self.gat_layers:
            # 初始化注意力参数
            init(gat.att_src)
            init(gat.att_dst)
            # 初始化线性变换参数
            if hasattr(gat, 'lin'):
                init(gat.lin.weight)
            elif hasattr(gat, 'lin_src') and gat.lin_src is not None:
                init(gat.lin_src.weight)
                init(gat.lin_dst.weight)

        # 初始化MLP参数
        for m in self.semantic_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _cal_sparse_adj(self):
        A_values = t.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list,
                                             value=A_values, sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices,
                                                  A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices,
                                                  D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        return G_indices, G_values

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        head_embeddings = t.nn.functional.normalize(head_embeddings)
        tail_embeddings = t.nn.functional.normalize(tail_embeddings)
        edge_alpha = (t.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list,
                                             value=edge_alpha, sparse_sizes=self.A_in_shape).cuda()
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_indices = t.stack([self.all_h_list, self.all_t_list], dim=0)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha
        return G_indices, G_values

    def _propagate_sem(self, embeds):
        """语义通道信息传播（基于GAT）"""
        edge_index = self.semantic_adj.indices()
        edge_weight = self.semantic_adj.values()
        for gat in self.gat_layers:
            embeds = gat(embeds, edge_index, edge_attr=edge_weight)
            embeds = F.leaky_relu(embeds)
        return embeds

    def forward(self, keep_rate=1.0):
        """双通道前向传播"""
        if not self.is_training and self.final_embeds is not None:
            user_embeds, item_embeds = t.split(self.final_embeds, [self.user_num, self.item_num], 0)
            return user_embeds, item_embeds

        # 协同通道 (DCCF)
        all_embeds = [t.concat([self.user_embeds, self.item_embeds], dim=0)]
        gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = [], [], [], []

        # 应用 edge dropping （如果在训练阶段）
        adj_indices = self.G_indices
        adj_values = self.G_values
        if self.is_training and keep_rate < 1.0:
            # 这里可以实现边丢弃，但DCCF不同于LightGCN，需要特殊处理
            # 为简化，我们这里不对DCCF的图结构应用edge dropping
            pass

        # DCCF的信息传播
        for i in range(self.layer_num):
            # Graph-based Message Passing
            gnn_layer_embeds = torch_sparse.spmm(adj_indices, adj_values,
                                                 self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

            # Intent-aware Information Aggregation
            u_embeds, i_embeds = t.split(all_embeds[i], [self.user_num, self.item_num], 0)
            u_int_embeds = t.softmax(u_embeds @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeds = t.softmax(i_embeds @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeds = t.concat([u_int_embeds, i_int_embeds], dim=0)

            # Adaptive Augmentation
            gnn_head_embeds = t.index_select(gnn_layer_embeds, 0, self.all_h_list)
            gnn_tail_embeds = t.index_select(gnn_layer_embeds, 0, self.all_t_list)
            int_head_embeds = t.index_select(int_layer_embeds, 0, self.all_h_list)
            int_tail_embeds = t.index_select(int_layer_embeds, 0, self.all_t_list)
            G_graph_indices, G_graph_values = self._adaptive_mask(gnn_head_embeds, gnn_tail_embeds)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeds, int_tail_embeds)
            gaa_layer_embeds = torch_sparse.spmm(G_graph_indices, G_graph_values,
                                                 self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])
            iaa_layer_embeds = torch_sparse.spmm(G_inten_indices, G_inten_values,
                                                 self.A_in_shape[0], self.A_in_shape[1], all_embeds[i])

            # 保存各种嵌入用于对比学习
            gnn_embeds.append(gnn_layer_embeds)
            int_embeds.append(int_layer_embeds)
            gaa_embeds.append(gaa_layer_embeds)
            iaa_embeds.append(iaa_layer_embeds)

            # 聚合当前层嵌入
            all_embeds.append(gnn_layer_embeds + int_layer_embeds + gaa_layer_embeds + iaa_layer_embeds + all_embeds[i])

        # 聚合所有层次的嵌入
        all_embeds = t.stack(all_embeds, dim=1)
        co_embeds = t.sum(all_embeds, dim=1, keepdim=False)

        # 语义通道
        sem_embeds = t.cat([
            self.semantic_mlp(self.usrprf_embeds),
            self.semantic_mlp(self.itmprf_embeds)
        ], 0)
        sem_embeds = self._propagate_sem(sem_embeds)

        # 门控融合
        gate_input = t.cat([co_embeds, sem_embeds], dim=1)
        gate = self.sigmoid(self.gate(gate_input))
        fused_embeds = gate * co_embeds + (1 - gate) * sem_embeds

        # 保存结果用于推理
        self.final_embeds = fused_embeds

        # 如果是训练阶段，返回额外的嵌入用于对比学习
        if self.is_training:
            return t.split(fused_embeds, [self.user_num, self.item_num],
                           0), gnn_embeds, int_embeds, gaa_embeds, iaa_embeds
        else:
            return t.split(fused_embeds, [self.user_num, self.item_num], 0)

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def _cal_cl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
        """DCCF原始对比学习损失"""
        users = t.unique(users)
        items = t.unique(items)  # different from original SSLRec, remove negative items
        cl_loss = 0.0
        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = t.split(gnn_emb[i], [self.user_num, self.item_num], 0)
            u_int_embs, i_int_embs = t.split(int_emb[i], [self.user_num, self.item_num], 0)
            u_gaa_embs, i_gaa_embs = t.split(gaa_emb[i], [self.user_num, self.item_num], 0)
            u_iaa_embs, i_iaa_embs = t.split(iaa_emb[i], [self.user_num, self.item_num], 0)

            u_gnn_embs = u_gnn_embs[users]
            u_int_embs = u_int_embs[users]
            u_gaa_embs = u_gaa_embs[users]
            u_iaa_embs = u_iaa_embs[users]

            i_gnn_embs = i_gnn_embs[items]
            i_int_embs = i_int_embs[items]
            i_gaa_embs = i_gaa_embs[items]
            i_iaa_embs = i_iaa_embs[items]

            cl_loss += cal_infonce_loss(u_gnn_embs, u_int_embs, u_int_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(u_gnn_embs, u_gaa_embs, u_gaa_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(u_gnn_embs, u_iaa_embs, u_iaa_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_int_embs, i_int_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_gaa_embs, i_gaa_embs, self.cl_temperature) / u_gnn_embs.shape[0]
            cl_loss += cal_infonce_loss(i_gnn_embs, i_iaa_embs, i_iaa_embs, self.cl_temperature) / u_gnn_embs.shape[0]
        return cl_loss

    def cal_loss(self, batch_data):
        self.is_training = True

        # 双通道前向传播（带edge dropping）
        result = self.forward(self.keep_rate)
        (user_embeds, item_embeds), gnn_embeds, int_embeds, gaa_embeds, iaa_embeds = result

        # 不带edge dropping的前向传播（用于对比学习）
        result_full = self.forward(1.0)
        (user_embeds_sem, item_embeds_sem), _, _, _, _ = result_full

        # 提取当前batch的嵌入
        ancs, poss, negs = batch_data

        # 从结构图前向传播结果中提取嵌入
        anc_embeds_struct = user_embeds[ancs]
        pos_embeds_struct = item_embeds[poss]
        neg_embeds_struct = item_embeds[negs]

        # 从语义图前向传播结果中提取嵌入
        anc_embeds_sem = user_embeds_sem[ancs]
        pos_embeds_sem = item_embeds_sem[poss]
        neg_embeds_sem = item_embeds_sem[negs]

        # 计算结构-语义对比损失
        user_contrast_loss = cal_infonce_loss(anc_embeds_struct, anc_embeds_sem, anc_embeds_sem, self.contrast_temp)
        item_contrast_loss = cal_infonce_loss(pos_embeds_struct, pos_embeds_sem, pos_embeds_sem, self.contrast_temp)
        contrast_loss = (user_contrast_loss + item_contrast_loss) * self.contrast_weight

        # BPR损失
        bpr_loss = cal_bpr_loss(anc_embeds_struct, pos_embeds_struct, neg_embeds_struct) / anc_embeds_struct.shape[0]

        # DCCF原始对比损失
        cl_loss = self.cl_weight * self._cal_cl_loss(ancs, poss, gnn_embeds, int_embeds, gaa_embeds, iaa_embeds)

        # 正则化损失
        reg_loss = self.reg_weight * (
                t.norm(self.user_embeds) +
                t.norm(self.item_embeds) +
                t.norm(self.user_intent) +
                t.norm(self.item_intent) +
                sum(t.norm(p) for p in self.semantic_mlp.parameters()) +
                sum(t.norm(p) for p in self.gat_layers.parameters())
        )

        # 知识蒸馏损失
        usrprf_embeds = self.semantic_mlp(self.usrprf_embeds)
        itmprf_embeds = self.semantic_mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc_embeds_struct, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds_struct, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds_struct, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds_struct.shape[0]
        kd_loss *= self.kd_weight

        # 总损失
        loss = bpr_loss + reg_loss + kd_loss + contrast_loss + cl_loss

        # 记录各项损失
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss,
            'cl_loss': cl_loss
        }
        return loss, losses

    def full_predict(self, batch_data):
        # 切换到预测模式
        self.is_training = False

        # 前向传播
        user_embeds, item_embeds = self.forward(1.0)

        # 解包批次数据
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        # 获取目标用户嵌入
        pck_user_embeds = user_embeds[pck_users]

        # 计算全量预测分数
        full_preds = pck_user_embeds @ item_embeds.T

        # 屏蔽训练交互
        if train_mask is not None:
            full_preds = self._mask_predict(full_preds, train_mask)

        return full_preds