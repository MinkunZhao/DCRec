# LightGCN + LLMRec
import torch as t
from torch import nn
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class LightGCN_LLMRec(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_LLMRec, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.llm_weight = self.hyper_config['llm_weight']
        self.fusion_temp = self.hyper_config['fusion_temp']

        # LLM semantic embeddings
        self.llm_user_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.llm_item_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()

        # Projection layers
        self.user_proj = nn.Linear(self.llm_user_embeds.shape[1], self.embedding_size)
        self.item_proj = nn.Linear(self.llm_item_embeds.shape[1], self.embedding_size)

        # Semantic adjacency matrix - would be computed during forward pass
        self.sem_adj = None

        self._init_weights()

    def _init_weights(self):
        init(self.user_proj.weight)
        init(self.item_proj.weight)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def _compute_semantic_adj(self):
        # Project LLM embeddings to the model's embedding space
        proj_user_embeds = self.user_proj(self.llm_user_embeds)
        proj_item_embeds = self.item_proj(self.llm_item_embeds)

        # Compute similarity-based adjacency matrix
        sim_matrix = t.sigmoid(proj_user_embeds @ proj_item_embeds.T / self.fusion_temp)

        # Create sparse adjacency matrix structure similar to interaction graph
        return sim_matrix

    def forward(self, adj=None, keep_rate=1.0, use_semantic=True):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]

        # Compute semantic adjacency if enabled
        if use_semantic and self.sem_adj is None:
            self.sem_adj = self._compute_semantic_adj()

        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]

        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)

        # Standard GNN propagation
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

        embeds = sum(embeds_list)

        # If using semantic information, enhance with semantic propagation
        if use_semantic:
            # Create projected embeddings for semantic graph
            sem_user_embeds = self.user_proj(self.llm_user_embeds)
            sem_item_embeds = self.item_proj(self.llm_item_embeds)

            # Propagate through semantic graph
            # (simplified - in a full implementation you'd do multi-hop propagation)
            sem_user_enhance = self.sem_adj @ sem_item_embeds
            sem_item_enhance = self.sem_adj.T @ sem_user_embeds

            # Fuse embeddings - weighted combination
            user_embeds = embeds[:self.user_num] + self.llm_weight * sem_user_enhance
            item_embeds = embeds[self.user_num:] + self.llm_weight * sem_item_enhance

            embeds = t.concat([user_embeds, item_embeds], axis=0)

        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:]

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)

        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        # Standard BPR loss for CF signals
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]

        # Semantic edge prediction loss
        if self.sem_adj is not None:
            # Get semantic prediction scores
            sem_scores = self.sem_adj[ancs, poss]
            # Binary cross-entropy for semantic edge prediction
            target = t.ones_like(sem_scores)
            semantic_loss = nn.functional.binary_cross_entropy(sem_scores, target)
            semantic_loss *= self.llm_weight
        else:
            semantic_loss = t.tensor(0.0).cuda()

        # Regularization loss
        reg_loss = self.reg_weight * reg_params(self)

        loss = bpr_loss + reg_loss + semantic_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'semantic_loss': semantic_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
