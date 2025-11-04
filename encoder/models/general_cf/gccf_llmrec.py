# GCCF + LLMRec
import torch as t
from torch import nn
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, reg_params
from encoder.models.base_model import BaseModel

init = nn.init.xavier_uniform_


class GCNLayer(nn.Module):
    def __init__(self, latdim):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(init(t.empty(latdim, latdim)))

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)  # @ self.W (Performs better without W)


class GCCF_LLMRec(BaseModel):
    def __init__(self, data_handler):
        super(GCCF_LLMRec, self).__init__(data_handler)

        self.adj = data_handler.torch_adj

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.llm_weight = self.hyper_config['llm_weight']
        self.fusion_temp = self.hyper_config['fusion_temp']

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.gcnLayers = nn.Sequential(*[GCNLayer(self.embedding_size) for i in range(self.layer_num)])

        # LLM semantic embeddings
        self.llm_user_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.llm_item_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()

        # Projection layers
        output_size = int((self.layer_num + 1) * self.embedding_size)
        self.user_proj = nn.Linear(self.llm_user_embeds.shape[1], output_size)
        self.item_proj = nn.Linear(self.llm_item_embeds.shape[1], output_size)

        # Semantic adjacency matrix
        self.sem_adj = None

        self.is_training = True

        self._init_weights()

    def _init_weights(self):
        init(self.user_proj.weight)
        init(self.item_proj.weight)

    def _compute_semantic_adj(self):
        # Project LLM embeddings
        proj_user_embeds = self.user_proj(self.llm_user_embeds)[:, :self.embedding_size]  # Use first chunk for adj
        proj_item_embeds = self.item_proj(self.llm_item_embeds)[:, :self.embedding_size]  # Use first chunk for adj

        # Compute similarity-based adjacency matrix
        sim_matrix = t.sigmoid(proj_user_embeds @ proj_item_embeds.T / self.fusion_temp)

        return sim_matrix

    def forward(self, adj=None, use_semantic=True):
        if adj is None:
            adj = self.adj

        # Compute semantic adjacency if enabled
        if use_semantic and self.sem_adj is None:
            self.sem_adj = self._compute_semantic_adj()

        if not self.is_training:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None

        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = t.concat(embeds_list, dim=-1)

        # If using semantic information, enhance with semantic propagation
        if use_semantic:
            # Get projected semantic embeddings
            sem_user_embeds = self.user_proj(self.llm_user_embeds)
            sem_item_embeds = self.item_proj(self.llm_item_embeds)

            # For concatenated embeddings, we need to propagate each layer's worth
            enhance_user = t.zeros_like(sem_user_embeds)
            enhance_item = t.zeros_like(sem_item_embeds)

            # Simple propagation on semantic graph
            base_size = self.embedding_size
            for i in range(self.layer_num + 1):
                start_idx = i * base_size
                end_idx = (i + 1) * base_size
                # Propagate this layer's embeddings
                enhance_user[:, start_idx:end_idx] = self.sem_adj @ sem_item_embeds[:, start_idx:end_idx]
                enhance_item[:, start_idx:end_idx] = self.sem_adj.T @ sem_user_embeds[:, start_idx:end_idx]

            # Fuse embeddings
            user_embeds = embeds[:self.user_num] + self.llm_weight * enhance_user
            item_embeds = embeds[self.user_num:] + self.llm_weight * enhance_item

            embeds = t.concat([user_embeds, item_embeds], axis=0)

        self.final_embeds = embeds
        last_embeds = embeds_list[-1]  # Last layer embeddings for potential use in loss
        return embeds[:self.user_num], embeds[self.user_num:], last_embeds

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds, _ = self.forward(self.adj)
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]

        # Standard BPR loss
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)

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

        loss = bpr_loss + reg_loss + semantic_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'semantic_loss': semantic_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds, _ = self.forward(self.adj)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
