# SimGCL + LLMRec
import torch as t
from torch import nn
import torch.nn.functional as F
from encoder.models.general_cf.lightgcn import LightGCN
from encoder.models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from encoder.config.configurator import configs

init = nn.init.xavier_uniform_


class SimGCL_LLMRec(LightGCN):
    def __init__(self, data_handler):
        super(SimGCL_LLMRec, self).__init__(data_handler)

        # hyper-parameter
        self.cl_weight = self.hyper_config['cl_weight']
        self.temperature = self.hyper_config['temperature']
        self.eps = self.hyper_config['eps']
        self.llm_weight = self.hyper_config['llm_weight']
        self.fusion_temp = self.hyper_config['fusion_temp']

        # LLM semantic embeddings
        self.llm_user_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.llm_item_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()

        # Projection layers
        self.user_proj = nn.Linear(self.llm_user_embeds.shape[1], self.embedding_size)
        self.item_proj = nn.Linear(self.llm_item_embeds.shape[1], self.embedding_size)

        # Semantic adjacency matrix
        self.sem_adj = None

        self._init_weights()

    def _init_weights(self):
        init(self.user_proj.weight)
        init(self.item_proj.weight)

    def _perturb_embedding(self, embeds):
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        return embeds + noise

    def _compute_semantic_adj(self):
        # Project LLM embeddings to the model's embedding space
        proj_user_embeds = self.user_proj(self.llm_user_embeds)
        proj_item_embeds = self.item_proj(self.llm_item_embeds)

        # Compute similarity-based adjacency matrix
        sim_matrix = t.sigmoid(proj_user_embeds @ proj_item_embeds.T / self.fusion_temp)

        return sim_matrix

    def forward(self, adj=None, perturb=False, use_semantic=True):
        if adj is None:
            adj = self.adj

        # Compute semantic adjacency if enabled
        if use_semantic and self.sem_adj is None:
            self.sem_adj = self._compute_semantic_adj()

        if not perturb:
            # Standard forward pass without perturbation
            embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
            embeds_list = [embeds]
            for i in range(self.layer_num):
                embeds = self._propagate(adj, embeds_list[-1])
                embeds_list.append(embeds)
            embeds = sum(embeds_list)
        else:
            # SimGCL perturbation
            embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
            embeds_list = [embeds]
            for i in range(self.layer_num):
                embeds = self._propagate(adj, embeds_list[-1])
                embeds = self._perturb_embedding(embeds)
                embeds_list.append(embeds)
            embeds = sum(embeds_list)

        # If using semantic information, enhance with semantic propagation
        if use_semantic:
            # Create projected embeddings for semantic graph
            sem_user_embeds = self.user_proj(self.llm_user_embeds)
            sem_item_embeds = self.item_proj(self.llm_item_embeds)

            # Propagate through semantic graph
            sem_user_enhance = self.sem_adj @ sem_item_embeds
            sem_item_enhance = self.sem_adj.T @ sem_user_embeds

            # Fuse embeddings
            user_embeds = embeds[:self.user_num] + self.llm_weight * sem_user_enhance
            item_embeds = embeds[self.user_num:] + self.llm_weight * sem_item_enhance

            return user_embeds, item_embeds

        return embeds[:self.user_num], embeds[self.user_num:]

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True

        # Two perturbed views for contrastive learning
        user_embeds1, item_embeds1 = self.forward(self.adj, perturb=True)
        user_embeds2, item_embeds2 = self.forward(self.adj, perturb=True)

        # Standard view without perturbation
        user_embeds3, item_embeds3 = self.forward(self.adj, perturb=False)

        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

        # Standard BPR loss
        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]

        # Contrastive loss from SimGCL
        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.temperature) + \
                  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.temperature)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight

        # Semantic edge prediction loss
        ancs, poss, negs = batch_data
        if self.sem_adj is not None:
            # Get semantic prediction scores
            sem_scores = self.sem_adj[ancs, poss]
            # Binary cross-entropy for semantic edge prediction
            target = t.ones_like(sem_scores)
            semantic_loss = nn.functional.binary_cross_entropy(sem_scores, target)
            semantic_loss *= self.llm_weight
        else:
            semantic_loss = t.tensor(0.0).cuda()

        reg_loss = self.reg_weight * reg_params(self)

        loss = bpr_loss + reg_loss + cl_loss + semantic_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'semantic_loss': semantic_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, perturb=False)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
