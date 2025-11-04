import torch as t
from torch import nn
import torch.nn.functional as F
from encoder.config.configurator import configs
from encoder.models.general_cf.lightgcn import LightGCN
from encoder.models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_


class MultiEmbedding(LightGCN):
    def __init__(self, data_handler):
        super(MultiEmbedding, self).__init__(data_handler)

        # hyper-parameter
        self.num_embeddings = self.hyper_config['num_embeddings']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.cl_weight = self.hyper_config['cl_weight']
        self.temperature = self.hyper_config['temperature']
        self.mix_ratio = self.hyper_config['mix_ratio']

        # replace single embedding with multiple embeddings
        self.user_embeds_list = nn.ParameterList([
            nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
            for _ in range(self.num_embeddings)
        ])
        self.item_embeds_list = nn.ParameterList([
            nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
            for _ in range(self.num_embeddings)
        ])

        # mapping networks for embedding spaces
        self.user_mapping = nn.Linear(self.embedding_size, self.embedding_size)
        self.item_mapping = nn.Linear(self.embedding_size, self.embedding_size)

        # for saving embeddings
        self.final_embeds_list = [None] * self.num_embeddings
        self.is_training = False

    def _aggregate_embeddings(self, embeddings_list):
        """Aggregate multiple embeddings using learned weights"""
        # Normalize embeddings
        normalized_embeddings = [F.normalize(emb, p=2, dim=1) for emb in embeddings_list]

        # Calculate similarity between embeddings
        weighted_sum = sum(normalized_embeddings) / len(normalized_embeddings)
        return weighted_sum

    def _propagate_single_view(self, adj, user_embeds, item_embeds):
        """Propagate a single embedding view through the graph"""
        embeds = t.concat([user_embeds, item_embeds], axis=0)
        embeds_list = [embeds]

        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

        embeds = sum(embeds_list)
        return embeds[:self.user_num], embeds[self.user_num:]

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj

        if not self.is_training and all(emb is not None for emb in self.final_embeds_list):
            # Aggregate final embeddings for inference
            user_embeds_all = []
            item_embeds_all = []
            for final_embeds in self.final_embeds_list:
                user_embeds_all.append(final_embeds[:self.user_num])
                item_embeds_all.append(final_embeds[self.user_num:])

            return self._aggregate_embeddings(user_embeds_all), self._aggregate_embeddings(item_embeds_all)

        # Process each embedding view
        user_embeds_all = []
        item_embeds_all = []

        for i in range(self.num_embeddings):
            if self.is_training:
                adj_view = self.edge_dropper(adj, keep_rate)
            else:
                adj_view = adj

            user_embeds, item_embeds = self._propagate_single_view(adj_view, self.user_embeds_list[i],
                                                                   self.item_embeds_list[i])

            # Apply mapping networks for alignment
            user_embeds = self.user_mapping(user_embeds)
            item_embeds = self.item_mapping(item_embeds)

            user_embeds_all.append(user_embeds)
            item_embeds_all.append(item_embeds)

            # Save the final embeddings for this view
            self.final_embeds_list[i] = t.concat([user_embeds, item_embeds], axis=0)

        # Mix embeddings based on mix ratio
        if self.mix_ratio > 0 and self.is_training:
            mixed_user_embeds = []
            mixed_item_embeds = []

            for i in range(self.num_embeddings):
                for j in range(i + 1, self.num_embeddings):
                    # Mix embeddings between views
                    alpha = t.rand(1).item() * self.mix_ratio
                    mixed_user = alpha * user_embeds_all[i] + (1 - alpha) * user_embeds_all[j]
                    mixed_item = alpha * item_embeds_all[i] + (1 - alpha) * item_embeds_all[j]

                    mixed_user_embeds.append(mixed_user)
                    mixed_item_embeds.append(mixed_item)

            # Add mixed embeddings to the list
            user_embeds_all.extend(mixed_user_embeds)
            item_embeds_all.extend(mixed_item_embeds)

        # Aggregate embeddings for final prediction
        return self._aggregate_embeddings(user_embeds_all), self._aggregate_embeddings(item_embeds_all)

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def _cal_alignment_loss(self, user_embeds_list, item_embeds_list, batch_data):
        """Calculate alignment loss between different embedding spaces"""
        ancs, poss, _ = batch_data
        alignment_loss = 0.0

        for i in range(len(user_embeds_list)):
            for j in range(i + 1, len(user_embeds_list)):
                user_i = user_embeds_list[i][ancs]
                user_j = user_embeds_list[j][ancs]
                item_i = item_embeds_list[i][poss]
                item_j = item_embeds_list[j][poss]

                # Alignment of user embeddings across views
                alignment_loss += cal_infonce_loss(user_i, user_j, user_embeds_list[j], self.temperature)
                # Alignment of item embeddings across views
                alignment_loss += cal_infonce_loss(item_i, item_j, item_embeds_list[j], self.temperature)

        return alignment_loss / (len(user_embeds_list) * (len(user_embeds_list) - 1) * ancs.shape[0])

    def _cal_uniformity_loss(self, user_embeds_list, item_embeds_list):
        """Calculate uniformity loss to prevent embedding collapse"""
        uniformity_loss = 0.0

        for user_embeds in user_embeds_list:
            # Normalize embeddings
            norm_embeds = F.normalize(user_embeds, p=2, dim=1)
            # Calculate cosine similarity
            sim_matrix = norm_embeds @ norm_embeds.T
            # Mask diagonal
            mask = t.eye(sim_matrix.size(0), device=sim_matrix.device)
            # Calculate uniformity loss
            uniformity_loss += t.sum(t.pow(sim_matrix * (1 - mask), 2)) / (
                        sim_matrix.size(0) * (sim_matrix.size(0) - 1))

        for item_embeds in item_embeds_list:
            # Normalize embeddings
            norm_embeds = F.normalize(item_embeds, p=2, dim=1)
            # Calculate cosine similarity
            sim_matrix = norm_embeds @ norm_embeds.T
            # Mask diagonal
            mask = t.eye(sim_matrix.size(0), device=sim_matrix.device)
            # Calculate uniformity loss
            uniformity_loss += t.sum(t.pow(sim_matrix * (1 - mask), 2)) / (
                        sim_matrix.size(0) * (sim_matrix.size(0) - 1))

        return uniformity_loss / (len(user_embeds_list) + len(item_embeds_list))

    def cal_loss(self, batch_data):
        self.is_training = True

        # Get embeddings from all views
        user_embeds_list = []
        item_embeds_list = []

        for i in range(self.num_embeddings):
            adj_view = self.edge_dropper(self.adj, self.keep_rate)
            user_embeds, item_embeds = self._propagate_single_view(adj_view, self.user_embeds_list[i],
                                                                   self.item_embeds_list[i])

            # Apply mapping networks
            user_embeds = self.user_mapping(user_embeds)
            item_embeds = self.item_mapping(item_embeds)

            user_embeds_list.append(user_embeds)
            item_embeds_list.append(item_embeds)

        # Get aggregated embeddings for BPR loss
        user_embeds_agg, item_embeds_agg = self._aggregate_embeddings(user_embeds_list), self._aggregate_embeddings(
            item_embeds_list)

        # Calculate BPR loss
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds_agg, item_embeds_agg, batch_data)
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]

        # Calculate alignment loss between different views
        alignment_loss = self._cal_alignment_loss(user_embeds_list, item_embeds_list, batch_data)

        # Calculate uniformity loss to prevent embedding collapse
        uniformity_loss = self._cal_uniformity_loss(user_embeds_list, item_embeds_list)

        # Calculate regularization loss
        reg_loss = self.reg_weight * reg_params(self)

        # Combine losses
        cl_loss = self.cl_weight * (alignment_loss + uniformity_loss)
        loss = bpr_loss + reg_loss + cl_loss

        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'alignment_loss': alignment_loss,
            'uniformity_loss': uniformity_loss,
            'cl_loss': cl_loss
        }

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
