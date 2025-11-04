import torch as t
from torch import nn
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class LightGCN_KAR(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_KAR, self).__init__(data_handler)
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
        self.kg_weight = self.hyper_config['kg_weight']  # 知识增强权重
        self.kg_temperature = self.hyper_config['kg_temperature']  # 知识融合温度系数

        # LLM-generated knowledge embeddings
        self.usr_knowledge_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itm_knowledge_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()

        # Knowledge fusion module
        self.knowledge_projector = nn.Sequential(
            nn.Linear(self.usr_knowledge_embeds.shape[1],
                      (self.usr_knowledge_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usr_knowledge_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.knowledge_projector:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
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

        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)

        # 投影LLM知识嵌入到相同的空间
        usr_knowledge_embeds = self.knowledge_projector(self.usr_knowledge_embeds)
        itm_knowledge_embeds = self.knowledge_projector(self.itm_knowledge_embeds)

        # 获取相应的知识嵌入
        anc_knowledge_embeds, pos_knowledge_embeds, neg_knowledge_embeds = self._pick_embeds(
            usr_knowledge_embeds, itm_knowledge_embeds, batch_data)

        # 主要推荐任务损失
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)
        reg_loss = self.reg_weight * reg_params(self)

        # 知识增强对比学习损失
        kg_loss = cal_infonce_loss(anc_embeds, anc_knowledge_embeds, usr_knowledge_embeds, self.kg_temperature) + \
                  cal_infonce_loss(pos_embeds, pos_knowledge_embeds, itm_knowledge_embeds, self.kg_temperature) + \
                  cal_infonce_loss(neg_embeds, neg_knowledge_embeds, itm_knowledge_embeds, self.kg_temperature)
        kg_loss /= anc_embeds.shape[0]
        kg_loss *= self.kg_weight

        loss = bpr_loss + reg_loss + kg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'kg_loss': kg_loss}
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
