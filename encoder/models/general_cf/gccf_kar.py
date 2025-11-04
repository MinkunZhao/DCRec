import torch as t
from torch import nn
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from encoder.models.base_model import BaseModel

init = nn.init.xavier_uniform_


class GCNLayer(nn.Module):
    def __init__(self, latdim):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(init(t.empty(latdim, latdim)))

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)  # @ self.W (Performs better without W)


class GCCF_KAR(BaseModel):
    def __init__(self, data_handler):
        super(GCCF_KAR, self).__init__(data_handler)

        self.adj = data_handler.torch_adj

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.kg_weight = self.hyper_config['kg_weight']  # 知识增强权重
        self.kg_temperature = self.hyper_config['kg_temperature']  # 知识融合温度系数

        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.gcnLayers = nn.Sequential(*[GCNLayer(self.embedding_size) for i in range(self.layer_num)])
        self.is_training = True

        # LLM-generated knowledge embeddings
        self.usr_knowledge_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itm_knowledge_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()

        # Knowledge fusion module
        output_size = int((self.layer_num + 1) * self.embedding_size)
        self.knowledge_projector = nn.Sequential(
            nn.Linear(self.usr_knowledge_embeds.shape[1], (self.usr_knowledge_embeds.shape[1] + output_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usr_knowledge_embeds.shape[1] + output_size) // 2, output_size)
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.knowledge_projector:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def forward(self, adj=None):
        if adj is None:
            adj = self.adj
        if not self.is_training:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = t.concat(embeds_list, dim=-1)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:], embeds_list[-1]

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
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)

        # 投影LLM知识嵌入到相同的空间
        usr_knowledge_embeds = self.knowledge_projector(self.usr_knowledge_embeds)
        itm_knowledge_embeds = self.knowledge_projector(self.itm_knowledge_embeds)

        # 获取相应的知识嵌入
        anc_knowledge_embeds, pos_knowledge_embeds, neg_knowledge_embeds = self._pick_embeds(
            usr_knowledge_embeds, itm_knowledge_embeds, batch_data)

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
        user_embeds, item_embeds, _ = self.forward(self.adj)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
