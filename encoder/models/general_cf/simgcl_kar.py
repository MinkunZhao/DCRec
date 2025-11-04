import torch as t
from torch import nn
import torch.nn.functional as F
from encoder.config.configurator import configs
from encoder.models.general_cf.lightgcn import LightGCN
from encoder.models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_


class SimGCL_KAR(LightGCN):
    def __init__(self, data_handler):
        super(SimGCL_KAR, self).__init__(data_handler)

        # hyper-parameter
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['temperature']
        self.kg_weight = self.hyper_config['kg_weight']  # 知识增强权重
        self.kg_temperature = self.hyper_config['kg_temperature']  # 知识融合温度系数
        self.eps = self.hyper_config['eps']

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

    def _perturb_embedding(self, embeds):
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        return embeds + noise

    def forward(self, adj=None, perturb=False):
        if adj is None:
            adj = self.adj
        if not perturb:
            return super(SimGCL_KAR, self).forward(adj, 1.0)
        embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds = self._perturb_embedding(embeds)
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        return embeds[:self.user_num], embeds[self.user_num:]

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds1, item_embeds1 = self.forward(self.adj, perturb=True)
        user_embeds2, item_embeds2 = self.forward(self.adj, perturb=True)
        user_embeds3, item_embeds3 = self.forward(self.adj, perturb=False)

        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

        # 投影LLM知识嵌入到相同的空间
        usr_knowledge_embeds = self.knowledge_projector(self.usr_knowledge_embeds)
        itm_knowledge_embeds = self.knowledge_projector(self.itm_knowledge_embeds)

        # 获取相应的知识嵌入
        anc_knowledge_embeds, pos_knowledge_embeds, neg_knowledge_embeds = self._pick_embeds(
            usr_knowledge_embeds, itm_knowledge_embeds, batch_data)

        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]

        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_temperature)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight

        # 知识增强对比学习损失
        kg_loss = cal_infonce_loss(anc_embeds3, anc_knowledge_embeds, usr_knowledge_embeds, self.kg_temperature) + \
                  cal_infonce_loss(pos_embeds3, pos_knowledge_embeds, itm_knowledge_embeds, self.kg_temperature) + \
                  cal_infonce_loss(neg_embeds3, neg_knowledge_embeds, itm_knowledge_embeds, self.kg_temperature)
        kg_loss /= anc_embeds3.shape[0]
        kg_loss *= self.kg_weight

        reg_loss = self.reg_weight * reg_params(self)

        loss = bpr_loss + reg_loss + cl_loss + kg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'kg_loss': kg_loss}
        return loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, False)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
