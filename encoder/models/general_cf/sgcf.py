import torch as t
from torch import nn
import torch.nn.functional as F
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class SGCF(BaseModel):
    def __init__(self, data_handler):
        super(SGCF, self).__init__(data_handler)
        # 基础参数
        self.adj = data_handler.torch_adj
        self.keep_rate = self.hyper_config['keep_rate']
        self.edge_dropper = SpAdjEdgeDrop()
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']

        # SGCF只需要基础的embedding
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

    def _propagate(self, adj, embeds):
        """SGCF核心传播机制 - 简化的图卷积
        核心思想：移除权重矩阵和非线性激活，仅保留消息传递
        """
        all_embeds = [embeds]
        for _ in range(self.layer_num):
            # 简单的邻居聚合，无权重矩阵和非线性激活
            embeds = t.spmm(adj, all_embeds[-1])
            all_embeds.append(embeds)

        # SGCF特点：对每层结果做简单平均，不使用拼接
        return sum(all_embeds) / len(all_embeds)

    def forward(self, keep_rate=1.0):
        # 合并用户物品嵌入
        all_embeds = t.cat([self.user_embeds, self.item_embeds], 0)

        # 边丢弃（训练时的正则化技术）
        dropped_adj = self.edge_dropper(self.adj, keep_rate)

        # 图传播
        all_embeds = self._propagate(dropped_adj, all_embeds)

        # 分离用户和物品嵌入
        user_embeds, item_embeds = t.split(all_embeds, [self.user_num, self.item_num])

        return user_embeds, item_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.keep_rate)
        anc, pos, neg = batch_data

        # 提取当前batch的嵌入
        anc_embeds = user_embeds[anc]
        pos_embeds = item_embeds[pos]
        neg_embeds = item_embeds[neg]

        # BPR损失 - SGCF使用基本的BPR损失
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)

        # L2正则化 - SGCF使用简单的正则化方式
        reg_loss = self.reg_weight * (
                t.norm(self.user_embeds) +
                t.norm(self.item_embeds)
        )

        # 总损失
        loss = bpr_loss + reg_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss
        }
        return loss, losses

    def full_predict(self, batch_data):
        # 切换到预测模式
        self.is_training = False
        user_embeds, item_embeds = self.forward(1.0)  # 预测时不做边丢弃

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
