import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop, NodeDrop

init = nn.init.xavier_uniform_


class SGL_crv(BaseModel):
    def __init__(self, data_handler):
        super(SGL_crv, self).__init__(data_handler)
        # 原始协同图参数
        self.adj = data_handler.torch_adj
        self.semantic_adj = data_handler.semantic_adj
        self.edge_dropper = SpAdjEdgeDrop()
        self.node_dropper = NodeDrop()
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']
        self.augmentation = configs['model']['augmentation']

        # 对比学习参数
        self.keep_rate = self.hyper_config['keep_rate']
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.contrast_weight = self.hyper_config['contrastive_weight']
        self.contrast_temp = self.hyper_config['contrast_temp']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']

        # 协同通道embedding
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

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
                edge_dim=1
            )
            for _ in range(self.hyper_config['layer_num'])
        ])

        # 门控融合参数
        self.gate = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for gat in self.gat_layers:
            init(gat.att_src)
            init(gat.att_dst)
            if hasattr(gat, 'lin'):
                init(gat.lin.weight)
            elif hasattr(gat, 'lin_src') and gat.lin_src is not None:
                init(gat.lin_src.weight)
                init(gat.lin_dst.weight)

    def _propagate_co(self, adj, embeds, keep_rate):
        if self.augmentation == 'node_drop':
            embeds = self.node_dropper(embeds, keep_rate)
        embeds_list = [embeds]
        for _ in range(self.layer_num):
            if self.augmentation == 'edge_drop':
                adj = self.edge_dropper(adj, keep_rate)
            embeds = t.spmm(adj, embeds_list[-1])
            embeds_list.append(embeds)
        return sum(embeds_list)

    def _propagate_sem(self, embeds):
        edge_index = self.semantic_adj.indices()
        edge_weight = self.semantic_adj.values()
        for gat in self.gat_layers:
            embeds = gat(embeds, edge_index, edge_attr=edge_weight)
            embeds = F.leaky_relu(embeds)
        return embeds

    def forward(self, keep_rate=1.0):
        # 协同通道
        co_embeds = t.cat([self.user_embeds, self.item_embeds], 0)
        co_embeds = self._propagate_co(self.adj, co_embeds, keep_rate)

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

        return fused_embeds[:self.user_num], fused_embeds[self.user_num:]

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        # 生成两个增强视图
        user_embeds1, item_embeds1 = self.forward(self.keep_rate)
        user_embeds2, item_embeds2 = self.forward(self.keep_rate)
        # 完整视图（用于BPR和KD）
        user_embeds3, item_embeds3 = self.forward(1.0)

        # 语义视图（用于对比）
        user_sem_embeds, item_sem_embeds = self.forward(1.0)

        # 提取batch嵌入
        anc1, pos1, neg1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc2, pos2, neg2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc3, pos3, neg3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)
        anc_sem, pos_sem, _ = self._pick_embeds(user_sem_embeds, item_sem_embeds, batch_data)

        # 对比学习损失（视图间）
        cl_loss = cal_infonce_loss(anc1, anc2, user_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(pos1, pos2, item_embeds2, self.cl_temperature)
        cl_loss /= anc1.shape[0]
        cl_loss *= self.cl_weight

        # 结构-语义对比损失
        contrast_loss = cal_infonce_loss(anc3, anc_sem, user_sem_embeds, self.contrast_temp) + \
                        cal_infonce_loss(pos3, pos_sem, item_sem_embeds, self.contrast_temp)
        contrast_loss *= self.contrast_weight

        # BPR损失
        bpr_loss = cal_bpr_loss(anc3, pos3, neg3) / anc3.shape[0]

        # 知识蒸馏损失
        usrprf_embeds = self.semantic_mlp(self.usrprf_embeds)
        itmprf_embeds = self.semantic_mlp(self.itmprf_embeds)
        anc_prf, pos_prf, neg_prf = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc3, anc_prf, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos3, pos_prf, itmprf_embeds, self.kd_temperature)
        kd_loss /= anc3.shape[0]
        kd_loss *= self.kd_weight

        # 正则化
        reg_loss = self.reg_weight * (
                t.norm(self.user_embeds) + t.norm(self.item_embeds) +
                sum(t.norm(p) for p in self.semantic_mlp.parameters()) +
                sum(t.norm(p) for p in self.gat_layers.parameters())
        )

        total_loss = bpr_loss + reg_loss + cl_loss + kd_loss + contrast_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'cl_loss': cl_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss
        }
        return total_loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds, item_embeds = self.forward(1.0)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        if train_mask is not None:
            full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
