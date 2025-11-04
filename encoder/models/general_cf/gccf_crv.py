'''import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class GCNLayer(nn.Module):
    def __init__(self, latdim):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(init(t.empty(latdim, latdim)))

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)  # 保持与GCCF_plus一致的传播方式

class GCCF_crv(BaseModel):
    def __init__(self, data_handler):
        super(GCCF_crv, self).__init__(data_handler)
        # 原始协同图参数
        self.adj = data_handler.torch_adj
        self.semantic_adj = data_handler.semantic_adj
        self.keep_rate = self.hyper_config['keep_rate']
        self.edge_dropper = SpAdjEdgeDrop()
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']
        self.contrast_weight = self.hyper_config['contrastive_weight']
        self.contrast_temp = self.hyper_config['contrast_temp']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']

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

        # GCCF参数
        self.gcnLayers = nn.ModuleList([GCNLayer(self.embedding_size) for _ in range(self.layer_num)])

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

    def _propagate_co(self, adj, embeds):
        """协同通道传播（保持GCCF的拼接方式）"""
        embeds_list = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embeds_list[-1])
            embeds_list.append(embeds)
        return t.cat(embeds_list, dim=-1)  # 各层拼接

    def _propagate_sem(self, embeds):
        """语义通道传播"""
        edge_index = self.semantic_adj.indices()
        edge_weight = self.semantic_adj.values()
        for gat in self.gat_layers:
            embeds = gat(embeds, edge_index, edge_attr=edge_weight)
            embeds = F.leaky_relu(embeds)
        return embeds

    def forward(self, keep_rate=1.0):
        # 协同通道
        co_embeds = t.cat([self.user_embeds, self.item_embeds], 0)
        dropped_adj = self.edge_dropper(self.adj, keep_rate)
        co_embeds = self._propagate_co(dropped_adj, co_embeds)

        # 语义通道
        sem_embeds = t.cat([
            self.semantic_mlp(self.usrprf_embeds),
            self.semantic_mlp(self.itmprf_embeds)
        ], 0)
        sem_embeds = self._propagate_sem(sem_embeds)

        # 维度对齐（因协同通道输出是拼接后的维度）
        co_embeds = co_embeds[:, :self.embedding_size]  # 取首段embedding用于门控

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
        # 结构图前向传播
        user_embeds_struct, item_embeds_struct = self.forward(self.keep_rate)
        # 语义图前向传播
        user_embeds_sem, item_embeds_sem = self.forward(1.0)

        # 提取当前batch的嵌入
        anc_embeds_struct, pos_embeds_struct, _ = self._pick_embeds(user_embeds_struct, item_embeds_struct, batch_data)
        anc_embeds_sem, pos_embeds_sem, _ = self._pick_embeds(user_embeds_sem, item_embeds_sem, batch_data)

        # 计算对比损失
        user_contrast_loss = cal_infonce_loss(anc_embeds_struct, anc_embeds_sem, anc_embeds_sem, self.contrast_temp)
        item_contrast_loss = cal_infonce_loss(pos_embeds_struct, pos_embeds_sem, pos_embeds_sem, self.contrast_temp)
        contrast_loss = (user_contrast_loss + item_contrast_loss) * self.contrast_weight

        user_embeds, item_embeds = self.forward(self.keep_rate)
        anc, pos, neg = batch_data

        # BPR损失
        anc_embeds = user_embeds[anc]
        pos_embeds = item_embeds[pos]
        neg_embeds = item_embeds[neg]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)

        # 正则化
        reg_loss = self.reg_weight * (
            t.norm(self.user_embeds) +
            t.norm(self.item_embeds) +
            sum(t.norm(p) for p in self.semantic_mlp.parameters()) +
            sum(t.norm(p) for p in self.gat_layers.parameters())
        )

        # 知识蒸馏损失
        usrprf_embeds = self.semantic_mlp(self.usrprf_embeds)
        itmprf_embeds = self.semantic_mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

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
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        if train_mask is not None:
            full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds'''




import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class GCNLayer(nn.Module):
    def __init__(self, latdim):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(init(t.empty(latdim, latdim)))

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)


class GCCF_CRV(BaseModel):
    def __init__(self, data_handler):
        super(GCCF_CRV, self).__init__(data_handler)
        # 原始协同图参数
        self.adj = data_handler.torch_adj
        self.semantic_adj = data_handler.semantic_adj
        self.keep_rate = self.hyper_config['keep_rate']
        self.edge_dropper = SpAdjEdgeDrop()
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']
        self.contrast_weight = self.hyper_config['contrastive_weight']
        self.contrast_temp = self.hyper_config['contrast_temp']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']

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

        # GCCF的GCN层
        self.gcnLayers = nn.ModuleList([GCNLayer(self.embedding_size) for _ in range(self.layer_num)])

        # GAT层
        self.gat_layers = nn.ModuleList([GATConv(
            self.embedding_size,
            self.embedding_size,
            heads=self.hyper_config['gat_heads'],
            concat=False,
            add_self_loops=False,
            edge_dim=1  # 支持边权重
        ) for _ in range(self.hyper_config['layer_num'])])

        # 门控融合
        self.gate = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        # 初始化GAT和GCN层权重
        for gat in self.gat_layers:
            init(gat.att_src)
            init(gat.att_dst)
            if hasattr(gat, 'lin'):
                init(gat.lin.weight)
            elif hasattr(gat, 'lin_src') and gat.lin_src is not None:
                init(gat.lin_src.weight)
                init(gat.lin_dst.weight)

        for gcn in self.gcnLayers:
            init(gcn.W)

    def _propagate_co(self, adj, embeds):
        """协同通道传播"""
        embeds_list = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embeds_list[-1])
            embeds_list.append(embeds)
        return sum(embeds_list)  # 各层求和

    def _propagate_sem(self, embeds):
        edge_index = self.semantic_adj.indices()
        edge_weight = self.semantic_adj.values()
        for gat in self.gat_layers:
            embeds = gat(embeds, edge_index, edge_attr=edge_weight)
            embeds = F.leaky_relu(embeds)
        return embeds

    def forward(self, keep_rate=1.0):  # 修改函数签名
        # 协同通道
        co_embeds = t.cat([self.user_embeds, self.item_embeds], 0)
        dropped_adj = self.edge_dropper(self.adj, keep_rate)
        co_embeds = self._propagate_co(dropped_adj, co_embeds)

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
        # 结构图前向传播
        user_embeds_struct, item_embeds_struct = self.forward(self.keep_rate)
        # 语义图前向传播
        user_embeds_sem, item_embeds_sem = self.forward(1.0)

        # 提取当前batch的嵌入
        anc_embeds_struct, pos_embeds_struct, _ = self._pick_embeds(user_embeds_struct, item_embeds_struct, batch_data)
        anc_embeds_sem, pos_embeds_sem, _ = self._pick_embeds(user_embeds_sem, item_embeds_sem, batch_data)

        # 计算对比损失
        user_contrast_loss = cal_infonce_loss(anc_embeds_struct, anc_embeds_sem, anc_embeds_sem, self.contrast_temp)
        item_contrast_loss = cal_infonce_loss(pos_embeds_struct, pos_embeds_sem, pos_embeds_sem, self.contrast_temp)
        contrast_loss = (user_contrast_loss + item_contrast_loss) * self.contrast_weight

        user_embeds, item_embeds = self.forward(self.keep_rate)
        anc, pos, neg = batch_data

        # BPR损失
        anc_embeds = user_embeds[anc]
        pos_embeds = item_embeds[pos]
        neg_embeds = item_embeds[neg]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)

        # 正则化
        reg_loss = self.reg_weight * (
                t.norm(self.user_embeds) +
                t.norm(self.item_embeds) +
                sum(t.norm(p) for p in self.semantic_mlp.parameters()) +
                sum(t.norm(p) for p in self.gat_layers.parameters())
        )

        usrprf_embeds = self.semantic_mlp(self.usrprf_embeds)
        itmprf_embeds = self.semantic_mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        loss = bpr_loss + reg_loss + kd_loss + contrast_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss  # 新增对比损失记录
        }
        return loss, losses

    def full_predict(self, batch_data):
        # 切换到预测模式
        self.is_training = False
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

