'''import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop

init = nn.init.xavier_uniform_


class SimGCL_crv(BaseModel):
    def __init__(self, data_handler):
        super(SimGCL_crv, self).__init__(data_handler)
        # 原始协同图参数
        self.adj = data_handler.torch_adj
        self.semantic_adj = data_handler.semantic_adj
        self.edge_dropper = SpAdjEdgeDrop()
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']
        self.contrast_weight = configs['model']['contrastive_weight']

        # 超参数配置
        self.keep_rate = self.hyper_config['keep_rate']
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.contrast_temp = self.hyper_config['contrast_temp']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.eps = self.hyper_config['eps']

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
        # GAT层初始化
        for gat in self.gat_layers:
            init(gat.att_src)
            init(gat.att_dst)
            if hasattr(gat, 'lin'):
                init(gat.lin.weight)
            elif hasattr(gat, 'lin_src') and gat.lin_src is not None:
                init(gat.lin_src.weight)
                init(gat.lin_dst.weight)

    def _perturb_embedding(self, embeds):
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        return embeds + noise

    def _propagate_co(self, adj, embeds, perturb=False):
        """协同通道传播（带扰动）"""
        embeds_list = [embeds]
        for _ in range(self.layer_num):
            embeds = t.spmm(adj, embeds_list[-1])
            if perturb:
                embeds = self._perturb_embedding(embeds)
            embeds_list.append(embeds)
        return sum(embeds_list)

    def _propagate_sem(self, embeds):
        """语义通道传播"""
        edge_index = self.semantic_adj.indices()
        edge_weight = self.semantic_adj.values()
        for gat in self.gat_layers:
            embeds = gat(embeds, edge_index, edge_attr=edge_weight)
            embeds = F.leaky_relu(embeds)
        return embeds

    def forward(self, keep_rate=1.0, perturb=False):
        # 协同通道传播
        co_embeds = t.cat([self.user_embeds, self.item_embeds], 0)
        dropped_adj = self.edge_dropper(self.adj, keep_rate)
        co_embeds = self._propagate_co(dropped_adj, co_embeds, perturb)

        # 语义通道传播
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

        # 生成对比学习视图
        user_embeds1, item_embeds1 = self.forward(self.keep_rate, perturb=True)
        user_embeds2, item_embeds2 = self.forward(self.keep_rate, perturb=True)
        user_embeds3, item_embeds3 = self.forward(self.keep_rate, perturb=False)

        # 语义图嵌入
        user_embeds_sem, item_embeds_sem = self.forward(1.0, perturb=False)

        # 提取批次嵌入
        anc1, pos1, neg1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc2, pos2, neg2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc3, pos3, neg3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)
        anc_sem, pos_sem, _ = self._pick_embeds(user_embeds_sem, item_embeds_sem, batch_data)

        # 对比学习损失
        cl_loss = cal_infonce_loss(anc1, anc2, user_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(pos1, pos2, item_embeds2, self.cl_temperature)
        cl_loss /= anc1.shape[0]
        cl_loss *= self.cl_weight

        # 跨视图对比损失
        contrast_loss = cal_infonce_loss(anc3, anc_sem, anc_sem, self.contrast_temp) + \
                        cal_infonce_loss(pos3, pos_sem, pos_sem, self.contrast_temp)
        contrast_loss *= self.contrast_weight

        # BPR损失
        bpr_loss = cal_bpr_loss(anc3, pos3, neg3) / anc3.shape[0]

        # 正则化损失
        reg_loss = self.reg_weight * (
                t.norm(self.user_embeds) +
                t.norm(self.item_embeds) +
                sum(t.norm(p) for p in self.semantic_mlp.parameters()) +
                sum(t.norm(p) for p in self.gat_layers.parameters())
        )

        # 知识蒸馏损失
        usrprf_embeds = self.semantic_mlp(self.usrprf_embeds)
        itmprf_embeds = self.semantic_mlp(self.itmprf_embeds)
        anc_prf, pos_prf, neg_prf = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc3, anc_prf, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos3, pos_prf, pos_prf, self.kd_temperature) + \
                  cal_infonce_loss(neg3, neg_prf, neg_prf, self.kd_temperature)
        kd_loss /= anc3.shape[0]
        kd_loss *= self.kd_weight

        total_loss = bpr_loss + reg_loss + cl_loss + contrast_loss + kd_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'cl_loss': cl_loss,
            'contrast_loss': contrast_loss,
            'kd_loss': kd_loss
        }
        return total_loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds, item_embeds = self.forward(1.0, perturb=False)
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


class SimGCL_crv(BaseModel):
    def __init__(self, data_handler):
        super(SimGCL_crv, self).__init__(data_handler)
        # 原始协同图参数
        self.adj = data_handler.torch_adj
        self.semantic_adj = data_handler.semantic_adj
        self.keep_rate = self.hyper_config['keep_rate']
        self.edge_dropper = SpAdjEdgeDrop()
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']

        # SimGCL特有参数
        self.eps = self.hyper_config['eps']
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.contrast_weight = self.hyper_config['contrastive_weight']
        self.contrast_temp = self.hyper_config['contrast_temp']

        # 通用参数
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

        # 门控融合参数
        self.gate = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        # 初始化GAT权重
        for gat in self.gat_layers:
            init(gat.att_src)
            init(gat.att_dst)
            if hasattr(gat, 'lin'):
                init(gat.lin.weight)
            elif hasattr(gat, 'lin_src') and gat.lin_src is not None:
                init(gat.lin_src.weight)
                init(gat.lin_dst.weight)

        # 初始化语义MLP权重
        for m in self.semantic_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _perturb_embedding(self, embeds):
        """SimGCL中的扰动机制"""
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        return embeds + noise

    def _propagate_co(self, adj, embeds, perturb=False):
        """协同通道传播（带扰动）"""
        embeds_list = [embeds]
        for _ in range(self.layer_num):
            embeds = t.spmm(adj, embeds_list[-1])
            if perturb:
                embeds = self._perturb_embedding(embeds)
            embeds_list.append(embeds)
        return sum(embeds_list)  # 各层求和

    def _propagate_sem(self, embeds):
        """语义通道传播（GAT）"""
        edge_index = self.semantic_adj.indices()
        edge_weight = self.semantic_adj.values()
        for gat in self.gat_layers:
            embeds = gat(embeds, edge_index, edge_attr=edge_weight)
            embeds = F.leaky_relu(embeds)
        return embeds

    def forward(self, perturb=False, keep_rate=1.0):
        # 协同通道
        co_embeds = t.cat([self.user_embeds, self.item_embeds], 0)
        # 应用edge dropping
        dropped_adj = self.edge_dropper(self.adj, keep_rate)
        co_embeds = self._propagate_co(dropped_adj, co_embeds, perturb)

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

        # 两个扰动版本的前向传播（SimGCL特性）
        user_embeds1, item_embeds1 = self.forward(perturb=True, keep_rate=self.keep_rate)
        user_embeds2, item_embeds2 = self.forward(perturb=True, keep_rate=self.keep_rate)
        # 无扰动版本的前向传播
        user_embeds3, item_embeds3 = self.forward(perturb=False, keep_rate=self.keep_rate)
        # 无扰动无edge dropping的版本（用于对比学习）
        user_embeds_sem, item_embeds_sem = self.forward(perturb=False, keep_rate=1.0)

        # 提取batch中的嵌入
        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)
        anc_embeds_sem, pos_embeds_sem, _ = self._pick_embeds(user_embeds_sem, item_embeds_sem, batch_data)

        # 计算BPR损失（使用无扰动版本）
        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3)

        # 计算SimGCL的对比损失（在两个扰动版本之间）
        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_temperature)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight

        # 计算结构图和语义图之间的对比损失
        user_contrast_loss = cal_infonce_loss(anc_embeds3, anc_embeds_sem, anc_embeds_sem, self.contrast_temp)
        item_contrast_loss = cal_infonce_loss(pos_embeds3, pos_embeds_sem, pos_embeds_sem, self.contrast_temp)
        contrast_loss = (user_contrast_loss + item_contrast_loss) * self.contrast_weight

        # 计算知识蒸馏损失
        usrprf_embeds = self.semantic_mlp(self.usrprf_embeds)
        itmprf_embeds = self.semantic_mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc_embeds3, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds3, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds3, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds3.shape[0]
        kd_loss *= self.kd_weight

        # 正则化损失
        reg_loss = self.reg_weight * (
                t.norm(self.user_embeds) +
                t.norm(self.item_embeds) +
                sum(t.norm(p) for p in self.semantic_mlp.parameters()) +
                sum(t.norm(p) for p in self.gat_layers.parameters())
        )

        # 总损失
        loss = bpr_loss + reg_loss + kd_loss + cl_loss + contrast_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'kd_loss': kd_loss,
            'cl_loss': cl_loss,
            'contrast_loss': contrast_loss
        }
        return loss, losses

    def full_predict(self, batch_data):
        # 切换到预测模式
        self.is_training = False
        # 使用无扰动、无edge dropping的版本进行预测
        user_embeds, item_embeds = self.forward(perturb=False, keep_rate=1.0)

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
