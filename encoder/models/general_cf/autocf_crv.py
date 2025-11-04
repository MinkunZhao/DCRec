import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.base_model import BaseModel
from encoder.models.loss_utils import reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_


class AutoCF_crv(BaseModel):
    def __init__(self, data_handler):
        super(AutoCF_crv, self).__init__(data_handler)
        # 原始协同图参数
        self.adj = data_handler.torch_adj
        self.all_one_adj = self.make_all_one_adj()
        self.semantic_adj = data_handler.semantic_adj  # 新增语义图
        self.keep_rate = configs['model']['keep_rate']

        # 基础嵌入层
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
            ) for _ in range(self.hyper_config['layer_num'])
        ])

        # 门控融合参数
        self.gate = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.sigmoid = nn.Sigmoid()

        # AutoCF原有模块
        self.gt_layer = configs['model']['gt_layer']
        self.gcn_layer = self.hyper_config['gcn_layer']
        self.gcnLayers = nn.Sequential(*[GCNLayer() for _ in range(self.gcn_layer)])
        self.gtLayers = nn.Sequential(*[GTLayer() for _ in range(self.gt_layer)])
        self.masker = RandomMaskSubgraphs()
        self.sampler = LocalGraph()

        # 超参数
        self.reg_weight = self.hyper_config['reg_weight']
        self.ssl_reg = self.hyper_config['ssl_reg']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.contrast_weight = self.hyper_config['contrastive_weight']
        self.contrast_temp = self.hyper_config['contrast_temp']

        self._init_weights()

    def _init_weights(self):
        # 初始化GAT参数
        for gat in self.gat_layers:
            init(gat.att_src)
            init(gat.att_dst)
            if hasattr(gat, 'lin'):
                init(gat.lin.weight)
            elif hasattr(gat, 'lin_src'):
                init(gat.lin_src.weight)
                init(gat.lin_dst.weight)
        # 初始化语义MLP
        for layer in self.semantic_mlp:
            if isinstance(layer, nn.Linear):
                init(layer.weight)

    def _propagate_sem(self, embeds):
        """语义通道传播"""
        edge_index = self.semantic_adj.indices()
        edge_weight = self.semantic_adj.values()
        for gat in self.gat_layers:
            embeds = gat(embeds, edge_index, edge_attr=edge_weight)
            embeds = F.leaky_relu(embeds)
        return embeds

    def get_ego_embeds(self):
        return t.concat([self.user_embeds, self.item_embeds], axis=0)

    def sample_subgraphs(self):
        return self.sampler(self.all_one_adj, self.get_ego_embeds())

    def mask_subgraphs(self, seeds):
        return self.masker(self.adj, seeds)

    def make_all_one_adj(self):
        idxs = self.adj._indices()
        vals = t.ones_like(self.adj._values())
        return t.sparse.FloatTensor(idxs, vals, self.adj.shape).cuda()

    def forward(self, encoder_adj, decoder_adj=None):
        # 协同通道传播
        co_embeds = t.cat([self.user_embeds, self.item_embeds], 0)
        embeds_list = [co_embeds]
        for gcn in self.gcnLayers:
            co_embeds = gcn(encoder_adj, embeds_list[-1])
            embeds_list.append(co_embeds)
        if decoder_adj is not None:
            for gt in self.gtLayers:
                co_embeds = gt(decoder_adj, embeds_list[-1])
                embeds_list.append(co_embeds)
        co_embeds = sum(embeds_list)

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


    def contrast(self, nodes, allEmbeds, allEmbeds2=None):
        if allEmbeds2 is not None:
            pckEmbeds = allEmbeds[nodes]
            scores = t.log(t.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
        else:
            uniqNodes = t.unique(nodes)
            pckEmbeds = allEmbeds[uniqNodes]
            scores = t.log(t.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
        return scores


    def cal_loss(self, batch_data, encoder_adj, decoder_adj):
        # 双通道前向传播
        user_co, item_co = self.forward(encoder_adj, decoder_adj)
        user_sem, item_sem = self.forward(encoder_adj, decoder_adj)  # 使用相同参数获取语义通道

        # 对比损失计算
        ancs, poss, _ = batch_data
        anc_co, pos_co = user_co[ancs], item_co[poss]
        anc_sem, pos_sem = user_sem[ancs], item_sem[poss]
        user_contrast = cal_infonce_loss(anc_co, anc_sem, user_sem, self.contrast_temp)
        item_contrast = cal_infonce_loss(pos_co, pos_sem, item_sem, self.contrast_temp)
        contrast_loss = (user_contrast + item_contrast) * self.contrast_weight

        # AutoCF原有损失
        # rec_loss = (-t.sum(anc_co * pos_co, dim=-1)).mean()
        rec_loss = 0.3 * (-t.sum(anc_co * pos_co, dim=-1)).mean()
        reg_loss = reg_params(self) * self.reg_weight
        cl_loss = (self.contrast(ancs, user_co) + self.contrast(poss, item_co)) * self.ssl_reg

        # 知识蒸馏损失
        usrprf = self.semantic_mlp(self.usrprf_embeds)
        itmprf = self.semantic_mlp(self.itmprf_embeds)
        kd_loss = cal_infonce_loss(anc_co, usrprf[ancs], usrprf, self.kd_temperature) + \
                  cal_infonce_loss(pos_co, itmprf[poss], itmprf, self.kd_temperature)
        kd_loss = kd_loss / (anc_co.shape[0] * 2) * self.kd_weight

        # 总损失
        total_loss = rec_loss + reg_loss + cl_loss + kd_loss + contrast_loss
        losses = {
            'rec_loss': rec_loss,
            'reg_loss': reg_loss,
            'cl_loss': cl_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss
        }
        return total_loss, losses

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, self.adj)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)


class GTLayer(nn.Module):
    def __init__(self):
        super(GTLayer, self).__init__()

        self.head_num = configs['model']['head_num']
        self.embedding_size = configs['model']['embedding_size']

        self.qTrans = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
        self.kTrans = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
        self.vTrans = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))

    def forward(self, adj, embeds):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, self.head_num, self.embedding_size // self.head_num])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, self.head_num, self.embedding_size // self.head_num])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, self.head_num, self.embedding_size // self.head_num])

        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        tem = t.zeros([adj.shape[0], self.head_num]).cuda()
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)  # eh

        resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, self.embedding_size])
        tem = t.zeros([adj.shape[0], self.embedding_size]).cuda()
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd
        return resEmbeds


class LocalGraph(nn.Module):
    def __init__(self):
        super(LocalGraph, self).__init__()
        self.seed_num = configs['model']['seed_num']

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise[noise == 0] = 1e-8
        noise = -t.log(-t.log(noise))
        return t.log(scores) + noise

    def forward(self, allOneAdj, embeds):
        # allOneAdj should be without self-loop
        # embeds should be zero-order embeds
        order = t.sparse.sum(allOneAdj, dim=-1).to_dense().view([-1, 1])
        fstEmbeds = t.spmm(allOneAdj, embeds) - embeds
        fstNum = order
        scdEmbeds = (t.spmm(allOneAdj, fstEmbeds) - fstEmbeds) - order * embeds
        scdNum = (t.spmm(allOneAdj, fstNum) - fstNum) - order
        subgraphEmbeds = (fstEmbeds + scdEmbeds) / (fstNum + scdNum + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
        embeds = F.normalize(embeds, p=2)
        scores = t.sigmoid(t.sum(subgraphEmbeds * embeds, dim=-1))
        scores = self.makeNoise(scores)
        _, seeds = t.topk(scores, self.seed_num)
        return scores, seeds


class RandomMaskSubgraphs(nn.Module):
    def __init__(self):
        super(RandomMaskSubgraphs, self).__init__()
        self.flag = False
        self.mask_depth = configs['model']['mask_depth']
        self.keep_rate = configs['model']['keep_rate']
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']

    def normalizeAdj(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds):
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        maskNodes = [seeds]

        for i in range(self.mask_depth):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = list()
            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                idct = t.logical_or(rowIdct, colIdct)

                if i != self.mask_depth - 1:
                    mskRows = rows[idct]
                    mskCols = cols[idct]
                    nxtSeeds.append(mskRows)
                    nxtSeeds.append(mskCols)

                rows = rows[t.logical_not(idct)]
                cols = cols[t.logical_not(idct)]
            if len(nxtSeeds) > 0:
                nxtSeeds = t.unique(t.concat(nxtSeeds))
                maskNodes.append(nxtSeeds)
        sampNum = int((self.user_num + self.item_num) * self.keep_rate)
        sampedNodes = t.randint(self.user_num + self.item_num, size=[sampNum]).cuda()
        if self.flag == False:
            l1 = adj._values().shape[0]
            l2 = rows.shape[0]
            print('-----')
            print('LENGTH CHANGE', '%.2f' % (l2 / l1), l2, l1)
            tem = t.unique(t.concat(maskNodes))
            print('Original SAMPLED NODES', '%.2f' % (tem.shape[0] / (self.user_num + self.item_num)), tem.shape[0],
                  (self.user_num + self.item_num))
        maskNodes.append(sampedNodes)
        maskNodes = t.unique(t.concat(maskNodes))
        if self.flag == False:
            print('AUGMENTED SAMPLED NODES', '%.2f' % (maskNodes.shape[0] / (self.user_num + self.item_num)),
                  maskNodes.shape[0], (self.user_num + self.item_num))
            self.flag = True
            print('-----')

        encoder_adj = self.normalizeAdj(
            t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows).cuda(), adj.shape))

        temNum = maskNodes.shape[0]
        temRows = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]
        temCols = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]

        newRows = t.concat([temRows, temCols, t.arange(self.user_num + self.item_num).cuda(), rows])
        newCols = t.concat([temCols, temRows, t.arange(self.user_num + self.item_num).cuda(), cols])

        # filter duplicated
        hashVal = newRows * (self.user_num + self.item_num) + newCols
        hashVal = t.unique(hashVal)
        newCols = hashVal % (self.user_num + self.item_num)
        newRows = ((hashVal - newCols) / (self.user_num + self.item_num)).long()

        decoder_adj = t.sparse.FloatTensor(t.stack([newRows, newCols], dim=0), t.ones_like(newRows).cuda().float(),
                                           adj.shape)
        return encoder_adj, decoder_adj




'''import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.base_model import BaseModel
from encoder.models.loss_utils import cal_infonce_loss, reg_params

init = nn.init.xavier_uniform_


class AutoCF_crv(BaseModel):
    def __init__(self, data_handler):
        super(AutoCF_crv, self).__init__(data_handler)

        # 原始协同图参数
        self.adj = data_handler.torch_adj
        self.semantic_adj = data_handler.semantic_adj  # 需要提前构建语义图
        self.all_one_adj = self.make_all_one_adj()
        self.keep_rate = configs['model']['keep_rate']

        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']

        # 超参配置
        self.gt_layer = configs['model']['gt_layer']
        self.gcn_layer = self.hyper_config['gcn_layer']
        self.reg_weight = self.hyper_config['reg_weight']
        self.ssl_reg = self.hyper_config['ssl_reg']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.contrast_weight = self.hyper_config['contrastive_weight']
        self.contrast_temp = self.hyper_config['contrast_temp']

        # 协同通道embedding
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        # 协同通道参数
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.gcn_layer)])
        self.gtLayers = nn.Sequential(*[GTLayer() for i in range(self.gt_layer)])

        self.masker = RandomMaskSubgraphs()
        self.sampler = LocalGraph()

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
                edge_dim=1  # 添加此行以支持边权重
            )
            for _ in range(self.gcn_layer)  # 使用与GCN相同的层数
        ])

        # 门控融合参数
        self.gate = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        # 初始化GAT参数
        for gat in self.gat_layers:
            # 初始化注意力参数
            init(gat.att_src)
            init(gat.att_dst)
            # 初始化线性变换参数（当concat=False时存在lin）
            if hasattr(gat, 'lin'):
                init(gat.lin.weight)
            elif hasattr(gat, 'lin_src') and gat.lin_src is not None:
                init(gat.lin_src.weight)
                init(gat.lin_dst.weight)

        # 初始化MLP参数
        for m in self.semantic_mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def make_all_one_adj(self):
        idxs = self.adj._indices()
        vals = t.ones_like(self.adj._values())
        shape = self.adj.shape
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def get_ego_embeds(self):
        return t.concat([self.user_embeds, self.item_embeds], axis=0)

    def sample_subgraphs(self):
        return self.sampler(self.all_one_adj, self.get_ego_embeds())

    def mask_subgraphs(self, seeds):
        return self.masker(self.adj, seeds)

    def _propagate_co(self, encoder_adj, decoder_adj=None):
        """协同通道传播"""
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embedsLst = [embeds]
        for i, gcn in enumerate(self.gcnLayers):
            embeds = gcn(encoder_adj, embedsLst[-1])
            embedsLst.append(embeds)
        if decoder_adj is not None:
            for gt in self.gtLayers:
                embeds = gt(decoder_adj, embedsLst[-1])
                embedsLst.append(embeds)
        embeds = sum(embedsLst)
        return embeds[:self.user_num], embeds[self.user_num:]

    def _propagate_sem(self, embeds):
        """语义通道传播"""
        edge_index = self.semantic_adj.indices()
        edge_weight = self.semantic_adj.values()
        for gat in self.gat_layers:
            embeds = gat(embeds, edge_index, edge_attr=edge_weight)
            embeds = F.leaky_relu(embeds)
        return embeds

    def forward(self, encoder_adj, decoder_adj=None, is_semantic=False):
        if is_semantic:
            # 语义通道
            sem_embeds = t.cat([
                self.semantic_mlp(self.usrprf_embeds),
                self.semantic_mlp(self.itmprf_embeds)
            ], 0)
            sem_embeds = self._propagate_sem(sem_embeds)
            return sem_embeds[:self.user_num], sem_embeds[self.user_num:]
        else:
            # 协同通道
            co_user_embeds, co_item_embeds = self._propagate_co(encoder_adj, decoder_adj)

            # 语义通道
            sem_embeds = t.cat([
                self.semantic_mlp(self.usrprf_embeds),
                self.semantic_mlp(self.itmprf_embeds)
            ], 0)
            sem_embeds = self._propagate_sem(sem_embeds)
            sem_user_embeds, sem_item_embeds = sem_embeds[:self.user_num], sem_embeds[self.user_num:]

            # 门控融合
            # 用户部分
            user_gate_input = t.cat([co_user_embeds, sem_user_embeds], dim=1)
            user_gate = self.sigmoid(self.gate(user_gate_input))
            fused_user_embeds = user_gate * co_user_embeds + (1 - user_gate) * sem_user_embeds

            # 物品部分
            item_gate_input = t.cat([co_item_embeds, sem_item_embeds], dim=1)
            item_gate = self.sigmoid(self.gate(item_gate_input))
            fused_item_embeds = item_gate * co_item_embeds + (1 - item_gate) * sem_item_embeds

            return fused_user_embeds, fused_item_embeds

    def contrast(self, nodes, allEmbeds, allEmbeds2=None):
        if allEmbeds2 is not None:
            pckEmbeds = allEmbeds[nodes]
            scores = t.log(t.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
        else:
            uniqNodes = t.unique(nodes)
            pckEmbeds = allEmbeds[uniqNodes]
            scores = t.log(t.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
        return scores

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs] if negs is not None else None
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        # 生成子图
        scores, seeds = self.sample_subgraphs()
        encoder_adj, decoder_adj = self.mask_subgraphs(seeds)

        # 协同图前向传播
        user_embeds_struct, item_embeds_struct = self.forward(encoder_adj, decoder_adj)
        # 语义图前向传播（不需要mask，保持完整图结构）
        user_embeds_sem, item_embeds_sem = self.forward(self.adj, self.adj, is_semantic=True)

        # 提取当前batch的嵌入
        ancs, poss, negs = batch_data
        anc_embeds_struct, pos_embeds_struct, _ = self._pick_embeds(user_embeds_struct, item_embeds_struct, batch_data)
        anc_embeds_sem, pos_embeds_sem, _ = self._pick_embeds(user_embeds_sem, item_embeds_sem, batch_data)

        # 计算对比损失（用户和正物品）
        user_contrast_loss = cal_infonce_loss(anc_embeds_struct, anc_embeds_sem, anc_embeds_sem, self.contrast_temp)
        item_contrast_loss = cal_infonce_loss(pos_embeds_struct, pos_embeds_sem, pos_embeds_sem, self.contrast_temp)
        contrast_loss = (user_contrast_loss + item_contrast_loss) * self.contrast_weight

        # 主融合模型前向传播
        user_embeds, item_embeds = self.forward(encoder_adj, decoder_adj)
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)

        # 计算主要推荐损失
        rec_loss = (-t.sum(anc_embeds * pos_embeds, dim=-1)).mean()

        # 计算原始AutoCF的自对比损失
        cl_loss = (self.contrast(ancs, user_embeds) + self.contrast(poss, item_embeds)) * self.ssl_reg + self.contrast(
            ancs, user_embeds, item_embeds)

        # 计算正则化损失
        reg_loss = self.reg_weight * (
                t.norm(self.user_embeds) +
                t.norm(self.item_embeds) +
                sum(t.norm(p) for p in self.semantic_mlp.parameters()) +
                sum(t.norm(p) for p in self.gat_layers.parameters()) +
                sum(t.norm(p) for p in self.gate.parameters())
        )

        # 计算知识蒸馏损失
        usrprf_embeds = self.semantic_mlp(self.usrprf_embeds)
        itmprf_embeds = self.semantic_mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature)
        if negs is not None:
            kd_loss += cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        # 计算总损失
        loss = rec_loss + reg_loss + cl_loss + kd_loss + contrast_loss
        losses = {
            'rec_loss': rec_loss,
            'reg_loss': reg_loss,
            'cl_loss': cl_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss
        }
        return loss, losses

    def full_predict(self, batch_data):
        # 全图预测时使用完整图结构
        user_embeds, item_embeds = self.forward(self.adj, self.adj)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds


# 保留原有的辅助类
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)


class GTLayer(nn.Module):
    def __init__(self):
        super(GTLayer, self).__init__()

        self.head_num = configs['model']['head_num']
        self.embedding_size = configs['model']['embedding_size']

        self.qTrans = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
        self.kTrans = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))
        self.vTrans = nn.Parameter(init(t.empty(self.embedding_size, self.embedding_size)))

    def forward(self, adj, embeds):
        indices = adj._indices()
        rows, cols = indices[0, :], indices[1, :]
        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, self.head_num, self.embedding_size // self.head_num])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, self.head_num, self.embedding_size // self.head_num])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, self.head_num, self.embedding_size // self.head_num])

        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        tem = t.zeros([adj.shape[0], self.head_num]).cuda()
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        att = expAtt / (attNorm + 1e-8)  # eh

        resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, self.embedding_size])
        tem = t.zeros([adj.shape[0], self.embedding_size]).cuda()
        resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd
        return resEmbeds


class LocalGraph(nn.Module):
    def __init__(self):
        super(LocalGraph, self).__init__()
        self.seed_num = configs['model']['seed_num']

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise[noise == 0] = 1e-8
        noise = -t.log(-t.log(noise))
        return t.log(scores) + noise

    def forward(self, allOneAdj, embeds):
        # allOneAdj should be without self-loop
        # embeds should be zero-order embeds
        order = t.sparse.sum(allOneAdj, dim=-1).to_dense().view([-1, 1])
        fstEmbeds = t.spmm(allOneAdj, embeds) - embeds
        fstNum = order
        scdEmbeds = (t.spmm(allOneAdj, fstEmbeds) - fstEmbeds) - order * embeds
        scdNum = (t.spmm(allOneAdj, fstNum) - fstNum) - order
        subgraphEmbeds = (fstEmbeds + scdEmbeds) / (fstNum + scdNum + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)
        embeds = F.normalize(embeds, p=2)
        scores = t.sigmoid(t.sum(subgraphEmbeds * embeds, dim=-1))
        scores = self.makeNoise(scores)
        _, seeds = t.topk(scores, self.seed_num)
        return scores, seeds


class RandomMaskSubgraphs(nn.Module):
    def __init__(self):
        super(RandomMaskSubgraphs, self).__init__()
        self.flag = False
        self.mask_depth = configs['model']['mask_depth']
        self.keep_rate = configs['model']['keep_rate']
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']

    def normalizeAdj(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds):
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        maskNodes = [seeds]

        for i in range(self.mask_depth):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = list()
            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                idct = t.logical_or(rowIdct, colIdct)

                if i != self.mask_depth - 1:
                    mskRows = rows[idct]
                    mskCols = cols[idct]
                    nxtSeeds.append(mskRows)
                    nxtSeeds.append(mskCols)

                rows = rows[t.logical_not(idct)]
                cols = cols[t.logical_not(idct)]
            if len(nxtSeeds) > 0:
                nxtSeeds = t.unique(t.concat(nxtSeeds))
                maskNodes.append(nxtSeeds)
        sampNum = int((self.user_num + self.item_num) * self.keep_rate)
        sampedNodes = t.randint(self.user_num + self.item_num, size=[sampNum]).cuda()
        if self.flag == False:
            l1 = adj._values().shape[0]
            l2 = rows.shape[0]
            print('-----')
            print('LENGTH CHANGE', '%.2f' % (l2 / l1), l2, l1)
            tem = t.unique(t.concat(maskNodes))
            print('Original SAMPLED NODES', '%.2f' % (tem.shape[0] / (self.user_num + self.item_num)), tem.shape[0],
                  (self.user_num + self.item_num))
        maskNodes.append(sampedNodes)
        maskNodes = t.unique(t.concat(maskNodes))
        if self.flag == False:
            print('AUGMENTED SAMPLED NODES', '%.2f' % (maskNodes.shape[0] / (self.user_num + self.item_num)),
                  maskNodes.shape[0], (self.user_num + self.item_num))
            self.flag = True
            print('-----')

        encoder_adj = self.normalizeAdj(
            t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows).cuda(), adj.shape))

        temNum = maskNodes.shape[0]
        temRows = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]
        temCols = maskNodes[t.randint(temNum, size=[adj._values().shape[0]]).cuda()]

        newRows = t.concat([temRows, temCols, t.arange(self.user_num + self.item_num).cuda(), rows])
        newCols = t.concat([temCols, temRows, t.arange(self.user_num + self.item_num).cuda(), cols])

        # filter duplicated
        hashVal = newRows * (self.user_num + self.item_num) + newCols
        hashVal = t.unique(hashVal)
        newCols = hashVal % (self.user_num + self.item_num)
        newRows = ((hashVal - newCols) / (self.user_num + self.item_num)).long()

        decoder_adj = t.sparse.FloatTensor(t.stack([newRows, newCols], dim=0), t.ones_like(newRows).cuda().float(),
                                           adj.shape)
        return encoder_adj, decoder_adj'''


