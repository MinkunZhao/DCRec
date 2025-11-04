import random
import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.base_model import BaseModel
from encoder.models.loss_utils import reg_params, cal_infonce_loss
import json
import requests
import os
from tqdm import tqdm
import time

init = nn.init.xavier_uniform_


class LLMGuideManager:
    """管理LLM API调用，为门控融合提供监督信号"""

    def __init__(self, user_num, item_num, embedding_size):
        # LLM API配置
        self.api_type = configs['model']['llm_model']
        self.api_key = configs['model']['api_key']
        self.api_url = "https://api.anthropic.com/v1/messages" if self.api_type == "claude" else "https://api.openai.com/v1/chat/completions"

        # 维度信息
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = embedding_size

        # 加载画像数据
        try:
            with open('../data/{}/usr_prf.json'.format(configs['data']['name']), 'r', encoding='utf-8') as f:
                self.user_profiles = json.load(f)
            with open('../data/{}/itm_prf.json'.format(configs['data']['name']), 'r', encoding='utf-8') as f:
                self.item_profiles = json.load(f)
            print(
                f"Successfully loaded profiles for {len(self.user_profiles)} users and {len(self.item_profiles)} items")
        except Exception as e:
            print(f"Error loading profiles: {e}")
            self.user_profiles = {}
            self.item_profiles = {}

        # 缓存标签结果
        self.label_cache_path = "llm_gate_labels_{}.pt".format(configs['data']['name'])
        self.load_or_create_labels()

    def load_or_create_labels(self):
        """加载或创建标签缓存"""
        if os.path.exists(self.label_cache_path):
            try:
                self.gate_labels = t.load(self.label_cache_path)
                print(f"Loaded gate labels from cache with shape {self.gate_labels.shape}")
                return
            except:
                print("Failed to load cached labels, will create new ones")

        # 初始化为0.5（平衡两个通道）
        self.gate_labels = t.ones(self.user_num + self.item_num, self.embedding_size) * 0.5
        self.gate_labels = self.gate_labels.cuda()

    def save_labels(self):
        """保存标签到缓存"""
        t.save(self.gate_labels, self.label_cache_path)
        print(f"Saved gate labels to {self.label_cache_path}")

    def get_structure_feature_text(self, node_id, is_user, adj):
        """获取节点的结构特征文本表示
        Args:
            node_id: 节点ID
            is_user: 是否为用户节点
            adj: 稀疏邻接矩阵
        Returns:
            结构特征的文本描述
        """
        if is_user:
            # 获取用户的物品交互
            row_idx = node_id
            # 获取该行的非零元素列索引
            indices = adj[row_idx].coalesce().indices()[0].cpu().numpy()
            # 只选择物品索引 (排除用户索引)
            item_indices = [idx for idx in indices if idx >= self.user_num]
            # 转换为物品ID
            item_ids = [idx - self.user_num for idx in item_indices]

            # 格式化为文本
            if len(item_ids) > 0:
                # 最多显示10个物品ID
                top_items = item_ids[:10]
                return f"User has interacted with {len(item_ids)} items, including: {', '.join(map(str, top_items))}"
            else:
                return "User has no interactions"
        else:
            # 获取物品的用户交互
            col_idx = self.user_num + node_id
            # 获取该列的非零元素行索引
            indices = adj.transpose(0, 1)[col_idx].coalesce().indices()[1].cpu().numpy()
            # 只选择用户索引
            user_indices = [idx for idx in indices if idx < self.user_num]

            # 格式化为文本
            if len(user_indices) > 0:
                # 最多显示10个用户ID
                top_users = user_indices[:10]
                return f"Item has been interacted by {len(user_indices)} users, including: {', '.join(map(str, top_users))}"
            else:
                return "Item has no interactions"

    def get_text_feature_description(self, node_id, is_user):
        """获取节点的文本特征描述"""
        if is_user:
            profile_data = self.user_profiles.get(str(node_id), {})
            profile = profile_data.get("profile", "No profile available")
            return f"User profile: {profile}"
        else:
            profile_data = self.item_profiles.get(str(node_id), {})
            profile = profile_data.get("profile", "No profile available")
            return f"Item profile: {profile}"

    def call_llm_api(self, prompt):
        """调用LLM API获取响应"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        if self.api_type == "claude":
            data = {
                "model": "claude-3-7-sonnet-20240229",
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        else:  # OpenAI
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000
            }

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()

            if self.api_type == "claude":
                return response.json()["content"][0]["text"]
            else:  # OpenAI
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API call failed: {e}")
            return None

    def get_llm_gate_label(self, node_id, is_user, adj, batch_size=32):
        """获取节点的LLM建议的门控标签权重
        Args:
            node_id: 节点ID
            is_user: 是否为用户节点
            adj: 邻接矩阵
        Returns:
            一个0-1之间的值，表示偏向协同信息的程度
        """
        # 获取文本和结构特征
        text_feature = self.get_text_feature_description(node_id, is_user)
        structure_feature = self.get_structure_feature_text(node_id, is_user, adj)

        # 构建提示
        node_type = "User" if is_user else "Item"
        prompt = f"""
        I'm working on a recommender system that combines two types of information:
        1. Collaborative information (from user-item interactions)
        2. Semantic information (from text descriptions)

        For the following {node_type.lower()}, I need to determine how to weight these two sources.

        SEMANTIC INFORMATION:
        {text_feature}

        COLLABORATIVE INFORMATION:
        {structure_feature}

        Based on the quality and informativeness of these two information sources, please return a SINGLE NUMBER between 0 and 1 that represents how much weight should be given to the collaborative information (structure).
        - 0 means rely completely on semantic information
        - 1 means rely completely on collaborative information
        - Values between 0 and 1 represent the mixing ratio

        Please consider:
        - How informative the collaborative pattern is (number and quality of interactions)
        - How specific and relevant the semantic description is

        Return ONLY a single number (e.g., "0.476"、"0.621"、"0.559") without any explanation.
        """

        # 调用API
        response = self.call_llm_api(prompt)
        print(response)
        if not response:
            return 0.5  # 默认平衡值

        # 解析响应
        try:
            # 尝试提取数字
            value = float(response.strip())
            # 确保值在0-1范围内
            value = max(0.0, min(1.0, value))
            return value
        except:
            print(f"Failed to parse LLM response: {response}")
            return 0.5

    def batch_process_nodes(self, adj, batch_size=10, max_nodes=30000, save_interval=100):
        """批量处理节点获取LLM建议的门控权重"""
        # 处理的节点数量上限
        total_nodes = min(self.user_num + self.item_num, max_nodes)

        for i in tqdm(range(0, total_nodes, batch_size)):
            batch_end = min(i + batch_size, total_nodes)

            for node_id in range(i, batch_end):
                is_user = node_id < self.user_num
                real_id = node_id if is_user else node_id - self.user_num

                # 获取LLM建议的权重
                weight = self.get_llm_gate_label(real_id, is_user, adj)
                # weight = random.random()

                # 更新标签
                self.gate_labels[node_id] = t.ones(self.embedding_size, device=self.gate_labels.device) * weight

                # 防止API限速
                time.sleep(1)

            # 定期保存
            if (i // batch_size) % save_interval == 0:
                self.save_labels()

        # 最终保存
        self.save_labels()
        return self.gate_labels


class AutoCF_eg_crv(BaseModel):
    def __init__(self, data_handler):
        super(AutoCF_eg_crv, self).__init__(data_handler)
        # 原始协同图参数
        self.adj = data_handler.torch_adj
        self.all_one_adj = self.make_all_one_adj()
        self.semantic_adj = data_handler.semantic_adj  # 新增语义图
        self.keep_rate = configs['model']['keep_rate']
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']

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

        # 新增：LLM指导的门控机制
        self.llm_guide = LLMGuideManager(self.user_num, self.item_num, self.embedding_size)
        self.gate_supervision_weight = self.hyper_config['gate_supervision_weight']

        # 启动时预处理一批节点的门控标签
        self.preprocess_gate_labels = configs['model']['preprocess_gate_labels']
        if self.preprocess_gate_labels:
            print("Pre-processing gate labels with LLM...")
            self.llm_guide.batch_process_nodes(self.adj)

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

        return fused_embeds[:self.user_num], fused_embeds[self.user_num:], gate

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
        self.is_training = True
        # 双通道前向传播
        user_co, item_co, gate_co = self.forward(encoder_adj, decoder_adj)
        user_sem, item_sem, gate_sem = self.forward(encoder_adj, decoder_adj)  # 使用相同参数获取语义通道

        # 对比损失计算
        ancs, poss, negs = batch_data
        anc_co, pos_co = user_co[ancs], item_co[poss]
        anc_sem, pos_sem = user_sem[ancs], item_sem[poss]
        user_contrast = cal_infonce_loss(anc_co, anc_sem, user_sem, self.contrast_temp)
        item_contrast = cal_infonce_loss(pos_co, pos_sem, item_sem, self.contrast_temp)
        contrast_loss = (user_contrast + item_contrast) * self.contrast_weight

        # AutoCF原有损失
        rec_loss = 0.3 * (-t.sum(anc_co * pos_co, dim=-1)).mean()
        reg_loss = reg_params(self) * self.reg_weight
        cl_loss = (self.contrast(ancs, user_co) + self.contrast(poss, item_co)) * self.ssl_reg

        # 知识蒸馏损失
        usrprf = self.semantic_mlp(self.usrprf_embeds)
        itmprf = self.semantic_mlp(self.itmprf_embeds)
        kd_loss = cal_infonce_loss(anc_co, usrprf[ancs], usrprf, self.kd_temperature) + \
                  cal_infonce_loss(pos_co, itmprf[poss], itmprf, self.kd_temperature)
        kd_loss = kd_loss / (anc_co.shape[0] * 2) * self.kd_weight

        # 门控监督损失 - 新增
        gate_labels = self.llm_guide.gate_labels
        # 提取当前批次涉及的用户和物品的门控标签
        anc_gate_labels = gate_labels[ancs]
        pos_gate_labels = gate_labels[self.user_num + poss]
        neg_gate_labels = gate_labels[self.user_num + negs]

        # 提取对应的门控预测
        anc_gate_pred = gate_co[ancs]
        pos_gate_pred = gate_co[self.user_num + poss]
        neg_gate_pred = gate_co[self.user_num + negs]

        # 计算MSE损失
        gate_loss = F.mse_loss(anc_gate_pred, anc_gate_labels) + \
                    F.mse_loss(pos_gate_pred, pos_gate_labels) + \
                    F.mse_loss(neg_gate_pred, neg_gate_labels)
        gate_loss /= 3.0  # 平均三个部分的损失
        gate_loss *= self.gate_supervision_weight  # 应用权重系数

        # 总损失
        total_loss = rec_loss + reg_loss + cl_loss + kd_loss + contrast_loss + gate_loss
        losses = {
            'rec_loss': rec_loss,
            'reg_loss': reg_loss,
            'cl_loss': cl_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss,
            'gate_loss': gate_loss  # 新增门控监督损失记录
        }
        return total_loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds, item_embeds, _ = self.forward(self.adj, self.adj)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T

        # 屏蔽训练交互
        if train_mask is not None:
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
