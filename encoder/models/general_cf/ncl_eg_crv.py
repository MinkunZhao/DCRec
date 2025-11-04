import random
import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.aug_utils import KMeansClustering
from encoder.models.loss_utils import cal_bpr_loss, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop
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


class NCL_eg_crv(BaseModel):
    def __init__(self, data_handler):
        super(NCL_eg_crv, self).__init__(data_handler)
        # 原始协同图参数
        self.adj = data_handler.torch_adj
        self.semantic_adj = data_handler.semantic_adj
        self.keep_rate = configs['model']['keep_rate']
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # 维度参数
        self.user_num = configs['data']['user_num']
        self.item_num = configs['data']['item_num']
        self.embedding_size = configs['model']['embedding_size']

        # NCL特有参数
        self.proto_weight = self.hyper_config['proto_weight']
        self.struct_weight = self.hyper_config['struct_weight']
        self.temperature = self.hyper_config['temperature']
        self.layer_num = self.hyper_config['layer_num']
        self.high_order = self.hyper_config['high_order']
        self.reg_weight = self.hyper_config['reg_weight']

        # 新增: 对比损失相关参数
        self.contrast_weight = self.hyper_config['contrastive_weight']
        self.contrast_temp = self.hyper_config['contrast_temp']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']

        # 聚类参数
        self.kmeans = KMeansClustering(
            cluster_num=self.hyper_config['cluster_num'],
            embedding_size=self.embedding_size
        )

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
        # 初始化权重
        for gat in self.gat_layers:
            # 初始化注意力参数
            init(gat.att_src)
            init(gat.att_dst)
            # 初始化线性变换参数
            if hasattr(gat, 'lin'):
                init(gat.lin.weight)
            elif hasattr(gat, 'lin_src') and gat.lin_src is not None:
                init(gat.lin_src.weight)
                init(gat.lin_dst.weight)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def _propagate_co(self, adj, embeds):
        """协同通道的多层图卷积传播"""
        embeds_list = [embeds]
        for _ in range(self.layer_num):
            embeds = t.spmm(adj, embeds_list[-1])
            embeds_list.append(embeds)
        return sum(embeds_list), embeds_list  # 返回各层的加和和全部的层表示

    def _propagate_sem(self, embeds):
        """语义通道的GAT传播"""
        edge_index = self.semantic_adj.indices()
        edge_weight = self.semantic_adj.values()
        for gat in self.gat_layers:
            embeds = gat(embeds, edge_index, edge_attr=edge_weight)
            embeds = F.leaky_relu(embeds)
        return embeds

    def _cluster(self):
        """使用K-means对用户和物品嵌入进行聚类"""
        self.user_centroids, self.user2cluster, _ = self.kmeans(self.user_embeds.detach())
        self.item_centroids, self.item2cluster, _ = self.kmeans(self.item_embeds.detach())

    def forward(self, adj=None, keep_rate=1.0, return_layers=False):
        # 处理输入参数
        if adj is None:
            adj = self.adj

        # 检查是否需要新的前向传播
        if not self.is_training and self.final_embeds is not None and not return_layers:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:], None

        # 协同通道
        co_embeds = t.cat([self.user_embeds, self.item_embeds], 0)
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        co_embeds_sum, co_embeds_list = self._propagate_co(adj, co_embeds)

        # 语义通道
        sem_embeds = t.cat([
            self.semantic_mlp(self.usrprf_embeds),
            self.semantic_mlp(self.itmprf_embeds)
        ], 0)
        sem_embeds = self._propagate_sem(sem_embeds)

        # 门控融合
        gate_input = t.cat([co_embeds_sum, sem_embeds], dim=1)
        gate = self.sigmoid(self.gate(gate_input))
        fused_embeds = gate * co_embeds_sum + (1 - gate) * sem_embeds

        # 保存结果
        self.final_embeds = fused_embeds
        self.final_embeds_list = co_embeds_list

        if return_layers:
            return fused_embeds[:self.user_num], fused_embeds[self.user_num:], co_embeds_list, gate
        else:
            return fused_embeds[:self.user_num], fused_embeds[self.user_num:], gate

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def _cal_struct_loss(self, context_embeds, ego_embeds, ancs, poss):
        """NCL原始的结构对比损失"""
        user_embeds1, item_embeds1 = context_embeds[:self.user_num], context_embeds[self.user_num:]
        user_embeds2, item_embeds2 = ego_embeds[:self.user_num], ego_embeds[self.user_num:]
        pck_user_embeds1 = user_embeds1[ancs]
        pck_user_embeds2 = user_embeds2[ancs]
        pck_item_embeds1 = item_embeds1[poss]
        pck_item_embeds2 = item_embeds2[poss]
        return (cal_infonce_loss(pck_user_embeds1, pck_user_embeds2, user_embeds2, self.temperature) +
                cal_infonce_loss(pck_item_embeds1, pck_item_embeds2, item_embeds2, self.temperature)) / \
            pck_user_embeds1.shape[0]

    def _cal_proto_loss(self, ego_embeds, ancs, poss):
        """NCL原始的原型对比损失"""
        user_embeds, item_embeds = ego_embeds[:self.user_num], ego_embeds[self.user_num:]
        user_clusters = self.user2cluster[ancs]
        item_clusters = self.item2cluster[poss]
        pck_user_embeds1 = user_embeds[ancs]
        pck_user_embeds2 = self.user_centroids[user_clusters]
        pck_item_embeds1 = item_embeds[poss]
        pck_item_embeds2 = self.item_centroids[item_clusters]
        return (cal_infonce_loss(pck_user_embeds1, pck_user_embeds2, self.user_centroids, self.temperature) +
                cal_infonce_loss(pck_item_embeds1, pck_item_embeds2, self.item_centroids, self.temperature)) / \
            pck_user_embeds1.shape[0]

    def cal_loss(self, batch_data):
        self.is_training = True

        # 解包批次数据
        ancs, poss, negs = batch_data[:3]

        # 运行聚类（如果需要）
        if not hasattr(self, 'user2cluster'):
            self._cluster()

        # 前向传播 - 获取融合嵌入和协同嵌入层
        user_embeds, item_embeds, co_embeds_list, gate = self.forward(self.adj, self.keep_rate, return_layers=True)

        # 获取ego和context嵌入（用于NCL原始的结构对比损失）
        ego_embeds = co_embeds_list[0]
        context_embeds = co_embeds_list[self.high_order]

        # 计算结构和原型对比损失（NCL原始损失）
        struct_loss = self._cal_struct_loss(context_embeds, ego_embeds, ancs, poss) * self.struct_weight
        proto_loss = self._cal_proto_loss(ego_embeds, ancs, poss) * self.proto_weight

        # 按相同方式获取融合嵌入，进行协同图渠道前向传播
        user_embeds_struct, item_embeds_struct, gate_struct = self.forward(self.adj, self.keep_rate)
        # 再进行一次传播获取语义嵌入
        user_embeds_sem, item_embeds_sem, gate_sem = self.forward(self.adj, 1.0)

        # 获取当前批次的嵌入
        anc_embeds_struct, pos_embeds_struct, _ = self._pick_embeds(user_embeds_struct, item_embeds_struct,
                                                                    batch_data[:3])
        anc_embeds_sem, pos_embeds_sem, _ = self._pick_embeds(user_embeds_sem, item_embeds_sem, batch_data[:3])

        # 计算对比损失（结构和语义之间的对比）
        user_contrast_loss = cal_infonce_loss(anc_embeds_struct, anc_embeds_sem, anc_embeds_sem, self.contrast_temp)
        item_contrast_loss = cal_infonce_loss(pos_embeds_struct, pos_embeds_sem, pos_embeds_sem, self.contrast_temp)
        contrast_loss = (user_contrast_loss + item_contrast_loss) * self.contrast_weight

        # 获取最终预测用的嵌入
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data[:3])

        # BPR损失
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]

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
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data[:3])
        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        # 门控监督损失
        gate_labels = self.llm_guide.gate_labels
        # 提取当前批次涉及的用户和物品的门控标签
        anc_gate_labels = gate_labels[ancs]
        pos_gate_labels = gate_labels[self.user_num + poss]
        neg_gate_labels = gate_labels[self.user_num + negs]

        # 提取对应的门控预测
        anc_gate_pred = gate[ancs]
        pos_gate_pred = gate[self.user_num + poss]
        neg_gate_pred = gate[self.user_num + negs]

        # 计算MSE损失
        gate_loss = F.mse_loss(anc_gate_pred, anc_gate_labels) + \
                    F.mse_loss(pos_gate_pred, pos_gate_labels) + \
                    F.mse_loss(neg_gate_pred, neg_gate_labels)
        gate_loss /= 3.0  # 平均三个部分的损失
        gate_loss *= self.gate_supervision_weight  # 应用权重系数

        # 总损失
        loss = bpr_loss + reg_loss + kd_loss + contrast_loss + struct_loss + proto_loss + gate_loss

        # 记录各部分损失
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss,
            'struct_loss': struct_loss,
            'proto_loss': proto_loss,
            'gate_loss': gate_loss
        }
        return loss, losses

    def full_predict(self, batch_data):
        # 切换到预测模式
        self.is_training = False

        # 获取融合嵌入
        user_embeds, item_embeds, _ = self.forward(self.adj, 1.0)

        # 解包批次数据
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()

        # 获取目标用户嵌入
        pck_user_embeds = user_embeds[pck_users]

        # 计算全量预测分数
        full_preds = pck_user_embeds @ item_embeds.T

        # 屏蔽训练交互
        full_preds = self._mask_predict(full_preds, train_mask)

        return full_preds
