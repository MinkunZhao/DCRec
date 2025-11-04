import random
import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop, NodeDrop
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


class SGL_eg_crv(BaseModel):
    def __init__(self, data_handler):
        super(SGL_eg_crv, self).__init__(data_handler)
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

        # 新增：LLM指导的门控机制
        self.llm_guide = LLMGuideManager(self.user_num, self.item_num, self.embedding_size)
        self.gate_supervision_weight = self.hyper_config['gate_supervision_weight']

        # 启动时预处理一批节点的门控标签
        # 在实际部署时可能需要提前运行或离线处理
        self.preprocess_gate_labels = configs['model']['preprocess_gate_labels']
        if self.preprocess_gate_labels:
            print("Pre-processing gate labels with LLM...")
            self.llm_guide.batch_process_nodes(self.adj)

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

    '''def _propagate_sem(self, embeds):
            edge_index = self.semantic_adj.indices()

            # 原始代码
            # edge_weight = self.semantic_adj.values()

            # 修改为多维边特征
            edge_values = self.semantic_adj.values()

            # 对于edge_dim=2的情况
            edge_attr = t.stack([edge_values, edge_values], dim=1)

            # 对于edge_dim=3的情况
            # edge_attr = t.stack([edge_values, edge_values, edge_values], dim=1)

            # edge_attr = t.stack([edge_values, edge_values, edge_values, edge_values], dim=1)

            # edge_attr = t.stack([edge_values, edge_values, edge_values, edge_values, edge_values], dim=1)

            for gat in self.gat_layers:
                embeds = gat(embeds, edge_index, edge_attr=edge_attr)
                embeds = F.leaky_relu(embeds)
            return embeds'''

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

        return fused_embeds[:self.user_num], fused_embeds[self.user_num:], gate

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        # 生成两个增强视图
        user_embeds1, item_embeds1, gate1 = self.forward(self.keep_rate)
        user_embeds2, item_embeds2, gate2 = self.forward(self.keep_rate)
        # 完整视图（用于BPR和KD）
        user_embeds3, item_embeds3, gate3 = self.forward(1.0)

        # 语义视图（用于对比）
        user_sem_embeds, item_sem_embeds, gate_sem = self.forward(1.0)

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

        # 新增：门控监督损失
        anc, pos, neg = batch_data
        gate_labels = self.llm_guide.gate_labels

        # 提取当前批次涉及的用户和物品的门控标签
        anc_gate_labels = gate_labels[anc]
        pos_gate_labels = gate_labels[self.user_num + pos]
        neg_gate_labels = gate_labels[self.user_num + neg]

        # 提取对应的门控预测
        anc_gate_pred = gate3[anc]
        pos_gate_pred = gate3[self.user_num + pos]
        neg_gate_pred = gate3[self.user_num + neg]

        # 计算MSE损失
        gate_loss = F.mse_loss(anc_gate_pred, anc_gate_labels) + \
                    F.mse_loss(pos_gate_pred, pos_gate_labels) + \
                    F.mse_loss(neg_gate_pred, neg_gate_labels)
        gate_loss /= 3.0  # 平均三个部分的损失
        gate_loss *= self.gate_supervision_weight  # 应用权重系数

        total_loss = bpr_loss + reg_loss + cl_loss + kd_loss + contrast_loss + gate_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'cl_loss': cl_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss,
            'gate_loss': gate_loss  # 门控监督损失记录
        }
        return total_loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds, item_embeds, _ = self.forward(1.0)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        if train_mask is not None:
            full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
