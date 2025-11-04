import random
import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
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
            transposed_adj = adj.transpose(0, 1)
            indices = transposed_adj[col_idx].coalesce().indices()

            if indices.shape[0] > 0:  # 确保有索引
                # 根据稀疏矩阵的结构选择正确的维度
                if indices.shape[0] == 1:  # 只有一维
                    user_indices = indices[0].cpu().numpy()
                else:  # 有两维
                    user_indices = indices[1].cpu().numpy()

                # 只选择用户索引
                user_indices = [idx for idx in user_indices if idx < self.user_num]

                # 格式化为文本
                if len(user_indices) > 0:
                    # 最多显示10个用户ID
                    top_users = user_indices[:10]
                    return f"Item has been interacted by {len(user_indices)} users, including: {', '.join(map(str, top_users))}"
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
                "max_tokens": 10000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        else:  # OpenAI
            data = {
                "model": "gpt-4.1-nano",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 10000
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

        Based on the quality and informativeness of these two information sources, please return a SINGLE NUMBER between 0.400 and 0.600 that represents how much weight should be given to the collaborative information (structure).
        - 0.400 means rely completely on semantic information
        - 0.600 means rely completely on collaborative information
        - Values between 0.400 and 0.600 represent the mixing ratio

        Please consider:
        - How informative the collaborative pattern is (number and quality of interactions)
        - How specific and relevant the semantic description is

        Return ONLY a single number (e.g., "0.476"、"0.521"、"0.539") without any explanation.
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


class LightGCN_eg_crv(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_eg_crv, self).__init__(data_handler)
        # 原始协同图参数
        self.adj = data_handler.torch_adj
        self.semantic_adj = data_handler.semantic_adj
        self.keep_rate = configs['model']['keep_rate']
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
        # 修改后的权重初始化
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

    def _propagate_co(self, adj, embeds):
        embeds_list = [embeds]
        for _ in range(configs['model']['layer_num']):
            embeds = t.spmm(adj, embeds_list[-1])
            embeds_list.append(embeds)
        return sum(embeds_list)  # 各层求和

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
        # 应用edge dropping
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

        return fused_embeds[:self.user_num], fused_embeds[self.user_num:], gate

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        # 结构图前向传播
        user_embeds_struct, item_embeds_struct, gate_struct = self.forward(self.keep_rate)
        # 语义图前向传播
        user_embeds_sem, item_embeds_sem, gate_sem = self.forward(1.0)

        # 提取当前batch的嵌入
        anc_embeds_struct, pos_embeds_struct, _ = self._pick_embeds(user_embeds_struct, item_embeds_struct, batch_data)
        anc_embeds_sem, pos_embeds_sem, _ = self._pick_embeds(user_embeds_sem, item_embeds_sem, batch_data)

        # 计算对比损失（用户和正物品）
        user_contrast_loss = cal_infonce_loss(anc_embeds_struct, anc_embeds_sem, anc_embeds_sem, self.contrast_temp)
        item_contrast_loss = cal_infonce_loss(pos_embeds_struct, pos_embeds_sem, pos_embeds_sem, self.contrast_temp)
        contrast_loss = (user_contrast_loss + item_contrast_loss) * self.contrast_weight

        user_embeds, item_embeds, gate = self.forward(self.keep_rate)
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

        # 门控监督损失
        gate_labels = self.llm_guide.gate_labels
        # print(gate_labels)
        # 提取当前批次涉及的用户和物品的门控标签
        anc_gate_labels = gate_labels[anc]
        pos_gate_labels = gate_labels[self.user_num + pos]
        neg_gate_labels = gate_labels[self.user_num + neg]

        # 提取对应的门控预测
        anc_gate_pred = gate[anc]
        pos_gate_pred = gate[self.user_num + pos]
        neg_gate_pred = gate[self.user_num + neg]

        # 计算MSE损失
        gate_loss = F.mse_loss(anc_gate_pred, anc_gate_labels) + \
                    F.mse_loss(pos_gate_pred, pos_gate_labels) + \
                    F.mse_loss(neg_gate_pred, neg_gate_labels)
        gate_loss /= 3.0  # 平均三个部分的损失
        gate_loss *= self.gate_supervision_weight  # 应用权重系数

        # 总损失
        loss = bpr_loss + reg_loss + kd_loss + contrast_loss + gate_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss,
            'gate_loss': gate_loss  # 门控监督损失记录
        }
        return loss, losses

    def full_predict(self, batch_data):
        # 切换到预测模式
        self.is_training = False
        user_embeds, item_embeds, _ = self.forward(1.0)

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




'''import torch as t
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from encoder.config.configurator import configs
from encoder.models.loss_utils import cal_bpr_loss, cal_infonce_loss
from encoder.models.base_model import BaseModel
from encoder.models.model_utils import SpAdjEdgeDrop
import json
import numpy as np
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

init = nn.init.xavier_uniform_


class LightGCN_eg_crv(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_eg_crv, self).__init__(data_handler)
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

        # 新增: LLM监督参数
        self.gate_supervision_weight = self.hyper_config['gate_supervision_weight']
        self.llm_model = configs['model']['llm_model']
        self.api_key = configs['model']['api_key']
        self.api_batch_size = configs['model']['api_batch_size']
        self.api_calls_per_node = configs['model']['api_calls_per_node']

        # 加载用户和商品画像
        self.usr_prf = self._load_json('../data/{}/usr_prf.json'.format(configs['data']['name']))
        self.itm_prf = self._load_json('../data/{}/itm_prf.json'.format(configs['data']['name']))

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

        # 新增: 预先计算的LLM监督标签
        self.llm_gate_labels = None
        self.node_indices_map = {}  # 用于保存采样的节点对应的索引

        # 预先获取LLM监督标签
        if configs.get('precompute_llm_labels', True):
            self._precompute_llm_labels()

        self._init_weights()

    def _load_json(self, file_path):
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}

    def _init_weights(self):
        # 修改后的权重初始化
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

    def _get_node_neighbors(self, node_id, is_user=True):
        """获取节点的邻居信息"""
        if is_user:
            start_idx = node_id
            end_idx = slice(self.user_num, self.user_num + self.item_num)
        else:
            start_idx = node_id + self.user_num
            end_idx = slice(0, self.user_num)

        # 从协同图中获取邻居
        neighbors = []
        adj = self.adj.to_dense()
        if is_user:
            neighbors_mask = adj[start_idx, end_idx] > 0
            neighbors = t.where(neighbors_mask)[0].cpu().numpy().tolist()
        else:
            neighbors_mask = adj[start_idx, end_idx] > 0
            neighbors = t.where(neighbors_mask)[0].cpu().numpy().tolist()

        return neighbors[:5]  # 最多返回5个邻居，避免文本过长

    def _construct_text_description(self, node_id, is_user=True):
        """构建节点的文本描述"""
        # 获取语义特征描述
        if is_user:
            profile_data = self.usr_prf.get(str(node_id), {})
            profile_text = profile_data.get("profile", "无可用的用户画像")
            semantic_text = f"用户{node_id}的语义特征: {profile_text}"
        else:
            profile_data = self.itm_prf.get(str(node_id), {})
            profile_text = profile_data.get("profile", "无可用的商品画像")
            semantic_text = f"商品{node_id}的语义特征: {profile_text}"

        # 获取结构特征描述
        neighbors = self._get_node_neighbors(node_id, is_user)
        if is_user:
            structure_parts = []
            for item_idx in neighbors:
                item_id = item_idx - self.user_num
                item_profile = self.itm_prf.get(str(item_id), {}).get("profile", "无描述")
                structure_parts.append(f"商品{item_id}: {item_profile[:100]}...")

            if not structure_parts:
                structure_text = f"用户{node_id}的协同关系特征: 该用户没有交互的商品"
            else:
                structure_text = f"用户{node_id}的协同关系特征 (交互过的商品): " + "; ".join(structure_parts)
        else:
            structure_parts = []
            for user_idx in neighbors:
                user_profile = self.usr_prf.get(str(user_idx), {}).get("profile", "无描述")
                structure_parts.append(f"用户{user_idx}: {user_profile[:100]}...")

            if not structure_parts:
                structure_text = f"商品{node_id}的协同关系特征: 该商品没有被任何用户交互过"
            else:
                structure_text = f"商品{node_id}的协同关系特征 (交互过该商品的用户): " + "; ".join(structure_parts)

        return semantic_text, structure_text

    def _call_llm_api(self, prompt):
        """调用LLM API获取门控权重建议"""
        if self.llm_model == 'claude':
            return self._call_claude_api(prompt)
        else:
            return self._call_gpt_api(prompt)

    def _call_claude_api(self, prompt):
        """调用Claude API"""
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json"
        }

        data = {
            "model": "claude-3-5-sonnet-20240620",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.2
        }

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                result = response.json()
                content = result["content"][0]["text"]
                # 从回答中提取0-1之间的数值
                import re
                matches = re.findall(r"([0-9]*\.?[0-9]+)", content)
                if matches:
                    for match in matches:
                        value = float(match)
                        if 0 <= value <= 1:
                            return value
                return 0.5  # 默认值
            else:
                print(f"API调用失败: {response.status_code}, {response.text}")
                return 0.5
        except Exception as e:
            print(f"API调用异常: {e}")
            return 0.5

    def _call_gpt_api(self, prompt):
        """调用GPT-4o-mini API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "你是一个推荐系统专家助手，需要帮助判断哪种特征更重要。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.2
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                # 从回答中提取0-1之间的数值
                import re
                matches = re.findall(r"([0-9]*\.?[0-9]+)", content)
                if matches:
                    for match in matches:
                        value = float(match)
                        if 0 <= value <= 1:
                            return value
                return 0.5  # 默认值
            else:
                print(f"API调用失败: {response.status_code}, {response.text}")
                return 0.5
        except Exception as e:
            print(f"API调用异常: {e}")
            return 0.5

    def _generate_llm_prompt(self, node_id, is_user=True):
        """生成LLM提示，询问门控权重"""
        semantic_text, structure_text = self._construct_text_description(node_id, is_user)

        prompt = f"""我有一个推荐系统，对于{'用户' if is_user else '商品'} {node_id}，有两种特征：

1. 语义特征 (Text Feature): {semantic_text}

2. 协同关系结构特征 (Structure Feature): {structure_text}

基于上述两种特征，请给出一个0到1之间的小数值(精确到0.001)，表示对于这个{'用户' if is_user else '商品'}，推荐系统应该更多地依赖哪种特征。
0表示完全依赖协同关系结构特征，1表示完全依赖语义特征，0.5表示两者等重。

请仅返回一个0到1之间的数值，不要解释过程。"""

        return prompt

    def _precompute_llm_labels(self):
        """预先计算LLM标签，避免在训练过程中重复API调用"""
        print("预计算LLM门控标签...")

        # 采样一部分节点，而不是全部节点
        sample_ratio = configs.get('llm_sample_ratio', 0.1)
        user_sample_size = max(int(self.user_num * sample_ratio), 100)
        item_sample_size = max(int(self.item_num * sample_ratio), 100)

        # 随机采样节点
        sampled_users = np.random.choice(self.user_num, min(user_sample_size, self.user_num), replace=False)
        sampled_items = np.random.choice(self.item_num, min(item_sample_size, self.item_num), replace=False)

        # 初始化标签存储
        self.llm_gate_labels = t.zeros(self.user_num + self.item_num).cuda()
        default_value = 0.5  # 默认值
        self.llm_gate_labels.fill_(default_value)

        # 记录采样的节点
        for i, user_id in enumerate(sampled_users):
            self.node_indices_map[('user', user_id)] = i

        for i, item_id in enumerate(sampled_items):
            self.node_indices_map[('item', item_id)] = i + len(sampled_users)

        # 如果存在缓存文件，则直接加载
        cache_file = 'llm_gate_labels_{}.pt'.format(configs['data']['name'])
        if os.path.exists(cache_file):
            try:
                saved_data = t.load(cache_file)
                self.llm_gate_labels = saved_data['labels'].cuda()
                self.node_indices_map = saved_data['indices_map']
                print(f"从缓存加载了LLM门控标签: {cache_file}")
                return
            except Exception as e:
                print(f"加载缓存标签失败: {e}，将重新计算")

        # 定义批量处理函数
        def process_node_batch(node_batch):
            results = []
            for node_type, node_id in node_batch:
                is_user = node_type == 'user'
                prompt = self._generate_llm_prompt(node_id, is_user)

                # 多次调用API，取平均值
                values = []
                for _ in range(self.api_calls_per_node):
                    # value = self._call_llm_api(prompt)
                    value = 0.8
                    print(value)
                    values.append(value)

                avg_value = sum(values) / len(values)
                results.append(((node_type, node_id), avg_value))
            return results

        # 准备所有节点
        all_nodes = [(('user', user_id), i) for i, user_id in enumerate(sampled_users)]
        all_nodes.extend([(('item', item_id), i + len(sampled_users)) for i, item_id in enumerate(sampled_items)])

        # 分批处理
        batches = [all_nodes[i:i + self.api_batch_size] for i in range(0, len(all_nodes), self.api_batch_size)]

        with ThreadPoolExecutor(max_workers=4) as executor:
            for batch in tqdm(batches, desc="获取LLM标签"):
                nodes_batch = [node[0] for node in batch]
                batch_results = process_node_batch(nodes_batch)

                # 更新标签
                for (node_type, node_id), value in batch_results:
                    idx = self.node_indices_map.get((node_type, node_id))
                    if idx is not None:
                        if node_type == 'user':
                            self.llm_gate_labels[node_id] = value
                        else:
                            self.llm_gate_labels[self.user_num + node_id] = value

        # 保存标签到文件
        if cache_file:
            t.save({
                'labels': self.llm_gate_labels.cpu(),
                'indices_map': self.node_indices_map
            }, cache_file)
            print(f"LLM标签已保存到: {cache_file}")

    def _get_llm_gate_label(self, node_indices):
        """获取LLM预测的门控标签"""
        return self.llm_gate_labels[node_indices]

    def _propagate_co(self, adj, embeds):
        """协同通道传播"""
        embeds_list = [embeds]
        for _ in range(configs['model']['layer_num']):
            embeds = t.spmm(adj, embeds_list[-1])
            embeds_list.append(embeds)
        return sum(embeds_list)  # 各层求和

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
        # 应用edge dropping
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

        return fused_embeds[:self.user_num], fused_embeds[self.user_num:], gate, co_embeds, sem_embeds

    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        # 结构图前向传播
        user_embeds_struct, item_embeds_struct, gates_struct, co_embeds_struct, sem_embeds_struct = self.forward(
            self.keep_rate)
        # 语义图前向传播
        user_embeds_sem, item_embeds_sem, gates_sem, co_embeds_sem, sem_embeds_sem = self.forward(1.0)

        # 提取当前batch的嵌入
        anc_embeds_struct, pos_embeds_struct, _ = self._pick_embeds(user_embeds_struct, item_embeds_struct, batch_data)
        anc_embeds_sem, pos_embeds_sem, _ = self._pick_embeds(user_embeds_sem, item_embeds_sem, batch_data)

        # 计算对比损失（用户和正物品）
        user_contrast_loss = cal_infonce_loss(anc_embeds_struct, anc_embeds_sem, anc_embeds_sem, self.contrast_temp)
        item_contrast_loss = cal_infonce_loss(pos_embeds_struct, pos_embeds_sem, pos_embeds_sem, self.contrast_temp)
        contrast_loss = (user_contrast_loss + item_contrast_loss) * self.contrast_weight

        user_embeds, item_embeds, gates, co_embeds, sem_embeds = self.forward(self.keep_rate)
        anc, pos, neg = batch_data

        # BPR损失
        anc_embeds = user_embeds[anc]
        pos_embeds = item_embeds[pos]
        neg_embeds = item_embeds[neg]
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds)

        # 获取当前batch用户和物品的LLM门控标签
        user_indices = anc.cpu().numpy()
        item_indices_pos = pos.cpu().numpy()
        item_indices_neg = neg.cpu().numpy()

        # 合并所有节点索引
        all_indices = t.cat([anc, pos + self.user_num, neg + self.user_num])
        # 获取对应的LLM标签
        llm_gate_labels = self._get_llm_gate_label(all_indices)

        # 分离用户和物品的门控值和标签
        user_gates = gates[:self.user_num][anc]
        pos_item_gates = gates[self.user_num:][pos]
        neg_item_gates = gates[self.user_num:][neg]

        # 确保shape匹配
        user_gates = user_gates.view(-1, self.embedding_size)
        pos_item_gates = pos_item_gates.view(-1, self.embedding_size)
        neg_item_gates = neg_item_gates.view(-1, self.embedding_size)

        # 计算门控监督损失
        user_llm_labels = llm_gate_labels[:len(anc)].view(-1, 1).expand_as(user_gates)
        pos_llm_labels = llm_gate_labels[len(anc):len(anc) + len(pos)].view(-1, 1).expand_as(pos_item_gates)
        neg_llm_labels = llm_gate_labels[len(anc) + len(pos):].view(-1, 1).expand_as(neg_item_gates)

        # 使用MSE损失计算门控监督损失
        gate_loss_user = F.mse_loss(user_gates, user_llm_labels)
        gate_loss_pos = F.mse_loss(pos_item_gates, pos_llm_labels)
        gate_loss_neg = F.mse_loss(neg_item_gates, neg_llm_labels)
        gate_supervision_loss = (gate_loss_user + gate_loss_pos + gate_loss_neg) / 3

        # 应用门控监督权重
        gate_supervision_loss *= self.gate_supervision_weight

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

        # 合并所有损失
        loss = bpr_loss + reg_loss + kd_loss + contrast_loss + gate_supervision_loss
        losses = {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'kd_loss': kd_loss,
            'contrast_loss': contrast_loss,
            'gate_loss': gate_supervision_loss  # 新增门控监督损失
        }
        return loss, losses

    def full_predict(self, batch_data):
        # 切换到预测模式
        self.is_training = False
        user_embeds, item_embeds, _, _, _ = self.forward(1.0)

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

        return full_preds'''
