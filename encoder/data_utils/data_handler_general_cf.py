import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from encoder.config.configurator import configs
from encoder.data_utils.datasets_general_cf import PairwiseTrnData, PairwiseWEpochFlagTrnData, AllRankTstData
import torch as t
import torch.utils.data as data


class DataHandlerGeneralCF:
    def __init__(self):
        self.sem_item_adj = None
        self.sem_user_adj = None
        if configs['data']['name'] == 'amazon':
            predir = '../data/amazon/'
        elif configs['data']['name'] == 'yelp':
            predir = '../data/yelp/'
        elif configs['data']['name'] == 'steam':
            predir = '../data/steam/'
        else:
            raise NotImplementedError
        self.trn_file = predir + 'trn_mat.pkl'
        self.val_file = predir + 'val_mat.pkl'
        self.tst_file = predir + 'tst_mat.pkl'

    def _build_semantic_graph(self):
        """Build semantic graph based on LLM features"""
        # pca = PCA(n_components=128)
        # usrprf = pca.fit_transform(configs['usrprf_embeds'])
        # itmprf = pca.fit_transform(configs['itmprf_embeds'])
        usrprf = configs['usrprf_embeds']  # (user_num, 1536)
        itmprf = configs['itmprf_embeds']  # (item_num, 1536)

        # 计算余弦相似度矩阵
        user_norm = normalize(usrprf, norm='l2', axis=1)
        item_norm = normalize(itmprf, norm='l2', axis=1)
        sim_matrix = user_norm @ item_norm.T  # (user_num, item_num)

        # 保留topk连接增强稀疏性
        k = configs['model']['sem_graph_topk']
        rows = []
        cols = []
        data = []
        for i in range(sim_matrix.shape[0]):
            topk_idx = np.argpartition(sim_matrix[i], -k)[-k:]
            rows.extend([i] * k)
            cols.extend(topk_idx)
            data.extend(sim_matrix[i][topk_idx])

        # 创建语义邻接矩阵
        # semantic_adj = coo_matrix((data, (rows, cols)),
        #                           shape=(self.user_num, self.item_num))

        # 构建为图结构（用户和商品节点）
        adj_size = len(usrprf) + len(itmprf)
        extended_rows = rows
        extended_cols = [col + len(usrprf) for col in cols]

        # 转换为稀疏张量
        idxs = t.from_numpy(np.vstack([extended_rows, extended_cols])).long().cuda()
        vals = t.from_numpy(np.array(data)).float().cuda()
        self.semantic_adj = t.sparse_coo_tensor(
            idxs, vals,
            (adj_size, adj_size)
        ).coalesce()

    '''def _build_semantic_graph(self):
        """Build semantic graph with bidirectional user-item edges"""
        usrprf = configs['usrprf_embeds']  # (user_num, 1536)
        itmprf = configs['itmprf_embeds']  # (item_num, 1536)

        # 计算余弦相似度矩阵
        user_norm = normalize(usrprf, norm='l2', axis=1)
        item_norm = normalize(itmprf, norm='l2', axis=1)
        sim_matrix = user_norm @ item_norm.T  # (user_num, item_num)

        # 保留topk连接增强稀疏性
        k = configs['model']['sem_graph_topk']
        rows = []
        cols = []
        data = []
        for i in range(sim_matrix.shape[0]):
            topk_idx = np.argpartition(sim_matrix[i], -k)[-k:]
            rows.extend([i] * k)
            cols.extend(topk_idx)
            data.extend(sim_matrix[i][topk_idx])

        # 构建双向图结构（用户和商品节点）
        adj_size = len(usrprf) + len(itmprf)

        # 用户->物品边
        ui_rows = rows
        ui_cols = [col + len(usrprf) for col in cols]

        # 物品->用户边（反向边）
        iu_rows = ui_cols  # 将之前的列作为新的行
        iu_cols = ui_rows  # 将之前的行作为新的列

        # 合并两个方向的边
        extended_rows = ui_rows + iu_rows
        extended_cols = ui_cols + iu_cols
        extended_data = data + data  # 复制相同的权重值

        # 转换为稀疏张量
        idxs = t.from_numpy(np.vstack([extended_rows, extended_cols])).long().cuda()
        vals = t.from_numpy(np.array(extended_data)).float().cuda()
        self.semantic_adj = t.sparse_coo_tensor(
            idxs, vals,
            (adj_size, adj_size)
        ).coalesce()'''

    '''def _build_semantic_graph(self):
        """Build semantic graph with user-item, user-user and item-item edges"""
        usrprf = configs['usrprf_embeds']  # (user_num, 1536)
        itmprf = configs['itmprf_embeds']  # (item_num, 1536)

        # 计算余弦相似度矩阵
        user_norm = normalize(usrprf, norm='l2', axis=1)
        item_norm = normalize(itmprf, norm='l2', axis=1)

        # User-Item相似度矩阵
        ui_sim_matrix = user_norm @ item_norm.T  # (user_num, item_num)

        # User-User相似度矩阵
        uu_sim_matrix = user_norm @ user_norm.T  # (user_num, user_num)

        # Item-Item相似度矩阵
        ii_sim_matrix = item_norm @ item_norm.T  # (item_num, item_num)

        # 保留user-item的topk连接
        k_ui = configs['model']['sem_graph_topk']
        ui_rows = []
        ui_cols = []
        ui_data = []
        for i in range(ui_sim_matrix.shape[0]):
            topk_idx = np.argpartition(ui_sim_matrix[i], -k_ui)[-k_ui:]
            ui_rows.extend([i] * k_ui)
            ui_cols.extend(topk_idx)
            ui_data.extend(ui_sim_matrix[i][topk_idx])

        # 保留user-user的topk连接
        k_uu = configs['model'].get('user_user_topk', k_ui // 2)  # 默认使用ui的一半
        uu_rows = []
        uu_cols = []
        uu_data = []
        for i in range(uu_sim_matrix.shape[0]):
            # 排除自连接
            sim_scores = uu_sim_matrix[i].copy()
            sim_scores[i] = -1  # 排除自己
            topk_idx = np.argpartition(sim_scores, -k_uu)[-k_uu:]
            uu_rows.extend([i] * k_uu)
            uu_cols.extend(topk_idx)
            uu_data.extend(uu_sim_matrix[i][topk_idx])

        # 保留item-item的topk连接
        k_ii = configs['model'].get('item_item_topk', k_ui // 2)  # 默认使用ui的一半
        ii_rows = []
        ii_cols = []
        ii_data = []
        for i in range(ii_sim_matrix.shape[0]):
            # 排除自连接
            sim_scores = ii_sim_matrix[i].copy()
            sim_scores[i] = -1  # 排除自己
            topk_idx = np.argpartition(sim_scores, -k_ii)[-k_ii:]
            ii_rows.extend([i] * k_ii)
            ii_cols.extend(topk_idx)
            ii_data.extend(ii_sim_matrix[i][topk_idx])

        # 构建图结构
        user_num = len(usrprf)
        item_num = len(itmprf)
        adj_size = user_num + item_num

        # 用户->物品边
        extended_ui_rows = ui_rows
        extended_ui_cols = [col + user_num for col in ui_cols]

        # 用户->用户边
        extended_uu_rows = uu_rows
        extended_uu_cols = uu_cols

        # 物品->物品边
        extended_ii_rows = [row + user_num for row in ii_rows]
        extended_ii_cols = [col + user_num for col in ii_cols]

        # 合并所有边
        extended_rows = extended_ui_rows + extended_uu_rows + extended_ii_rows
        extended_cols = extended_ui_cols + extended_uu_cols + extended_ii_cols
        extended_data = ui_data + uu_data + ii_data

        # 转换为稀疏张量
        idxs = t.from_numpy(np.vstack([extended_rows, extended_cols])).long().cuda()
        vals = t.from_numpy(np.array(extended_data)).float().cuda()
        self.semantic_adj = t.sparse_coo_tensor(
            idxs, vals,
            (adj_size, adj_size)
        ).coalesce()'''

    '''def _build_semantic_graph(self):
        """Build semantic graph using combined cosine similarity and TF-IDF weights"""
        usrprf = configs['usrprf_embeds']  # (user_num, 1536)
        itmprf = configs['itmprf_embeds']  # (item_num, 1536)

        # 1. 计算余弦相似度矩阵
        user_norm = normalize(usrprf, norm='l2', axis=1)
        item_norm = normalize(itmprf, norm='l2', axis=1)
        cosine_sim_matrix = user_norm @ item_norm.T  # (user_num, item_num)

        # 2. 计算TF-IDF权重
        # 将嵌入向量视为文档向量，计算其重要性
        # 这里用一种简化方式：计算每个维度的方差作为特征重要性
        user_tfidf = np.var(usrprf, axis=0)
        item_tfidf = np.var(itmprf, axis=0)

        # 标准化为0-1范围
        user_tfidf = (user_tfidf - user_tfidf.min()) / (user_tfidf.max() - user_tfidf.min() + 1e-10)
        item_tfidf = (item_tfidf - item_tfidf.min()) / (item_tfidf.max() - item_tfidf.min() + 1e-10)

        # 3. 计算用户和物品向量的TF-IDF加权平均值
        user_importance = usrprf * user_tfidf
        item_importance = itmprf * item_tfidf

        # 4. 计算TF-IDF相似度
        user_importance_norm = normalize(user_importance, norm='l2', axis=1)
        item_importance_norm = normalize(item_importance, norm='l2', axis=1)
        tfidf_sim_matrix = user_importance_norm @ item_importance_norm.T

        # 5. 综合两种相似度 (可以调整两个相似度的权重)
        alpha = 0  # 余弦相似度的权重
        combined_sim_matrix = alpha * cosine_sim_matrix + (1 - alpha) * tfidf_sim_matrix

        # 6. 保留topk连接增强稀疏性
        k = configs['model']['sem_graph_topk']
        rows = []
        cols = []
        data = []
        for i in range(combined_sim_matrix.shape[0]):
            topk_idx = np.argpartition(combined_sim_matrix[i], -k)[-k:]
            rows.extend([i] * k)
            cols.extend(topk_idx)
            data.extend(combined_sim_matrix[i][topk_idx])

        # 7. 构建为图结构（用户和商品节点）
        adj_size = len(usrprf) + len(itmprf)
        extended_rows = rows
        extended_cols = [col + len(usrprf) for col in cols]

        # 8. 转换为稀疏张量
        idxs = t.from_numpy(np.vstack([extended_rows, extended_cols])).long().cuda()
        vals = t.from_numpy(np.array(data)).float().cuda()
        self.semantic_adj = t.sparse_coo_tensor(
            idxs, vals,
            (adj_size, adj_size)
        ).coalesce()'''

    def _load_one_mat(self, file):
        """Load one single adjacent matrix from file

        Args:
            file (string): path of the file to load

        Returns:
            scipy.sparse.coo_matrix: the loaded adjacent matrix
        """
        with open(file, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)
        return mat

    def _normalize_adj(self, mat):
        """Laplacian normalization for mat in coo_matrix

        Args:
            mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

        Returns:
            scipy.sparse.coo_matrix: normalized adjacent matrix
        """
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def _make_torch_adj(self, mat, self_loop=False):
        """Transform uni-directional adjacent matrix in coo_matrix into bi-directional adjacent matrix in torch.sparse.FloatTensor

        Args:
            mat (coo_matrix): the uni-directional adjacent matrix

        Returns:
            torch.sparse.FloatTensor: the bi-directional matrix
        """
        if not self_loop:
            a = csr_matrix((configs['data']['user_num'], configs['data']['user_num']))
            b = csr_matrix((configs['data']['item_num'], configs['data']['item_num']))
        else:
            data = np.ones(configs['data']['user_num'])
            row_indices = np.arange(configs['data']['user_num'])
            column_indices = np.arange(configs['data']['user_num'])
            a = csr_matrix((data, (row_indices, column_indices)), shape=(configs['data']['user_num'], configs['data']['user_num']))

            data = np.ones(configs['data']['item_num'])
            row_indices = np.arange(configs['data']['item_num'])
            column_indices = np.arange(configs['data']['item_num'])
            b = csr_matrix((data, (row_indices, column_indices)), shape=(configs['data']['item_num'], configs['data']['item_num']))

        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = self._normalize_adj(mat)

        # make torch tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])

    def load_data(self):
        trn_mat = self._load_one_mat(self.trn_file)
        val_mat = self._load_one_mat(self.val_file)
        tst_mat = self._load_one_mat(self.tst_file)

        self.trn_mat = trn_mat
        configs['data']['user_num'], configs['data']['item_num'] = trn_mat.shape
        self.torch_adj = self._make_torch_adj(trn_mat)

        if configs['model']['name'] == 'gccf':
            self.torch_adj = self._make_torch_adj(trn_mat, self_loop=True)

        if configs['train']['loss'] == 'pairwise':
            trn_data = PairwiseTrnData(trn_mat)
        elif configs['train']['loss'] == 'pairwise_with_epoch_flag':
            trn_data = PairwiseWEpochFlagTrnData(trn_mat)

        val_data = AllRankTstData(val_mat, trn_mat)
        tst_data = AllRankTstData(tst_mat, trn_mat)
        self.test_dataloader = data.DataLoader(tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.valid_dataloader = data.DataLoader(val_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=0)
        self.train_dataloader = data.DataLoader(trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)
        if configs['model']['name'][-3:] == 'crv':
            self._build_semantic_graph()
