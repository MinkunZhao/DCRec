import torch
import pandas as pd

class CandidateGenerator:
    def __init__(self, model, data_handler):
        self.model = model
        self.data = data_handler

    def generate(self, topk=50, neg_ratio=5):
        """生成含多模态特征的候选对"""
        all_users = self.data.unique_users
        candidates = []

        # 生成正例候选
        with torch.no_grad():
            user_emb, item_emb = self.model.get_embeddings()
            scores = user_emb @ item_emb.T

            for u in all_users:
                # 取topk正例
                pos_items = torch.topk(scores[u], topk).indices
                for i in pos_items:
                    candidates.append((
                        u, i.item(), 1,  # (user, item, label)
                        self.data.get_features(u, i.item())
                    ))

                # 负采样
                neg_items = self.data.negative_sampling(u, neg_ratio)
                for i in neg_items:
                    candidates.append((
                        u, i, 0,
                        self.data.get_features(u, i)
                    ))

        return pd.DataFrame(candidates,
                            columns=['user', 'item', 'label', 'features'])
