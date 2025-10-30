# router.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from train_utils import gumbel_softmax_sample # 引入 Gumbel 采样

class BudgetEncoder(nn.Module):
    """将连续的预算 (budget) 编码为嵌入向量，使用最近的已知预算的线性插值。"""
    def __init__(self, known_budgets: List[float] = [1.0, 0.75, 0.5, 0.25], embed_dim: int = 32, device: str = "cuda"):
        super().__init__()
        self.known_budgets = torch.tensor(known_budgets, device=device)
        self.embedding = nn.Embedding(len(known_budgets), embed_dim) 
        self.device = device # 存储设备信息

    def forward(self, b: float) -> torch.Tensor:
        # 确保输入 b 是 tensor 并在正确的设备上
        b_tensor = torch.tensor([b], dtype=torch.float32, device=self.device)
        known_budgets_device = self.known_budgets.to(b_tensor.device)
        
        # 1. 精确匹配
        for i, budget_val in enumerate(known_budgets_device):
            if torch.isclose(b_tensor, budget_val):
                return self.embedding(torch.tensor([i], device=b_tensor.device))
        
        # 2. 边界检查
        if b_tensor < known_budgets_device.min(): 
            below = above = known_budgets_device.min()
        elif b_tensor > known_budgets_device.max(): 
            below = above = known_budgets_device.max()
        else:
        # 3. 线性插值
            # 找到低于/高于 b_tensor 的最近已知预算
            below = known_budgets_device[known_budgets_device <= b_tensor].max()
            above = known_budgets_device[known_budgets_device >= b_tensor].min()

        idx_below = (known_budgets_device == below).nonzero(as_tuple=True)[0]
        idx_above = (known_budgets_device == above).nonzero(as_tuple=True)[0]
        
        emb_below = self.embedding(idx_below)
        emb_above = self.embedding(idx_above)

        if torch.isclose(below, above):
            return emb_below 
        
        w = (b_tensor - below) / (above - below)
        # 插值
        return (1 - w) * emb_below + w * emb_above

class ElasticRouter(nn.Module):
    """
    弹性路由器的核心模块，根据预算预测子网的配置比例和层级掩码。
    """
    def __init__(self, num_layers: int, known_budgets: List[float] = [1.0, 0.75, 0.5, 0.25], hidden_dim: int = 64, device: str = "cuda"):
        super().__init__()
        self.encoder = BudgetEncoder(known_budgets, device=device)
        
        embed_dim = self.encoder.embedding.embedding_dim
        
        # 定义用于预测子网比例的 MLPs (Gumbel-Softmax)
        self.hidden_size_mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 8))
        self.head_mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 8))
        self.inter_mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 8))
        
        # 定义用于预测层级掩码的 MLP (Sigmoid)
        self.layer_mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_layers))
        
        # 离散比例值 (固定参数，不需要梯度)
        self.ratios = nn.Parameter(torch.tensor([1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125], device=device), requires_grad=False)
        self.num_layers = num_layers
        
        # Gumbel-Softmax 温度参数
        self._tau = 5.0
        self._tau_min = 0.5
        self._anneal_rate = 0.95

    def anneal_tau(self):
        """退火 Gumbel-Softmax 温度。"""
        self._tau = max(self._tau * self._anneal_rate, self._tau_min)

    def get_tau(self):
        """获取当前 Gumbel-Softmax 温度。"""
        return self._tau

    def forward(self, budget: float) -> Dict[str, torch.Tensor]:
        # 1. 预算编码
        h = self.encoder(budget).squeeze(0)
        
        # 2. 预测 logits
        logits_hid = self.hidden_size_mlp(h)
        logits_head = self.head_mlp(h)
        logits_inter = self.inter_mlp(h)
        logits_layer = self.layer_mlp(h)

        # 3. Gumbel-Softmax 采样 (用于比例)
        # 在训练时，p_X 是连续的期望值 (soft approximation of one-hot)
        p_hid = gumbel_softmax_sample(logits_hid, self._tau)
        p_head = gumbel_softmax_sample(logits_head, self._tau)
        p_inter = gumbel_softmax_sample(logits_inter, self._tau)
        
        # 4. Sigmoid (用于层级掩码)
        # 在训练时，p_layer 是连续的 0-1 概率
        p_layer = torch.sigmoid(logits_layer) 
        
        # 5. 计算连续比例的期望值 (Ratio = sum(p_i * ratio_i))
        ratios_device = self.ratios.to(p_hid.device)
        hid_ratio = torch.sum(p_hid * ratios_device)
        head_ratio = torch.sum(p_head * ratios_device)
        inter_ratio = torch.sum(p_inter * ratios_device)
        
        # 6. 二值化层级掩码 (在评估时使用)
        layer_mask = (p_layer > 0.5).float() 
        
        # 返回用于训练和评估的参数
        return {
            "hidden_ratio": hid_ratio.item(),      # H 维度比例
            "head_ratio": head_ratio.item(),        # Head 比例 (ha)
            "inter_ratio": inter_ratio.item(),      # MLP 中间维度比例 (inter)
            "layer_mask": layer_mask.tolist(),      # 层级掩码 (0 或 1)
            "tau": self._tau                        # 当前温度
        }
