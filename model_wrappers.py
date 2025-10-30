# model_wrappers.py

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from elastic_components import ElasticNorm, ElasticDecoderLayer, ElasticLinear # 引入弹性组件
from typing import Optional, List
import math

class PolicyAwareModulation(nn.Module):
    """
    策略感知调制 (PAM)，根据子网比例 (ratio) 生成 scale 和 shift 因子来调整 hidden_states。
    """
    def __init__(self, hidden_size: int, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size 

        # 用于处理 ratio 嵌入的 MLP
        self.fc1 = nn.Linear(self.embed_dim, hidden_size)
        self.act = nn.SiLU()
        self.scale_mlp = nn.Linear(hidden_size, hidden_size)
        self.shift_mlp = nn.Linear(hidden_size, hidden_size)

    def sinusoidal_embed(self, ratio: float, device: torch.device) -> torch.Tensor:
        """用正弦函数生成位置编码风格的嵌入"""
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)

        if not isinstance(ratio, torch.Tensor):
            ratio_tensor = torch.tensor([ratio], dtype=torch.float32, device=device)
        else:
            ratio_tensor = ratio.to(device) 

        emb = ratio_tensor.unsqueeze(-1) * freq
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, hidden_states: torch.Tensor, ratio: float) -> torch.Tensor:
        device = hidden_states.device 
        active_hidden_size = hidden_states.size(-1) 
        
        # 1. 生成比例嵌入
        emb = self.sinusoidal_embed(ratio, device)
        
        # 2. 计算 scale 和 shift
        h = self.fc1(emb)
        h = self.act(h)
        scale = self.scale_mlp(h) 
        shift = self.shift_mlp(h) 
        
        # 3. 裁剪 scale/shift 到当前激活的隐藏维度
        scale_cropped = scale.squeeze(0)[:active_hidden_size]
        shift_cropped = shift.squeeze(0)[:active_hidden_size]
        
        # 4. 调制: y = x * (1 + scale) + shift
        return hidden_states * (1 + scale_cropped) + shift_cropped

class ElasticQwenWrapper(nn.Module):
    """
    Qwen 模型主体 (embed_tokens, layers, norm) 的弹性封装。
    """
    def __init__(self, base_model: PreTrainedModel, config: PretrainedConfig):
        super().__init__()
        self.embed_tokens = base_model.embed_tokens
        
        rms_norm_eps = getattr(config, 'rms_norm_eps', 1e-6)
        self.norm = ElasticNorm(base_model.norm, config.hidden_size, epsilon=rms_norm_eps)
        
        # 将所有的 base_layer 替换为 ElasticDecoderLayer
        self.layers = nn.ModuleList([ElasticDecoderLayer(layer, config) for layer in base_model.layers])
        self.layer_mask: Optional[List[float]] = None
        self.config = config

    def set_active_subnet(self, h_ratio: float = 1.0, ha_ratio: float = 1.0, intermediate_ratio: float = 1.0, layer_mask: Optional[List[float]] = None):
        """设置整个模型所有层的子网参数"""
        self.layer_mask = layer_mask if layer_mask is not None else [1.0] * len(self.layers)
        
        # 设置所有层的子网比例
        for layer in self.layers:
             layer.set_active_subnet(ha_ratio=ha_ratio, h_ratio=h_ratio, intermediate_ratio=intermediate_ratio)

        # Norm 层的 active_size 必须与 layer 的 active_hidden 匹配
        active_hidden_size = self.layers[0].active_hidden
        self.norm.set_active_size(active_hidden_size)
        
        # 确保 layer_mask 长度正确
        if len(self.layer_mask) != len(self.layers):
             if len(self.layer_mask) < len(self.layers):
                 self.layer_mask = self.layer_mask + [1.0] * (len(self.layers) - len(self.layer_mask))
             else:
                 self.layer_mask = self.layer_mask[:len(self.layers)]


    def forward(self, input_embeds: torch.Tensor, layer_mask: Optional[List[float]] = None) -> torch.Tensor:
        hidden_states = input_embeds
        mask = layer_mask if layer_mask is not None else self.layer_mask
        
        if mask is None or len(mask) != len(self.layers):
             mask = [1.0] * len(self.layers) 
        
        # 按照 layer_mask 选择性地执行层
        for i, use_layer in enumerate(mask):
            current_layer = self.layers[i]
            # 只有当 use_layer > 0.5 时才执行该层
            if use_layer > 0.5:
                 hidden_states = current_layer(hidden_states)

        # 最终 Norm
        return self.norm(hidden_states)

class ElasticQwenForTraining(nn.Module):
    """
    用于训练的 ElasticQwen 顶层模型，包含嵌入层、主体和 LM Head。
    """
    def __init__(self, base_model: AutoModelForCausalLM):
        super().__init__()
        # 封装模型主体
        self.base_model = ElasticQwenWrapper(base_model.model, base_model.config)
        # 封装 LM Head
        self.lm_head = ElasticLinear(base_model.lm_head)
        self.config = base_model.config
        self.embed_tokens = self.base_model.embed_tokens # 方便直接访问嵌入层

    def set_active_subnet(self, h_ratio: float = 1.0, ha_ratio: float = 1.0, intermediate_ratio: float = 1.0, layer_mask: Optional[List[float]] = None):
        """设置所有弹性组件的子网参数"""
        self.base_model.set_active_subnet(h_ratio, ha_ratio, intermediate_ratio, layer_mask)
        
        # LM Head 只需设置输入维度 (与 h_ratio 对应)
        self.lm_head.set_active_size(in_ratio=h_ratio, out_ratio=1.0) # out_ratio 1.0 = vocab_size

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # 1. 嵌入
        input_embeds = self.embed_tokens(input_ids)
        
        # 2. 主体前向传播 (已包含 layer_mask 逻辑)
        hidden_states = self.base_model(input_embeds)
        
        # 3. LM Head 投影
        hidden_states = hidden_states.half() # 匹配 Qwen 的 float16 习惯
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 计算语言模型损失 (CrossEntropyLoss)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return logits, loss
