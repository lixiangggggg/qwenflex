# elastic_components.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
import math
from model_wrappers import PolicyAwareModulation # 引入 PolicyAwareModulation
from typing import Optional

class ElasticNorm(nn.Module):
    """可伸缩的 RMS Norm (或 Layer Norm)"""
    def __init__(self, base_norm: nn.Module, max_hidden_size: int, epsilon: float):
        super().__init__()
        self.max_hidden = max_hidden_size
        self.active_hidden = max_hidden_size
        # 复制基准 Norm 的权重
        self.weight = nn.Parameter(base_norm.weight.data.clone())
        
        if hasattr(base_norm, 'bias') and base_norm.bias is not None:
            self.bias = nn.Parameter(base_norm.bias.data.clone())
            self.has_bias = True
        else:
            self.register_parameter('bias', None) 
            self.has_bias = False
            
        self.eps = epsilon 

    def set_active_size(self, active_size: int):
        self.active_hidden = active_size

    def forward(self, x: torch.Tensor):
        x_cropped = x[..., :self.active_hidden]
        W = self.weight[:self.active_hidden]
        
        # RMS Norm
        variance = x_cropped.pow(2).mean(-1, keepdim=True)
        eps_tensor = torch.tensor(self.eps, dtype=variance.dtype, device=variance.device)
        x_cropped = x_cropped * torch.rsqrt(variance + eps_tensor)
        
        output = x_cropped * W
        
        if self.has_bias:
            output = output + self.bias[:self.active_hidden]
            
        return output

class ElasticLinear(nn.Module):
    """可伸缩的 Linear 层，通过裁剪权重和偏置来实现维度变化"""
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.max_in = base_linear.in_features
        self.max_out = base_linear.out_features
        self.weight = nn.Parameter(base_linear.weight.data.clone()) 
        self.bias = nn.Parameter(base_linear.bias.data.clone()) if base_linear.bias is not None else None
        self.active_in = self.max_in
        self.active_out = self.max_out

    def set_active_size(self, in_ratio: float = 1.0, out_ratio: float = 1.0):
        self.active_in = int(self.max_in * in_ratio)
        self.active_out = int(self.max_out * out_ratio)
        # 确保尺寸至少为 1
        self.active_in = max(1, self.active_in)
        self.active_out = max(1, self.active_out)


    def forward(self, x: torch.Tensor):
        # 裁剪权重和偏置
        W = self.weight[:self.active_out, :self.active_in]
        b = self.bias[:self.active_out] if self.bias is not None else None
        # 裁剪输入
        x_cropped = x[..., :self.active_in]
        return F.linear(x_cropped, W, b)

class ElasticAttention(nn.Module):
    def __init__(self, base_attn, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_hidden = config.hidden_size
        self.num_kv_heads = config.num_key_value_heads

        self.q_proj = ElasticLinear(base_attn.q_proj)
        self.k_proj = ElasticLinear(base_attn.k_proj)
        self.v_proj = ElasticLinear(base_attn.v_proj)
        self.o_proj = ElasticLinear(base_attn.o_proj)

        self.active_heads = self.num_heads
        self.active_hidden = self.max_hidden
        self.active_kv_heads = self.num_kv_heads

    # ... (set_active_heads 方法保持不变) ...
    def set_active_heads(self, ha_ratio=1.0, h_ratio=1.0):
        self.active_heads = max(1, int(self.num_heads * ha_ratio))
        self.active_kv_heads = max(1, int(self.num_kv_heads * ha_ratio))

        self.active_hidden = int(self.max_hidden * h_ratio)

        qkv_out_dim = self.active_heads * self.head_dim
        kv_out_dim = self.active_kv_heads * self.head_dim

        self.q_proj.set_active_size(in_ratio=h_ratio, out_ratio=qkv_out_dim / self.q_proj.max_out)
        self.k_proj.set_active_size(in_ratio=h_ratio, out_ratio=kv_out_dim / self.k_proj.max_out)
        self.v_proj.set_active_size(in_ratio=h_ratio, out_ratio=kv_out_dim / self.v_proj.max_out)

        self.o_proj.set_active_size(
            in_ratio=qkv_out_dim / self.o_proj.max_in,
            out_ratio=h_ratio 
        )

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        B, T_new, H = hidden_states.size() # T_new is the length of the new input

        # 1. Subnet Selection (Hidden Dimension)
        if self.active_hidden < self.max_hidden:
            hidden_states = hidden_states[..., :self.active_hidden]

        # 2. QKV Projection (using active subnet)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        num_heads = self.active_heads
        kv_heads = self.active_kv_heads
        head_dim = self.head_dim

        # 3. Reshape QKV
        q = q.view(B, T_new, num_heads, head_dim).transpose(1, 2)  # (B, H, T_new, D)
        k = k.view(B, T_new, kv_heads, head_dim).transpose(1, 2)    # (B, KvH, T_new, D)
        v = v.view(B, T_new, kv_heads, head_dim).transpose(1, 2)    # (B, KvH, T_new, D)

        # 4. KV Cache Integration (for incremental decoding)
        new_past_key_value = (k, v)
        if past_key_value is not None:
            # past_key_value contains (past_k, past_v)
            past_k, past_v = past_key_value 
            # Concatenate past keys/values with current keys/values
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            # Update the cache with the full K, V
            new_past_key_value = (k, v) 
        
        T_full = k.size(2) # T_full is the total sequence length (past + new)

        # 5. Grouped Query Attention (GQA) logic
        repeat_factor = num_heads // kv_heads
        if repeat_factor > 1:
            # Repeats K and V heads to match the number of Q heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # 6. Attention Score (Q @ K^T)
        # (B, H, T_new, D) @ (B, H, D, T_full) -> (B, H, T_new, T_full)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        # 7. Attention Mask Application
        if attention_mask is not None:
            # Mask is usually (B, 1, T_new, T_full) or similar
            # Add a large negative number to masked positions (where attention_mask is 0/False)
            if attention_mask.dim() == 4:
                # Add the mask directly (standard for transformers)
                attn_weights = attn_weights + attention_mask
            else:
                # Simple case: (B, T_full) mask for padding/causality (needs expansion)
                # Note: Exact mask shape handling depends on the calling Qwen model's format
                # For safety, we rely on the caller to provide the correct 4D mask in practice.
                
                # A standard mask application logic (simplified):
                if attention_mask.dim() == 2: # (B, T_full) padding mask 
                    # Assuming attention_mask is 0 for padding, 1 for real tokens
                    mask_4d = attention_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, T_full)
                    mask_4d = (1.0 - mask_4d) * torch.finfo(attn_weights.dtype).min
                    # We only apply the mask if we are processing the full sequence or the mask shape matches
                    if T_full == mask_4d.size(-1):
                         attn_weights = attn_weights + mask_4d

        # 8. Softmax and Output
        attn = F.softmax(attn_weights, dim=-1)
        
        # (B, H, T_new, T_full) @ (B, H, T_full, D) -> (B, H, T_new, D)
        out = torch.matmul(attn, v)

        # 9. Final Projection
        out = out.transpose(1, 2).contiguous().view(B, T_new, num_heads * head_dim)
        out = self.o_proj(out)
        
        # Return the output and the updated KV cache
        return out, new_past_key_value


class ElasticMLP(nn.Module):
    """可伸缩的前馈网络 (FFN)"""
    def __init__(self, base_mlp: nn.Module):
        super().__init__()
        # Qwen 使用 SwiGLU: x * act(x) -> down_proj
        self.gate_proj = ElasticLinear(base_mlp.gate_proj)
        self.up_proj = ElasticLinear(base_mlp.up_proj)
        self.down_proj = ElasticLinear(base_mlp.down_proj)
        self.act_fn = base_mlp.act_fn

    def set_active_size(self, input_ratio: float = 1.0, intermediate_ratio: float = 1.0):
        # H -> I (Gate, Up)
        self.gate_proj.set_active_size(in_ratio=input_ratio, out_ratio=intermediate_ratio)
        self.up_proj.set_active_size(in_ratio=input_ratio, out_ratio=intermediate_ratio)
        # I -> H (Down)
        self.down_proj.set_active_size(in_ratio=intermediate_ratio, out_ratio=input_ratio)

    def forward(self, hidden_states: torch.Tensor):
        # Qwen 2.5-0.5B 使用 float16
        x = hidden_states.to(torch.float16)
        
        # SwiGLU: (Gate * Activation(Up))
        x = self.gate_proj(x) * self.act_fn(self.up_proj(x))
        x = self.down_proj(x)
        return x

class ElasticDecoderLayer(nn.Module):
    """可伸缩的 Transformer Decoder Layer"""
    def __init__(self, base_layer: nn.Module, config: PretrainedConfig):
        super().__init__()
        self.self_attn = ElasticAttention(base_layer.self_attn, config)
        self.mlp = ElasticMLP(base_layer.mlp)
        
        # RMS Norm
        rms_norm_eps = getattr(config, 'rms_norm_eps', 1e-6)
        self.input_layernorm = ElasticNorm(base_layer.input_layernorm, config.hidden_size, epsilon=rms_norm_eps)
        self.post_attention_layernorm = ElasticNorm(base_layer.post_attention_layernorm, config.hidden_size, epsilon=rms_norm_eps)
        
        # 策略感知调制 (Policy-Aware Modulation)
        self.attn_mod = PolicyAwareModulation(config.hidden_size) 
        self.mlp_mod = PolicyAwareModulation(config.hidden_size) 
        self.active_hidden = config.hidden_size 
        self.ha_ratio = 1.0
        self.intermediate_ratio = 1.0

    def set_active_subnet(self, ha_ratio: float = 1.0, h_ratio: float = 1.0, intermediate_ratio: float = 1.0):
        # 1. 设置 Attention 子网
        self.self_attn.set_active_heads(ha_ratio=ha_ratio, h_ratio=h_ratio)
        self.active_hidden = self.self_attn.active_hidden
        
        # 2. 设置 Norm 子网
        self.input_layernorm.set_active_size(self.active_hidden)
        self.post_attention_layernorm.set_active_size(self.active_hidden)
        
        # 3. 设置 MLP 子网
        self.mlp.set_active_size(input_ratio=h_ratio, intermediate_ratio=intermediate_ratio)
        
        # 4. 存储比例用于调制
        self.ha_ratio = ha_ratio
        self.intermediate_ratio = intermediate_ratio

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 裁剪输入以匹配当前激活的隐藏维度
        layer_input = hidden_states[..., :self.active_hidden].contiguous() 
        residual = layer_input 
        
        # --- Attention Sub-Block ---
        norm_input = self.input_layernorm(layer_input) 
        norm_input = norm_input.to(torch.float16) # 匹配 Qwen 的 float16 习惯
        attn_output = self.self_attn(norm_input) 
        
        # 策略感知调制 (Attention Output)
        attn_output = self.attn_mod(attn_output, self.ha_ratio) 
        hidden_states = residual + attn_output # Add
        
        # --- MLP Sub-Block ---
        mlp_residual = hidden_states
        norm_input = self.post_attention_layernorm(mlp_residual) 
        
        mlp_output = self.mlp(norm_input) 
        
        # 策略感知调制 (MLP Output)
        mlp_output = self.mlp_mod(mlp_output, self.intermediate_ratio) 
        
        hidden_states = mlp_residual + mlp_output # Add
        
        # ⚠️ 注意: 这里丢失了原始输入 hidden_states 中被裁剪的部分
        # 正确的做法应该是将输出补齐到原始 hidden_states 的最大维度 (如果需要)
        # 但为了保持和原始代码逻辑一致，我们只返回裁剪后的维度。
        return hidden_states
