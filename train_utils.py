# train_utils.py

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from transformers import PretrainedConfig
import math

def sample_gumbel(shape, device="cuda", eps=1e-20):
    """从 Gumbel 分布中采样。"""
    u = torch.rand(shape, device=device)
    # print(u.device) # 原始代码中有这个 print，但为了模块化我暂时注释
    return -torch.log(-torch.log(u + eps) + eps)

def gumbel_softmax_sample(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Gumbel-Softmax 采样。"""
    g = sample_gumbel(logits.shape, device=logits.device)
    y = (logits + g) / tau
    return F.softmax(y, dim=-1)

def estimate_param_fraction(h_ratio, inter_ratio, head_ratio, layer_mask: List[float], cfg: PretrainedConfig):
    """
    估算当前子网相对于完整模型的参数比例 (Fraction)。
    注意：这里简化了估计，只考虑了 Attention 和 MLP 的权重，并且假设了 Qwen 的结构。
    """
    # h_ratio, inter_ratio, head_ratio 可以是 float 或 0-dim tensor
    
    H_full = cfg.hidden_size
    I_full = cfg.intermediate_size
    
    h = h_ratio
    inter = inter_ratio
    
    # Attention 模块的参数比例 (简化估计，主要考虑 QKV 和 O 投影)
    # H -> QKV/O projection: 4 * H^2
    # Active: QKV: 2*H*H_kv + H*H_q. O: H_q * H 
    # 简化为：(h*H_full)^2 / (H_full^2) 
    attn_frac = (h * H_full) ** 2 / (H_full ** 2)
    
    # MLP 模块的参数比例 (Gate/Up/Down projection)
    # H -> I -> H: 2 * H * I
    # Active: h*H_full * inter*I_full * 2
    # 简化为：((h * H_full) * (inter * I_full)) / (H_full * I_full)
    mlp_frac = ((h * H_full) * (inter * I_full)) / (H_full * I_full)
    
    # 单层平均参数比例 (粗略估计)
    # 假设 Attention 和 MLP 占模型参数的比例大致相当 (各约 50%)
    per_layer_frac = (attn_frac + mlp_frac) / 2.0 
    
    # 深度比例
    keep_frac = sum(layer_mask)/len(layer_mask)
    
    total_frac = per_layer_frac * keep_frac
    
    # 如果是 tensor，返回 tensor
    if isinstance(h_ratio, torch.Tensor):
         return total_frac
    # 如果是 float，返回 float
    return total_frac if isinstance(total_frac, float) else total_frac.item()

@torch.no_grad()
def evaluate_model(
    elastic_model: 'ElasticQwenForTraining', # 使用引号避免循环引用
    router: 'ElasticRouter',
    test_loader: DataLoader,
    target_device: torch.device,
    test_budget: float = 0.5,
) -> Dict[str, float]:
    """
    根据给定的预算，使用路由器确定固定的子网，并在测试数据加载器上评估模型性能 (PPL)。
    """
    print(f"\n--- Starting Evaluation at Budget: {test_budget:.2f} ---")
    
    elastic_model.eval()
    router.eval()
    total_layers = len(elastic_model.base_model.layers)
    total_loss = 0.0
    num_batches = 0

    # 1. 路由器前向传播 (获取确定的子网参数)
    with torch.no_grad():
        router_output = router(test_budget)
        
        # 提取训练时的连续比例
        h_ratio_cont = torch.tensor(router_output["hidden_ratio"], device=target_device)
        ha_ratio_cont = torch.tensor(router_output["head_ratio"], device=target_device)
        inter_ratio_cont = torch.tensor(router_output["inter_ratio"], device=target_device)
        layer_mask_list = router_output["layer_mask"]
        
        # 使用连续值设置比例
        h_ratio = h_ratio_cont.item()
        ha_ratio = ha_ratio_cont.item()
        inter_ratio = inter_ratio_cont.item()
        
        # 确保 layer_mask 长度正确
        if len(layer_mask_list) != total_layers:
            if len(layer_mask_list) < total_layers:
                layer_mask_list = layer_mask_list + [1.0] * (total_layers - len(layer_mask_list))
            else:
                layer_mask_list = layer_mask_list[:total_layers]
        
        # 计算子网参数占比 (Fraction)
        fra = estimate_param_fraction(h_ratio_cont, inter_ratio_cont, ha_ratio_cont, layer_mask_list, elastic_model.config)


    # 2. 设置固定的子网
    elastic_model.set_active_subnet(
        h_ratio=h_ratio, 
        ha_ratio=ha_ratio, 
        intermediate_ratio=inter_ratio, 
        layer_mask=layer_mask_list
    )
    
    print(f"[SUB] Set Subnet: H={h_ratio:.3f}, HA={ha_ratio:.3f}, I={inter_ratio:.3f}, Depth={int(sum(layer_mask_list))}/{total_layers}. Est. Params: {fra:.3f}")
    
    # 3. 在测试集上运行评估
    for step, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(target_device)
        labels = input_ids.clone().to(target_device)
        
        # 前向传播
        _, loss = elastic_model(input_ids, labels)
        
        if loss is not None:
            total_loss += loss.item()
            num_batches += 1
        
        if step % 100 == 0 and step > 0:
            print(f"  [EVAL] Processed {step} batches. Current Avg Loss: {total_loss / num_batches:.4f}")

    if num_batches == 0:
        print("[WARN] No batches processed for evaluation.")
        return {"budget": test_budget, "loss": 0.0, "ppl": float('inf'), "param_frac": fra}

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"\n[EVAL DONE] Budget={test_budget:.2f} | Avg Loss={avg_loss:.4f} | PPL={perplexity:.2f}")

    return {
        "budget": test_budget,
        "loss": avg_loss,
        "ppl": perplexity,
        "param_frac": fra
    }
