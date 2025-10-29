import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
import math
import warnings
from torch.utils.data import Dataset, DataLoader
import sys
import random
from typing import Optional, List, Tuple, Dict
import os

class LocalTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=64):
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

def sample_gumbel(shape, device="cuda", eps=1e-20):
    u = torch.rand(shape, device=device)
    print(u.device)
    return -torch.log(-torch.log(u + eps) + eps)

def gumbel_softmax_sample(logits, tau):
    g = sample_gumbel(logits.shape, device=logits.device)
    y = (logits + g) / tau
    return F.softmax(y, dim=-1)

class BudgetEncoder(nn.Module):
    def __init__(self, known_budgets=[1.0, 0.75, 0.5, 0.25], embed_dim=32, device="cuda"):
        super().__init__()
        self.known_budgets = torch.tensor(known_budgets, device=device)
        self.embedding = nn.Embedding(len(known_budgets), embed_dim) 

    def forward(self, b):
        b = torch.tensor([b], dtype=torch.float32, device=self.known_budgets.device)
        
        for i, budget_val in enumerate(self.known_budgets):
            if torch.isclose(b, budget_val):
                return self.embedding(torch.tensor([i], device=b.device))
        
        known_budgets_device = self.known_budgets.to(b.device)
        if b < known_budgets_device.min() or b > known_budgets_device.max():
             if b < known_budgets_device.min(): below = above = known_budgets_device.min()
             else: below = above = known_budgets_device.max()
        else:
             below = torch.max(known_budgets_device[known_budgets_device <= b])
             above = torch.min(known_budgets_device[known_budgets_device >= b])

        idx_below = (known_budgets_device == below).nonzero(as_tuple=True)[0]
        idx_above = (known_budgets_device == above).nonzero(as_tuple=True)[0]
        
        emb_below = self.embedding(idx_below)
        emb_above = self.embedding(idx_above)

        if torch.isclose(below, above):
             return emb_below 
        
        w = (b - below) / (above - below)
        return (1 - w) * emb_below + w * emb_above

class ElasticRouter(nn.Module):
    def __init__(self, num_layers, known_budgets=[1.0, 0.75, 0.5, 0.25], hidden_dim=64, device="cuda"):
        super().__init__()
        self.encoder = BudgetEncoder(known_budgets, device=device)
        
        embed_dim = self.encoder.embedding.embedding_dim
        
        self.hidden_size_mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 8))
        self.head_mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 8))
        self.inter_mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 8))
        self.layer_mlp = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_layers))
        
        self.ratios = nn.Parameter(torch.tensor([1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125], device=device), requires_grad=False)
        self.num_layers = num_layers
        
        self._tau = 5.0
        self._tau_min = 0.5
        self._anneal_rate = 0.95

    def anneal_tau(self):
        self._tau = max(self._tau * self._anneal_rate, self._tau_min)

    def get_tau(self):
        return self._tau

    def forward(self, budget: float) -> Dict[str, torch.Tensor]:
        h = self.encoder(budget).squeeze(0)
        
        logits_hid = self.hidden_size_mlp(h)
        logits_head = self.head_mlp(h)
        logits_inter = self.inter_mlp(h)
        logits_layer = self.layer_mlp(h)

        p_hid = gumbel_softmax_sample(logits_hid, self._tau)
        p_head = gumbel_softmax_sample(logits_head, self._tau)
        p_inter = gumbel_softmax_sample(logits_inter, self._tau)
        
        p_layer = torch.sigmoid(logits_layer) 
        
        hid_ratio = torch.sum(p_hid * self.ratios.to(p_hid.device))
        head_ratio = torch.sum(p_head * self.ratios.to(p_head.device))
        inter_ratio = torch.sum(p_inter * self.ratios.to(p_inter.device))
        
        layer_mask = (p_layer > 0.5).float() 
        
        return {
            "hidden_ratio": hid_ratio.item(),
            "head_ratio": head_ratio.item(),
            "inter_ratio": inter_ratio.item(),
            "layer_mask": layer_mask.tolist(), 
            "tau": self._tau
        }



class PolicyAwareModulation(nn.Module):
    def __init__(self, hidden_size, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size 

        self.fc1 = nn.Linear(self.embed_dim, hidden_size)
        self.act = nn.SiLU()
        self.scale_mlp = nn.Linear(hidden_size, hidden_size)
        self.shift_mlp = nn.Linear(hidden_size, hidden_size)

    def sinusoidal_embed(self, ratio, device):
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)

        if not isinstance(ratio, torch.Tensor):
            ratio = torch.tensor([ratio], dtype=torch.float32, device=device)
        else:
            ratio = ratio.to(device) 

        emb = ratio.unsqueeze(-1) * freq
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, hidden_states: torch.Tensor, ratio: float):
        device = hidden_states.device 
        active_hidden_size = hidden_states.size(-1) 
        emb = self.sinusoidal_embed(ratio, device)
        h = self.fc1(emb)
        h = self.act(h)
        scale = self.scale_mlp(h) 
        shift = self.shift_mlp(h) 
        
        scale_cropped = scale.squeeze(0)[:active_hidden_size]
        shift_cropped = shift.squeeze(0)[:active_hidden_size]
        
        return hidden_states * (1 + scale_cropped) + shift_cropped

class ElasticNorm(nn.Module):
    def __init__(self, base_norm, max_hidden_size: int, epsilon: float):
        super().__init__()
        self.max_hidden = max_hidden_size
        self.active_hidden = max_hidden_size
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
        variance = x_cropped.pow(2).mean(-1, keepdim=True)
        eps_tensor = torch.tensor(self.eps, dtype=variance.dtype, device=variance.device)
        x_cropped = x_cropped * torch.rsqrt(variance + eps_tensor)
        output = x_cropped * W
        
        if self.has_bias:
             output = output + self.bias[:self.active_hidden]
             
        return output

class ElasticLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.max_in = base_linear.in_features
        self.max_out = base_linear.out_features
        self.weight = nn.Parameter(base_linear.weight.data.clone()) 
        self.bias = nn.Parameter(base_linear.bias.data.clone()) if base_linear.bias is not None else None
        self.active_in = self.max_in
        self.active_out = self.max_out

    def set_active_size(self, in_ratio=1.0, out_ratio=1.0):
        self.active_in = int(self.max_in * in_ratio)
        self.active_out = int(self.max_out * out_ratio)

    def forward(self, x):
        W = self.weight[:self.active_out, :self.active_in]
        b = self.bias[:self.active_out] if self.bias is not None else None
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

    def forward(self, hidden_states, **kwargs):
        B, T, H = hidden_states.size()

        if self.active_hidden < self.max_hidden:
            hidden_states = hidden_states[..., :self.active_hidden]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        num_heads = self.active_heads
        kv_heads = self.active_kv_heads
        head_dim = self.head_dim

        q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, kv_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, kv_heads, head_dim).transpose(1, 2)

        repeat_factor = num_heads // kv_heads
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, num_heads * head_dim)

        out = self.o_proj(out)
        return out

class ElasticMLP(nn.Module):
    def __init__(self, base_mlp):
        super().__init__()
        self.gate_proj = ElasticLinear(base_mlp.gate_proj)
        self.up_proj = ElasticLinear(base_mlp.up_proj)
        self.down_proj = ElasticLinear(base_mlp.down_proj)
        self.act_fn = base_mlp.act_fn

    def set_active_size(self, input_ratio=1.0, intermediate_ratio=1.0):
        self.gate_proj.set_active_size(in_ratio=input_ratio, out_ratio=intermediate_ratio)
        self.up_proj.set_active_size(in_ratio=input_ratio, out_ratio=intermediate_ratio)
        self.down_proj.set_active_size(in_ratio=intermediate_ratio, out_ratio=input_ratio)

    def forward(self, hidden_states):
        x = hidden_states.to(torch.float16)
        x = self.gate_proj(x) * self.act_fn(self.up_proj(x))
        x = self.down_proj(x)
        return x

class ElasticDecoderLayer(nn.Module):
    def __init__(self, base_layer, config: PretrainedConfig):
        super().__init__()
        self.self_attn = ElasticAttention(base_layer.self_attn, config)
        self.mlp = ElasticMLP(base_layer.mlp)
        rms_norm_eps = config.rms_norm_eps if hasattr(config, 'rms_norm_eps') else 1e-6
        self.input_layernorm = ElasticNorm(base_layer.input_layernorm, config.hidden_size, epsilon=rms_norm_eps)
        self.post_attention_layernorm = ElasticNorm(base_layer.post_attention_layernorm, config.hidden_size, epsilon=rms_norm_eps)
        self.attn_mod = PolicyAwareModulation(config.hidden_size) 
        self.mlp_mod = PolicyAwareModulation(config.hidden_size) 
        self.active_hidden = config.hidden_size 

    def set_active_subnet(self, ha_ratio=1.0, h_ratio=1.0, intermediate_ratio=1.0):
        self.self_attn.set_active_heads(ha_ratio=ha_ratio, h_ratio=h_ratio)
        self.active_hidden = self.self_attn.active_hidden
        self.input_layernorm.set_active_size(self.active_hidden)
        self.post_attention_layernorm.set_active_size(self.active_hidden)
        self.mlp.set_active_size(input_ratio=h_ratio, intermediate_ratio=intermediate_ratio)
        self.ha_ratio = ha_ratio
        self.intermediate_ratio = intermediate_ratio

    def forward(self, hidden_states): 
        layer_input = hidden_states[..., :self.active_hidden].contiguous() 
        residual = layer_input 
        
        norm_input = self.input_layernorm(layer_input) 
        norm_input = norm_input.to(torch.float16)
        attn_output = self.self_attn(norm_input) 
        
        attn_output = self.attn_mod(attn_output, self.ha_ratio) 
        hidden_states = residual + attn_output 
        
        mlp_residual = hidden_states
        norm_input = self.post_attention_layernorm(mlp_residual) 
        
        mlp_output = self.mlp(norm_input) 
        mlp_output = self.mlp_mod(mlp_output, self.intermediate_ratio) 
        
        hidden_states = mlp_residual + mlp_output 
        return hidden_states

class ElasticQwenWrapper(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: PretrainedConfig):
        super().__init__()
        self.embed_tokens = base_model.embed_tokens
        rms_norm_eps = config.rms_norm_eps if hasattr(config, 'rms_norm_eps') else 1e-6
        self.norm = ElasticNorm(base_model.norm, config.hidden_size, epsilon=rms_norm_eps)
        self.layers = nn.ModuleList([ElasticDecoderLayer(layer, config) for layer in base_model.layers])
        self.layer_mask = None
        self.config = config

    def set_active_subnet(self, h_ratio=1.0, ha_ratio=1.0, intermediate_ratio=1.0, layer_mask: Optional[List[float]] = None):
        self.layer_mask = layer_mask if layer_mask is not None else [1.0] * len(self.layers)
        
        self.layers[0].set_active_subnet(ha_ratio=ha_ratio, h_ratio=h_ratio, intermediate_ratio=intermediate_ratio)
        active_hidden_size = self.layers[0].active_hidden
        
        self.norm.set_active_size(active_hidden_size)
        
        for layer in self.layers:
             layer.set_active_subnet(ha_ratio=ha_ratio, h_ratio=h_ratio, intermediate_ratio=intermediate_ratio)


    def forward(self, input_embeds: torch.Tensor, layer_mask: Optional[List[float]] = None) -> torch.Tensor:
        hidden_states = input_embeds
        mask = layer_mask if layer_mask is not None else self.layer_mask
        
        if mask is None or len(mask) != len(self.layers):
             mask = [1.0] * len(self.layers) 
        
        for i, use_layer in enumerate(mask):
            current_layer = self.layers[i]
            if use_layer > 0.5:
                 hidden_states = current_layer(hidden_states)

        return self.norm(hidden_states)

class ElasticQwenForTraining(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM):
        super().__init__()
        self.base_model = ElasticQwenWrapper(base_model.model, base_model.config)
        self.lm_head = ElasticLinear(base_model.lm_head)
        self.config = base_model.config
        self.embed_tokens = self.base_model.embed_tokens

    def set_active_subnet(self, h_ratio=1.0, ha_ratio=1.0, intermediate_ratio=1.0, layer_mask: Optional[List[float]] = None):
        self.base_model.set_active_subnet(h_ratio, ha_ratio, intermediate_ratio, layer_mask)
        
        active_hidden_size = self.base_model.layers[0].active_hidden
        self.lm_head.set_active_size(in_ratio=h_ratio, out_ratio=1.0)
          
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.base_model(input_embeds)
        
        hidden_states = hidden_states.half() 
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
             shift_logits = logits[..., :-1, :].contiguous()
             shift_labels = labels[..., 1:].contiguous()
             
             loss_fct = nn.CrossEntropyLoss()
             loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
             
        return logits, loss
def estimate_param_fraction(h_ratio, inter_ratio, head_ratio, layer_mask, cfg):
    device = h_ratio.device if isinstance(h_ratio, torch.Tensor) else None

    H_full = cfg.hidden_size
    I_full = cfg.intermediate_size

    h = h_ratio
    inter = inter_ratio
    attn_frac = (h * H_full) ** 2 / (H_full ** 2)
    mlp_frac = (h * H_full) * (inter * I_full) / (H_full * I_full)
    per_layer_frac = attn_frac + mlp_frac
    per_layer_frac = per_layer_frac / 2.0
    keep_frac = sum(layer_mask)/len(layer_mask)
    total_frac = per_layer_frac * keep_frac
    return total_frac
@torch.no_grad() # 确保在评估模式下不计算梯度
def evaluate_model(
    elastic_model: ElasticQwenForTraining,
    router: ElasticRouter,
    test_loader: DataLoader,
    target_device: torch.device,
    test_budget: float = 0.5,
) -> Dict[str, float]:
    """
    根据给定的预算，使用路由器确定固定子网，并在测试数据加载器上评估模型性能 (PPL)。
    """
    print(f"\n--- Starting Evaluation at Budget: {test_budget:.2f} ---")
    
    elastic_model.eval()
    router.eval()
    total_layers = len(elastic_model.base_model.layers)
    total_loss = 0.0
    num_batches = 0

    # 1. 路由器前向传播 (使用固定的 Gumbel-Softmax)
    # 在评估时，我们通常希望使用 argmax (或 Sigmoid 阈值) 来获得固定的、可量化的子网。
    # 这里我们模拟 argmax 逻辑来获得确定的 ratio 和 mask。
    with torch.no_grad():
        router_output = router(test_budget)
        
        # 提取训练时的连续比例
        h_ratio_cont = torch.tensor(router_output["hidden_ratio"], device=target_device)
        ha_ratio_cont = torch.tensor(router_output["head_ratio"], device=target_device)
        inter_ratio_cont = torch.tensor(router_output["inter_ratio"], device=target_device)
        layer_mask_list = router_output["layer_mask"]
        
        # ⚠️ 注意: 在实际推理中，你可能需要对 ratio 进行离散化 (argmax)
        # 但为了简化，我们暂时使用训练时 Gumbel-Softmax 输出的期望值 (连续值) 来设置比例，
        # 并使用 layer_mask 的二值化结果。
        
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
        fra = estimate_param_fraction(h_ratio_cont, inter_ratio_cont, ha_ratio_cont, layer_mask_list, elastic_model.config).item()


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

# --------------------------------------------------------------------------------

if __name__ == '__main__':
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)

    device_id = 2
    target_device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {target_device}")

    warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

    model_name = "Qwen/Qwen2.5-0.5B"
    model_path = "reordered_qwen"
    BATCH_SIZE = 2
    SEQ_LEN = 128
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_TRAINING_EPOCHS = 3
    
    data_path = "/home/lx/my_qwen2.5_0.5B/my_txt.txt"
    save_dir = "elastic_qwen_output"
    os.makedirs(save_dir, exist_ok=True)

    TRAINING_BUDGETS = [1.0, 0.9, 0.7, 0.5, 0.3]

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_name}: {e}")
        sys.exit(1)
    
    try:
        dataset = LocalTextDataset(data_path, tokenizer, max_length=SEQ_LEN)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"[INFO] Dataset loaded with {len(dataset)} samples. Batch size: {BATCH_SIZE}")
    except FileNotFoundError:
        print(f"[FATAL] Data file not found at: {data_path}")
        sys.exit(1)
        
    elastic_model = ElasticQwenForTraining(base_model).train().to(target_device)
    total_layers = len(elastic_model.base_model.layers)
    router = ElasticRouter(num_layers=total_layers, known_budgets=TRAINING_BUDGETS, device=target_device).to(target_device)
    
    params = list(elastic_model.parameters()) + list(router.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-5)

    print(f"\n[INFO] Starting Router-Driven Elastic Training on {target_device}.")
    print(f"[INFO] Training for {NUM_TRAINING_EPOCHS} epochs.")

    global_step = 0
    
    for epoch in range(NUM_TRAINING_EPOCHS):
        running_loss = 0.0
        
        for step, batch in enumerate(loader, 1): 
            current_budget = float(random.choice(TRAINING_BUDGETS))
            current_budget_tensor = torch.tensor(current_budget, dtype=torch.float32, device=target_device)
            
            router_output = router(current_budget)
            h_ratio = float(router_output["hidden_ratio"])
            ha_ratio = float(router_output["head_ratio"])
            inter_ratio = float(router_output["inter_ratio"])
            layer_mask = router_output["layer_mask"]
            current_tau = router_output["tau"]
            
            fra = estimate_param_fraction(torch.tensor(h_ratio, device=target_device), torch.tensor(inter_ratio, device=target_device), torch.tensor(ha_ratio, device=target_device), layer_mask, elastic_model.config)
            loss_penalty = (fra-current_budget_tensor)*(fra-current_budget_tensor)
            
            if len(layer_mask) != total_layers:
                 if len(layer_mask) < total_layers:
                     layer_mask = layer_mask + [1.0] * (total_layers - len(layer_mask))
                 else:
                     layer_mask = layer_mask[:total_layers]

            elastic_model.set_active_subnet(h_ratio=h_ratio, ha_ratio=ha_ratio, intermediate_ratio=inter_ratio, layer_mask=layer_mask)

            input_ids = batch["input_ids"].to(target_device)
            labels = input_ids.clone().to(target_device)

            logits, loss = elastic_model(input_ids, labels)
            
            loss = loss + loss_penalty.to(loss.device) * 0.1
            loss_to_backprop = loss / GRADIENT_ACCUMULATION_STEPS
            loss_to_backprop.backward()

            running_loss += loss.item()

            if step % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                router.anneal_tau()

                global_step += 1
                avg_loss = running_loss / GRADIENT_ACCUMULATION_STEPS
                print(f"[STEP] Epoch {epoch+1}/{NUM_TRAINING_EPOCHS} | global_step={global_step:4d} | budget={current_budget:.2f} | loss={avg_loss:.4f} | tau={router.get_tau():.4f} | subnet H={h_ratio:.3f} HA={ha_ratio:.3f} I={inter_ratio:.3f} depth={int(sum(layer_mask))}/{total_layers}")
                running_loss = 0.0
            else:
                 print(f"[ACC] Epoch {epoch+1} step={step} (accumulating) | budget={current_budget:.2f} | loss={loss.item():.4f}")

    print("[INFO] Training finished.")

    try:
        ckpt_path = "elastic_qwen_checkpoint.pt"
        save_dict = {
            "model_state_dict": {k: v.cpu() for k, v in elastic_model.state_dict().items()},
            "router_state_dict": {k: v.cpu() for k, v in router.state_dict().items()},
            "tokenizer": None,
        }
        torch.save(save_dict, ckpt_path)
        print(f"[INFO] Checkpoint saved to {ckpt_path}")
    except Exception as e:
        print(f"[WARN] Failed to save checkpoint: {e}")
# --- 8. 模型评估 (新增部分) ---
    print("\n===============================")
    print("=== STARTING MODEL EVALUATION ===")
    print("===============================\n")

    # 重新加载数据加载器以作为测试集 (在实际项目中应使用单独的测试集文件)
    # 我们假设这里复用 loader 作为测试集，但您应该在实际应用中使用一个独立的测试文件 (e.g., test_data.txt)
    # test_data_path = "my_test_data.txt" 
    # test_dataset = LocalTextDataset(test_data_path, tokenizer, max_length=SEQ_LEN)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # 评估时通常不打乱
    
    # 这里我们直接复用训练时的 loader (仅为演示)
    test_loader = loader 
    
    # 定义要评估的预算点
    EVAL_BUDGETS = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2] 
    
    results = {}

    for budget in EVAL_BUDGETS:
        # 重置模型到评估模式
        elastic_model.train(False)
        router.train(False)
        
        # 执行评估
        res = evaluate_model(
            elastic_model=elastic_model,
            router=router,
            test_loader=test_loader,
            target_device=target_device,
            test_budget=budget
        )
        results[budget] = res
        
    print("\n--- Summary of Evaluation Results ---")
    for budget, res in results.items():
        print(f"Budget: {budget:.2f} | Param Frac: {res['param_frac']:.3f} | Loss: {res['loss']:.4f} | PPL: {res['ppl']:.2f}")