import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from functools import partial
import torch.nn.functional as F
import collections

def compute_importance(model_name="Qwen/Qwen2.5-0.5B", num_samples=50, device="cuda"):
    """
    计算 Attention 模块 (o_proj) 和 MLP 模块 (down_proj) 输出的绝对平均激活值作为通道重要性。
    """
    # 检查 CUDA 设备
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA 不可用，正在使用 CPU。")
        device = "cpu"

    print(f"✅ 正在加载模型：{model_name} 到 {device}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ 模型或分词器加载失败：{e}")
        return

    model.eval()

    # 存储激活值：{layer_name: {'attn': tensor, 'mlp': tensor}}
    activations = defaultdict(partial(defaultdict, float))

    # ✅ Attention hook：注册在 o_proj 之后
    def attn_hook(name):
        def hook(module, inp, out):
            with torch.no_grad():
                # out.shape: (batch_size, seq_len, hidden_size)
                # 求所有样本和序列位置的平均绝对值，得到 hidden_size 维度的重要性
                val = out.detach().abs().mean(dim=(0, 1))
                activations[name]['attn'] += val.cpu()
        return hook

    # ✅ MLP hook：注册在 down_proj 之后
    def mlp_hook(name):
        def hook(module, inp, out):
            with torch.no_grad():
                # out.shape: (batch_size, seq_len, hidden_size)
                # 求所有样本和序列位置的平均绝对值，得到 hidden_size 维度的重要性
                val = out.detach().abs().mean(dim=(0, 1))
                activations[name]['mlp'] += val.cpu()
        return hook

    # ✅ 注册 hook
    for i, layer in enumerate(model.model.layers):
        # Attention 的输出通道与 Attention 内部的 QKV 通道对应 (hidden_size)
        layer.self_attn.o_proj.register_forward_hook(attn_hook(f"layer_{i}"))
        # MLP 的输出通道与 Attention 的输出通道维度相同 (hidden_size)
        layer.mlp.down_proj.register_forward_hook(mlp_hook(f"layer_{i}"))

    texts = [f"今天天气真不错，我们一起去散步吧 {i}" for i in range(num_samples)]
    print(f"✅ 正在用 {num_samples} 个样本运行前向传播以收集激活...")
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)

    with torch.no_grad():
        model(**inputs)

    torch.save(activations, "activations.pt")
    print("✅ 已保存激活重要性到 activations.pt")


def reorder_qwen_weights(model_name="Qwen/Qwen2-1.5B", activation_path="activations.pt", save_path="./reordered2.5_qwen"):
    """
    根据激活重要性文件重排序 Qwen2 模型的 Attention 和 MLP 权重。
    """
    try:
        with torch.serialization.safe_globals([
            collections.defaultdict,
            partial
        ]):
            activations = torch.load(activation_path, weights_only=False)
    except Exception as e:
        print(f"❌ 激活文件加载失败，请先运行 compute_importance：{e}")
        return

    print(f"✅ 正在加载模型：{model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return


    for i, layer in enumerate(model.model.layers):
        name = f"layer_{i}"
        if name not in activations:
            continue
        
        print(f"🔄 正在处理 {name}...")

        # 注意：此处使用的 attn_imp 和 mlp_imp 都是 hidden_size 维度的重要性

        # 因为 Qwen2 使用 GQA/MHA，其内部 key/value 通道数通常与 Attention 的输出通道数不同。
        # 这里为了简化，我们仅重排 Q_proj 和 O_proj 的通道。
        # 如果需要重排 K/V，需要根据模型配置计算它们各自的通道索引。
        attn_imp = activations[name]['attn'] # shape: [hidden_size]

        # ✅ Attention部分排序
        idx_attn = torch.argsort(attn_imp, descending=True)
        
        # Q_proj, K_proj, V_proj 的形状：[num_heads * head_dim, hidden_size]
        Wq = layer.self_attn.q_proj.weight.data
        Wk = layer.self_attn.k_proj.weight.data
        Wv = layer.self_attn.v_proj.weight.data
        
        # O_proj 的形状：[hidden_size, num_heads * head_dim]
        Wo = layer.self_attn.o_proj.weight.data
        
        # 修正：使用索引张量对维度进行索引 (Attention 的输出通道重排)
        # QKV 的输出维度（行）对应 Attention 的输出通道。
        layer.self_attn.q_proj.weight.data = Wq[idx_attn, :]
        # layer.self_attn.k_proj.weight.data = Wk[idx_attn, :] # 暂不重排 K/V，保留原逻辑
        # layer.self_attn.v_proj.weight.data = Wv[idx_attn, :] # 暂不重排 K/V，保留原逻辑
        
        # 修正：O_proj 的输入维度（列）对应 Attention 的输出通道。
        layer.self_attn.o_proj.weight.data = Wo[:, idx_attn]


        # ✅ MLP部分排序
        mlp_imp = activations[name]['mlp'] # shape: [hidden_size]
        
        # up_proj 的形状：[intermediate_size, hidden_size]
        W1 = layer.mlp.up_proj.weight.data
        
        # down_proj 的形状：[hidden_size, intermediate_size]
        W2 = layer.mlp.down_proj.weight.data
        
        # 根据 MLP hook 得到的 hidden_size 维度重要性进行排序
        idx_mlp = torch.argsort(mlp_imp, descending=True)
        
        # 错误行：layer.mlp.up_proj.weight.data = W1[:idx_mlp] 
        # 修正：up_proj 的行（输出维度）对应 intermediate_size，列（输入维度）对应 hidden_size
        # 我们对 W1 的 **列**（hidden_size）进行重排
        layer.mlp.up_proj.weight.data = W1[:, idx_mlp]
        
        # 错误行：layer.mlp.down_proj.weight.data = W2[idx_mlp]
        # 修正：down_proj 的行（输出维度）对应 hidden_size，列（输入维度）对应 intermediate_size
        # 我们对 W2 的 **行**（hidden_size）进行重排
        layer.mlp.down_proj.weight.data = W2[idx_mlp, :]


    model.save_pretrained(save_path)
    print(f"✅ 已保存重排序后的 Qwen 模型到 {save_path}")


if __name__ == "__main__":
    # 配置
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    DEVICE = "cuda" # 确保您的环境中支持 CUDA
    NUM_SAMPLES = 50
    ACTIVATION_PATH = "activations.pt"
    SAVE_PATH = "./reordered_qwen"

    # 1. 计算重要性 (激活值)
    compute_importance(model_name=MODEL_NAME, num_samples=NUM_SAMPLES, device=DEVICE)
    
    # 2. 重排序权重并保存
    reorder_qwen_weights(model_name=MODEL_NAME, activation_path=ACTIVATION_PATH, save_path=SAVE_PATH)