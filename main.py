# main.py

import torch
import torch.nn as nn
import random
import sys
import os
import warnings
from torch.utils.data import DataLoader

# --- 导入模块 ---
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import (
    SEED, DEVICE_ID, BATCH_SIZE, SEQ_LEN, GRADIENT_ACCUMULATION_STEPS,
    NUM_TRAINING_EPOCHS, TRAINING_BUDGETS, EVAL_BUDGETS, LEARNING_RATE,
    MODEL_NAME, MODEL_PATH, DATA_PATH, SAVE_DIR, CKPT_PATH
)
from data_utils import LocalTextDataset
from router import ElasticRouter
from model_wrappers import ElasticQwenForTraining
from train_utils import estimate_param_fraction, evaluate_model # 导入评估和辅助函数


if __name__ == '__main__':
    # --- 1. 初始化和设置 ---
    random.seed(SEED)
    torch.manual_seed(SEED)

    device_id = DEVICE_ID
    target_device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {target_device}")

    warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 2. 模型和分词器加载 ---
    try:
        # 尝试从 reordered_qwen 路径加载模型 (假设权重已准备好)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        # 从 huggingface 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        # 确保 tokenizer 有 pad_token，否则 DataLoader 可能会出问题
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token
             print(f"[INFO] Set pad_token to eos_token: {tokenizer.pad_token}")
             
    except Exception as e:
        print(f"[ERROR] Failed to load model {MODEL_PATH} or tokenizer {MODEL_NAME}: {e}")
        sys.exit(1)
    
    # --- 3. 数据集和数据加载器加载 ---
    try:
        dataset = LocalTextDataset(DATA_PATH, tokenizer, max_length=SEQ_LEN)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"[INFO] Dataset loaded with {len(dataset)} samples. Batch size: {BATCH_SIZE}")
    except FileNotFoundError:
        print(f"[FATAL] Data file not found at: {DATA_PATH}")
        sys.exit(1)
        
    # --- 4. 弹性模型和路由器初始化 ---
    elastic_model = ElasticQwenForTraining(base_model).train().to(target_device)
    total_layers = len(elastic_model.base_model.layers)
    
    router = ElasticRouter(
        num_layers=total_layers, 
        known_budgets=TRAINING_BUDGETS, 
        device=target_device
    ).to(target_device)
    
    params = list(elastic_model.parameters()) + list(router.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

    print(f"\n[INFO] Starting Router-Driven Elastic Training on {target_device}.")
    print(f"[INFO] Training for {NUM_TRAINING_EPOCHS} epochs.")

    # --- 5. 训练循环 ---
    global_step = 0
    
    for epoch in range(NUM_TRAINING_EPOCHS):
        running_loss = 0.0
        
        for step, batch in enumerate(loader, 1): 
            # 随机选择一个预算进行训练
            current_budget = float(random.choice(TRAINING_BUDGETS))
            current_budget_tensor = torch.tensor(current_budget, dtype=torch.float32, device=target_device)
            
            # 路由器前向传播，得到子网配置
            router_output = router(current_budget)
            h_ratio = float(router_output["hidden_ratio"])
            ha_ratio = float(router_output["head_ratio"])
            inter_ratio = float(router_output["inter_ratio"])
            layer_mask = router_output["layer_mask"]
            current_tau = router_output["tau"]
            
            # 预算损失惩罚: L_budget = (frac - budget)^2
            fra = estimate_param_fraction(
                torch.tensor(h_ratio, device=target_device), 
                torch.tensor(inter_ratio, device=target_device), 
                torch.tensor(ha_ratio, device=target_device), 
                layer_mask, 
                elastic_model.config
            )
            loss_penalty = (fra-current_budget_tensor)*(fra-current_budget_tensor)
            
            # 确保 layer_mask 长度正确
            if len(layer_mask) != total_layers:
                if len(layer_mask) < total_layers:
                    layer_mask = layer_mask + [1.0] * (total_layers - len(layer_mask))
                else:
                    layer_mask = layer_mask[:total_layers]

            # 设置激活的子网
            elastic_model.set_active_subnet(
                h_ratio=h_ratio, 
                ha_ratio=ha_ratio, 
                intermediate_ratio=inter_ratio, 
                layer_mask=layer_mask
            )

            # 准备输入数据
            input_ids = batch["input_ids"].to(target_device)
            labels = input_ids.clone().to(target_device)

            # 模型前向传播
            logits, loss = elastic_model(input_ids, labels)
            
            # 总损失 = LM 损失 + 预算惩罚
            loss = loss + loss_penalty.to(loss.device) * 0.1 # 0.1 是惩罚权重
            loss_to_backprop = loss / GRADIENT_ACCUMULATION_STEPS
            loss_to_backprop.backward()

            running_loss += loss.item()

            if step % GRADIENT_ACCUMULATION_STEPS == 0:
                # 梯度累积步完成，执行优化
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                router.anneal_tau() # 退火 Gumbel-Softmax 温度

                global_step += 1
                avg_loss = running_loss / GRADIENT_ACCUMULATION_STEPS
                print(f"[STEP] Epoch {epoch+1}/{NUM_TRAINING_EPOCHS} | global_step={global_step:4d} | budget={current_budget:.2f} | loss={avg_loss:.4f} | tau={router.get_tau():.4f} | subnet H={h_ratio:.3f} HA={ha_ratio:.3f} I={inter_ratio:.3f} depth={int(sum(layer_mask))}/{total_layers}")
                running_loss = 0.0
            else:
                 # 梯度累积进行中
                 print(f"[ACC] Epoch {epoch+1} step={step} (accumulating) | budget={current_budget:.2f} | loss={loss.item():.4f}")

    print("[INFO] Training finished.")

    # --- 6. 保存检查点 ---
    try:
        save_dict = {
            "model_state_dict": {k: v.cpu() for k, v in elastic_model.state_dict().items()},
            "router_state_dict": {k: v.cpu() for k, v in router.state_dict().items()},
            "tokenizer": None, # 分词器无需保存到这里
        }
        torch.save(save_dict, CKPT_PATH)
        print(f"[INFO] Checkpoint saved to {CKPT_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to save checkpoint: {e}")

    # --- 7. 模型评估 ---
    print("\n===============================")
    print("=== STARTING MODEL EVALUATION ===")
    print("===============================\n")

    # 复用训练时的 loader 作为测试集 (仅为演示)
    test_loader = loader 
    
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
    print("-------------------------------------")
    print("{:<10} | {:<12} | {:<10} | {:<5}".format("Budget", "Param Frac", "Loss", "PPL"))
    print("-------------------------------------")
    for budget, res in results.items():
        print("{:<10.2f} | {:<12.3f} | {:<10.4f} | {:<5.2f}".format(
            budget, res['param_frac'], res['loss'], res['ppl']
        ))
    print("-------------------------------------")
