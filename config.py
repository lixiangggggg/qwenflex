# config.py

import os

# --- Model & Path Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_PATH = "reordered_qwen"
# 假设您的数据路径保持不变
DATA_PATH = "/home/lx/my_qwen2.5_0.5B/my_txt.txt"
SAVE_DIR = "elastic_qwen_output"
CKPT_PATH = "elastic_qwen_checkpoint.pt"

# --- Training Configuration ---
SEED = 1234
DEVICE_ID = 2  # cuda:2
BATCH_SIZE = 2
SEQ_LEN = 128
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAINING_EPOCHS = 3
TRAINING_BUDGETS = [1.0, 0.9, 0.7, 0.5, 0.3]
EVAL_BUDGETS = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2]
LEARNING_RATE = 1e-5
