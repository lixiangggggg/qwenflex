# data_utils.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class LocalTextDataset(Dataset):
    """
    一个简单的文本数据集，用于从本地文件加载文本行并进行分词。
    """
    def __init__(self, file_path, tokenizer: AutoTokenizer, max_length=128):
        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        # 注意: 原始代码中没有返回 'labels'，但为了训练，我们通常使用 input_ids 作为 labels
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Squeeze(0) 是为了移除 batch 维度
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
