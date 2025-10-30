import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset as HFDataset 
from typing import Optional, Dict, Any, List
import sys
import os

# --------------------------------------------------------------------------------
# 1. PyTorch Dataset Wrapper (保持不变，确保兼容 labels)
# --------------------------------------------------------------------------------

class TokenizedTextDataset(Dataset):
    """
    用于 PyTorch DataLoader 消费的最终 Tokenized Dataset。
    它封装 Hugging Face Datasets，并处理 input_ids, attention_mask, 和 labels。
    """
    def __init__(self, hf_dataset: HFDataset):
        self.dataset = hf_dataset

    def __len__(self):
        if hasattr(self.dataset, '__len__'):
            return len(self.dataset)
        else:
            print("[WARN] Dataset length is unavailable in streaming mode.")
            return 0 

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 将列表转换为 PyTorch Tensor
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item.get("attention_mask", [1] * len(input_ids)), dtype=torch.long)
        
        # SFT 模式下必须有 labels
        labels = torch.tensor(item["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask, 
            "labels": labels
        }

# --------------------------------------------------------------------------------
# 2. SFT 数据处理核心函数 (新增)
# --------------------------------------------------------------------------------

def prepare_sft_data(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    instruction_column: str = "instruction",
    response_column: str = "response",
    split: str = "train",
    max_length: int = 512,
    num_proc: int = 8,
    batch_size: int = 4,
    cache_dir: Optional[str] = None
) -> DataLoader:
    """
    从 Hugging Face Hub 下载 SFT 语料库，进行格式化、分词和损失掩码处理，
    并返回一个 PyTorch DataLoader。

    Args:
        tokenizer: 模型的 AutoTokenizer 实例。
        dataset_name: 要下载的 Hugging Face SFT 语料库名称 (e.g., 'tatsu-lab/alpaca').
        instruction_column: 数据集中指令所在的列名 (e.g., 'instruction').
        response_column: 数据集中响应所在的列名 (e.g., 'output').
        split: 使用的数据集分区 ('train', 'validation', 'test')。
        max_length: 序列的最大长度。
        num_proc: 用于并行化 Tokenization 的进程数。
        batch_size: DataLoader 的批次大小。
        cache_dir: 数据集的缓存目录。
    
    Returns:
        PyTorch DataLoader: 准备好的 SFT 数据加载器。
    """
    print(f"[INFO] Starting SFT data preparation for: {dataset_name}")
    
    # 1. 下载和加载数据集
    try:
        raw_datasets = load_dataset(
            dataset_name, 
            split=split,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
    except Exception as e:
        print(f"[FATAL] Failed to load dataset {dataset_name}. Error: {e}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(raw_datasets)} raw documents.")

    # 2. SFT 核心：格式化、分词和损失掩码
    def tokenize_and_mask_function(examples: Dict[str, List[Any]]):
        
        # 结果字典
        results = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for instruction, response in zip(examples[instruction_column], examples[response_column]):
            # 构建完整的输入序列 (Qwen 的格式通常包含 <|im_start|> 和 <|im_end|> 等特殊 token)
            # 这里我们使用一个通用的格式作为示例
            full_text = f"Instruction: {instruction}\nResponse: {response}{tokenizer.eos_token}"
            instruction_text = f"Instruction: {instruction}\nResponse: " # 仅指令部分（用于确定掩码长度）
            
            # 1. 对完整序列进行分词
            full_tokenized = tokenizer(
                full_text, 
                max_length=max_length, 
                truncation=True, 
                padding=False
            )
            
            # 2. 对指令部分进行分词，用于确定需要掩码的长度
            instruction_tokenized = tokenizer(
                instruction_text,
                max_length=max_length, 
                truncation=True, 
                padding=False
            )
            
            input_ids = full_tokenized["input_ids"]
            attention_mask = full_tokenized["attention_mask"]
            
            # 3. 创建 Labels 并应用掩码
            # SFT 中，Labels 初始等于 Input IDs
            labels = input_ids.copy()
            
            # 确定指令部分的长度（注意：要防止越界）
            # instruction_len 是不计算损失的部分
            instruction_len = len(instruction_tokenized["input_ids"]) 
            
            # 将指令部分的 labels 设置为 -100 (Loss Masking)
            # 模型的损失只在 response token 上计算
            labels[:instruction_len] = [-100] * instruction_len
            
            # 4. 填充至 max_length（可选，但通常在 SFT 中保持一致的序列长度）
            padding_len = max_length - len(input_ids)
            if padding_len > 0:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                
                input_ids += [pad_id] * padding_len
                attention_mask += [0] * padding_len
                # Labels 的 padding 部分也必须是 -100
                labels += [-100] * padding_len
            
            results["input_ids"].append(input_ids)
            results["attention_mask"].append(attention_mask)
            results["labels"].append(labels)

        return results

    # 3. 对数据集进行处理
    processed_datasets = raw_datasets.map(
        tokenize_and_mask_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_datasets.column_names,
    )
    
    print(f"[INFO] Final dataset size after processing: {len(processed_datasets)} blocks.")

    # 4. 转换为 PyTorch Dataset
    pytorch_dataset = TokenizedTextDataset(processed_datasets)

    # 5. 创建 PyTorch DataLoader
    data_loader = DataLoader(
        pytorch_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True 
    )
    
    print(f"[INFO] SFT DataLoader successfully created with Batch Size: {batch_size}.")
    return data_loader

# --------------------------------------------------------------------------------
# 示例用法 (用于测试该文件独立性)
# --------------------------------------------------------------------------------

if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    # 假设你的模型名称
    MODEL_NAME = "Qwen/Qwen2.5-0.5B" 
    
    # SFT 数据集示例：tatsu-lab/alpaca (英文) 或 yiqinzhong/alpaca_chinese (中文)
    SFT_DATASET_NAME = "tatsu-lab/alpaca" 
    INSTRUCTION_COL = "instruction"
    RESPONSE_COL = "output"
    SFT_SPLIT = "train" 
    
    SEQ_LEN = 256 # SFT 序列长度通常比预训练长
    BATCH_SIZE = 2
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer {MODEL_NAME}: {e}")
        sys.exit(1)
        
    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
    print(f"[INFO] Tokenizer loaded. Pad token ID: {tokenizer.pad_token_id}")

    try:
        sft_loader = prepare_sft_data(
            tokenizer=tokenizer,
            dataset_name=SFT_DATASET_NAME,
            instruction_column=INSTRUCTION_COL,
            response_column=RESPONSE_COL,
            split=SFT_SPLIT,
            max_length=SEQ_LEN,
            num_proc=os.cpu_count() or 1,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        print(f"[FATAL] SFT data preparation failed: {e}")
        sys.exit(1)


    print("\n--- Testing SFT DataLoader Output (Checking Loss Masking) ---")
    
    # 取第一个批次进行检查
    batch = next(iter(sft_loader))
    
    sample_input_ids = batch['input_ids'][0].tolist()
    sample_labels = batch['labels'][0].tolist()
    
    print(f"Batch Input IDs Shape: {batch['input_ids'].shape}")
    print(f"Batch Labels Shape: {batch['labels'].shape}")
    
    # 查找第一个非 -100 的标签索引，这是损失开始计算的位置
    loss_start_index = next((i for i, label in enumerate(sample_labels) if label != -100), -1)

    if loss_start_index != -1:
        # 解码并展示掩码情况
        instruction_masked = tokenizer.decode(sample_input_ids[:loss_start_index], skip_special_tokens=True)
        response_unmasked = tokenizer.decode(sample_input_ids[loss_start_index:], skip_special_tokens=True)
        
        print(f"\n[SUCCESS] Loss Masking Applied:")
        print(f"  Instruction (Loss Masked Part): {instruction_masked[:50]}...")
        print(f"  Response (Loss Calculated Part): {response_unmasked[:50]}...")
        print(f"  Loss Starts at Index: {loss_start_index}")
        
    else:
        print("[ERROR] Could not find unmasked labels (-100). Check data structure.")
