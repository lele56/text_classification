from datasets import DatasetDict, load_dataset
import re
import logging
import torch 
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from trainer_mamba import MambaForSentimentAnalysis
logger = logging.getLogger(__name__)


def clean_text(text):
    """清理文本中的不明确Unicode字符并去除空白文本"""
    if not text or not isinstance(text, str) or not text.strip():
        return None  # 返回None表示空白文本或非字符串
    
    # 保留中文、英文、数字、常用标点和空格
    pattern = re.compile(
        r'[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a'
        r'\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001'
        r'\uff1f\u300a\u300b\u3008\u3009\u0020\u2018\u2019\u2013\u2014]'
    )
    cleaned = pattern.sub('', text)
    
    # 合并连续空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else None

def load_and_split_data(data_args, model_args):
    """加载并自动划分数据集为训练/验证/测试集"""
    # 加载原始数据集
    if data_args.validation_file is None and data_args.train_file is not None:
        # 只有训练数据时自动划分
        raw_datasets = load_dataset(
            "json", 
            data_files={"train": data_args.train_file},
            cache_dir=model_args.cache_dir
        )
        
        # 清理文本并过滤掉空文本和无效标签
        cleaned_datasets = raw_datasets.map(
            lambda example: {
                "text": clean_text(example.get("text", "")),
                "label": example.get("label", None)
            },
            batched=False
        ).filter(lambda x: x["text"] is not None and x["label"] is not None)  # 同时检查文本和标签
        
        # 划分数据集 (80%训练, 10%验证, 10%测试)
        splits = cleaned_datasets["train"].train_test_split(
            test_size=0.2, shuffle=True, seed=42
        )
        test_valid = splits["test"].train_test_split(
            test_size=0.5, shuffle=True, seed=42
        )
        
        tokenized_datasets = DatasetDict({
            "train": splits["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"]
        })
        
    else:
        # 已有划分的数据集
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file 
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        
        # 清理文本
        tokenized_datasets = raw_datasets.map(
            lambda example: {"text": clean_text(example["text"])},
            batched=False
        )

    # 确保测试集存在
    if "test" not in tokenized_datasets:
        logger.warning("警告: 数据集不包含测试集, 将使用验证集作为测试集")
        tokenized_datasets["test"] = tokenized_datasets["validation"]
    
    return raw_datasets,tokenized_datasets

def preprocess_function(examples, tokenizer, data_args, training_args, model):
    """优化后的预处理函数"""
    # 统一过滤逻辑
    valid_indices = []
    for i in range(len(examples["text"])):
        text = examples["text"][i]
        label = examples["label"][i]
        
        # 添加空字符串检查
        if (text is not None and 
            isinstance(text, str) and 
            text.strip() != "" and  # 确保不是空字符串
            label is not None and
            1 <= label <= 5):
            valid_indices.append(i)
    
    if not valid_indices:
        raise ValueError("所有输入数据均为空或无效")
    
    examples["text"] = [examples["text"][i] for i in valid_indices]
    examples["label"] = [examples["label"][i] for i in valid_indices]
    
    is_mamba = isinstance(model, MambaForSentimentAnalysis)
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=min(64, data_args.max_seq_length),
        padding='max_length',  # 确保填充到固定长度
        return_tensors="pt",
        return_attention_mask= True  # Mamba不需要attention_mask
    )
    
    # 确保标签格式正确
    labels = torch.tensor(examples["label"], dtype=torch.long) - 1  # 转换为0-4
    
    # 构建结果字典
    result = {
        'input_ids': inputs['input_ids'],
        'labels': labels
    }
    
    # 仅当不是Mamba模型时才添加attention_mask
    if not is_mamba:
        result['attention_mask'] = inputs['attention_mask']
    
    return result