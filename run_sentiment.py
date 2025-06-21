#!/usr/bin/env python
# coding=utf-8
# 5级情感分析微调脚本

import logging
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import torch.cuda as cuda
from typing import Optional
from matplotlib import rcParams
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPTNeoXTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, f1_score

from arguments import ModelArguments, DataTrainingArguments
from data_utils import load_and_split_data
from model_utils import load_model
from train_utils import setup_trainer, train_and_evaluate, benchmark_model, compute_metrics
from visualization_utils import plot_comparison
from trainer_mamba import MambaForSentimentAnalysis, MambaTrainer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from model_utils import setup_environment
from data_utils import preprocess_function
logger = logging.getLogger(__name__)


def main(model_args=None, data_args=None, training_args=None):
    """主执行函数"""
    torch.backends.cuda.max_split_size_mb = 64  # 减少内存碎片
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    setup_environment()
    
    # 解析参数
    if model_args is None or data_args is None or training_args is None:
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置缓存目录
    if model_args.cache_dir is None:
        model_args.cache_dir = os.path.join(os.getcwd(), "cache")
        os.makedirs(model_args.cache_dir, exist_ok=True)

    # 加载数据和模型
    raw_datasets, tokenized_datasets = load_and_split_data(data_args, model_args)
    tokenizer, model = load_model(model_args, data_args, device)
    
    # 分割数据集
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets.get("validation")
    test_dataset = tokenized_datasets.get("test")

    # 分别预处理三个数据集
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, data_args, training_args, model),
        batched=True,
        remove_columns=[col for col in train_dataset.column_names if col not in ["input_ids", "labels", "attention_mask"]]
    )
    
    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, data_args, training_args, model),
        batched=True,
    ) if eval_dataset else None
    
    tokenized_test = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer, data_args, training_args, model),
        batched=True,
    ) if test_dataset else None

    # 重新组合成DatasetDict后添加验证
    tokenized_datasets = {
        "train": tokenized_train,
        "validation": tokenized_eval,
        "test": tokenized_test
    }
    trainer = setup_trainer(
        model=model,
        training_args=training_args,
        tokenized_datasets=tokenized_datasets,
        compute_metrics=compute_metrics
    )

    # 调用train_and_evaluate执行训练和评估
    eval_results = train_and_evaluate(
        trainer=trainer,
        tokenized_datasets=tokenized_datasets,
        training_args=training_args
    )

    return model, tokenized_datasets, eval_results # 返回模型、tokenized数据集和评估结果


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 硬编码参数设置
    class Args:
        #model_name_or_path = r"E:\project\JD_sentiment\MiniRBT-h256"  # 使用哈工大模型
        train_file = r"E:\project\JD_sentiment\sentiment_data.json"
        validation_file = None  # 显式设置为None
        output_dir = "./sentiment_model/"
        do_train = True
        do_eval = True 
        num_labels = 5
        per_device_train_batch_size = 32  # 增大batch size提高利用率
        gradient_accumulation_steps = 2  # 平衡内存和效率
        max_seq_length = 128  # 适当减小序列长度
        fp16 = True
        gradient_checkpointing = False
        optim = "adamw_torch_fused"
        torch_compile = True  # 启用模型编译优化
        num_train_epochs = 2 # 添加训练轮数参数
        learning_rate = 2e-5
        use_custom_trainer = False  # 添加此参数控制是否使用自定义Trainer
    
    # 测试两种模型
    comparison_results = {}
    for model_name in [
        r"E:\project\JD_sentiment\mamba-130m",
        r"E:\project\JD_sentiment\bert_base",  # BERT模型
        #r"E:\project\JD_sentiment\mamba-130m",  # Mamba模型
    ]: 
        # 为每个模型创建单独的输出目录
        model_output_dir = os.path.join(Args.output_dir, os.path.basename(model_name))
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 将参数转换为HuggingFace需要的格式
        model_args = ModelArguments(model_name_or_path = model_name)
        data_args = DataTrainingArguments(
            train_file=Args.train_file,
            validation_file=Args.validation_file, 
            num_labels=Args.num_labels
        )
        training_args = TrainingArguments(
            output_dir=model_output_dir,
            do_train=Args.do_train,
            do_eval=Args.do_eval,
            per_device_train_batch_size=Args.per_device_train_batch_size,
            gradient_accumulation_steps=Args.gradient_accumulation_steps,
            num_train_epochs=Args.num_train_epochs,
            learning_rate=Args.learning_rate,
            fp16=torch.cuda.is_available(),      
            fp16_full_eval=False,
            bf16=False,
            use_cpu=not torch.cuda.is_available(),
            max_grad_norm=1.0,
            report_to=["none"],
            logging_strategy="steps",
            logging_steps=100,
            save_strategy="epoch",
            no_cuda=not torch.cuda.is_available(),
            gradient_checkpointing=False,
            optim = "adamw_torch_fused" , # 使用融合优化器
            fp16_opt_level="O1",  # 优化级别
        )
    
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
               
        # 调用主函数并接收模型、数据集和评估结果
        model, tokenized_datasets, eval_results = main(model_args, data_args, training_args)
        

        # 添加benchmark测试
        if tokenized_datasets.get("validation"):
            benchmark = benchmark_model(model, tokenizer, tokenized_datasets["validation"][:100], device)

        comparison_results[model_name] = {
            "accuracy": eval_results.get("eval_accuracy", 0.0),
            "macro_f1": eval_results.get("eval_macro_f1", 0.0),
            "speed": benchmark.get("speed", 0),
            "memory_usage": benchmark.get("memory_usage", 0),
            "warning": benchmark.get("warning", "")
        }
        
        # 保存单个模型结果
        model_save_path = os.path.join(model_output_dir, "model_comparison.png")
        plot_comparison({model_name: comparison_results[model_name]}, save_path=model_save_path)
        
        # 保存模型和tokenizer
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        
    # 保存整体比较结果
    plot_comparison(comparison_results, save_path=os.path.join(Args.output_dir, "all_models_comparison.png"))