import os
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from datasets import load_dataset, DatasetDict
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Any
from safetensors.torch import save_file
import logging
logger = logging.getLogger(__name__)
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.trainer_utils import EvalPrediction
# 添加缺失的常量定义
PREFIX_CHECKPOINT_DIR = "checkpoint"
TRAINING_ARGS_NAME = "training_args.bin"
# 添加nested_detach的备用实现
def nested_detach(tensors):
    """Detach tensors in nested structure"""
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: nested_detach(v) for k, v in tensors.items()}
    elif isinstance(tensors, torch.Tensor):
        return tensors.detach()
    return tensors



class MambaForSentimentAnalysis(nn.Module):
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        """直接使用已加载的模型，避免重复加载"""
        model = cls(pretrained_model_name, **kwargs)
        if device is not None:
            model.to(device)
        if dtype is not None:
            model.to(dtype)
        return model
    def __init__(self, base_model_name: str, num_labels: int = 5):
        super().__init__()
        self.base_model = MambaLMHeadModel.from_pretrained(base_model_name)
        self.base_model.return_hidden_states = True
        
        # 添加config属性
        self.config = self.base_model.config
        # 彻底解耦合权重共享
        self.base_model.lm_head.weight = nn.Parameter(
            self.base_model.backbone.embedding.weight.detach().clone()
        )
        
        # 获取模型的隐藏维度
        hidden_size = self.base_model.config.d_model 
        
        # 添加分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels)
        )
        self._init_classifier_weights()
        self.num_labels = num_labels
        self.gradient_checkpointing = True

    def _init_classifier_weights(self):
        """增强版权重初始化"""
        for name, layer in self.classifier.named_children():
            if isinstance(layer, nn.Linear):
                # 使用正确的非线性函数名称
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                logger.info(f"初始化 {name}.weight - 均值: {layer.weight.mean().item():.4f}, 标准差: {layer.weight.std().item():.4f}")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)
                    logger.info(f"初始化 {name}.bias - 均值: {layer.bias.mean().item():.4f}")
    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None, return_dict=True):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        #print(f"前向传播前内存: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # 强制释放中间变量
            with torch.no_grad():
                # 直接运行base_model检查原始输出
                raw_outputs = self.base_model(input_ids)
                logger.debug(f"Raw outputs stats - min: {raw_outputs[0].min()}, max: {raw_outputs[0].max()}")
                
                if self.training and self.gradient_checkpointing:
                    outputs = torch.utils.checkpoint.checkpoint(
                        self.base_model,
                        input_ids,
                        use_reentrant=True,  # 改为True确保正确性
                        preserve_rng_state=True
                    )
                else:
                    outputs = self.base_model(input_ids)
        
        
        # 保存outputs引用用于后续检查
        outputs_ref = outputs
        
        # 获取隐藏状态
        if hasattr(outputs_ref, 'hidden_states'):
            hidden_states = outputs_ref.hidden_states[-1]
        else:
            hidden_states = outputs_ref[0][..., :self.base_model.config.d_model]
        
        # 现在可以安全删除outputs
        del outputs
        torch.cuda.empty_cache()
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # 池化策略

        pooled = hidden_states.mean(dim=1)
            
        # 添加调试信息
        logger.debug(f"Pooled output stats - mean: {pooled.mean().item():.4f}, std: {pooled.std().item():.4f}")
        
        # 确保形状正确 [batch_size, hidden_dim]
        if pooled.size(1) != hidden_dim:
            pooled = pooled.view(batch_size, hidden_dim)
            
        # 强制类型转换并添加梯度检查
        pooled = pooled.to(self.classifier[1].weight.dtype)
        pooled.requires_grad_(True)  # 确保梯度能传播
        
        # 检查分类器权重
        for name, param in self.classifier.named_parameters():
            if param.requires_grad:
                logger.info(f"参数 {name} - 均值: {param.mean().item():.4f}, 梯度: {'存在' if param.grad is not None else '无'}")
                if param.grad is not None:
                    logger.info(f"梯度均值: {param.grad.mean().item():.4f}")
        
        logits = self.classifier(pooled)
        
        # 计算损失
        loss = None
        if labels is not None:

            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # 确保loss不为零
            if loss.item() == 0:
                logger.warning("计算得到的损失为0，可能存在错误")
        # 确保返回的字典包含所有必要字段
        output = {
            'logits': logits,
            'loss': loss,
            'hidden_states': outputs_ref.hidden_states if hasattr(outputs_ref, 'hidden_states') else None,
        }
        
        # 确保loss不为None
        if output['loss'] is None:
            output['loss'] = torch.tensor(0.0, device=logits.device)
        
        return output if return_dict else (logits,)
    def save_pretrained(self, save_directory):
        """保存模型到指定目录"""
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存模型权重
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        
        # 保存配置
        if hasattr(self, 'config'):
            config_dict = {k: v for k, v in self.config.__dict__.items() 
                         if not k.startswith('_') and not callable(v)}
            torch.save(config_dict, os.path.join(save_directory, "config.pt"))

class MambaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # 移除tokenizer参数以避免弃用警告
        if 'tokenizer' in kwargs:
            kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        # 完全禁用可能导致问题的Trainer默认行为
        self.args.remove_unused_columns = False
        self.args.include_inputs_for_metrics = False
        self.args.ddp_find_unused_parameters = False
        self.args.disable_tqdm = False
        self.eval_progress_bar = None
        self.predict_progress_bar = None


    def pad_across_processes(self, tensor, dim=0, pad_index=0, pad_first=False):
        """重写pad_across_processes方法，使其能处理None值"""
        if tensor is None:
            return None
        return super().pad_across_processes(tensor, dim=dim, pad_index=pad_index, pad_first=pad_first)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 禁用内部进度条
        self.args.disable_tqdm = False

        try:
            # 设置评估进度条
            self.eval_progress_bar = tqdm(
                total=len(self.get_eval_dataloader(eval_dataset)),
                desc="评估进度",
                leave=True
            )
            model = self._wrap_model(self.model, training=False)
            model.eval()
            
            all_preds = []
            all_labels = []
            total_loss = 0.0
            batch_count = 0
            
            for batch in self.get_eval_dataloader(eval_dataset):
                # 更新进度条
                self.eval_progress_bar.update(1)
                # 添加批次调试信息
                logger.debug(f"评估批次内容: {batch.keys()}")
                
                # 执行预测
                loss, logits, labels = self.prediction_step(model, batch, prediction_loss_only=False)
                
                # 记录损失值
                if loss is not None:
                    total_loss += loss.item()
                    batch_count += 1
                # 处理logits为元组的情况
                if isinstance(logits, tuple):
                    logits = logits[0]  # 取第一个元素作为logits
                # 检查logits
                if logits is None:
                    logger.error("获取到None logits!")
                    continue
                    
                # 添加调试信息
                logger.debug(f"logits形状: {logits.shape}")
                
                all_preds.append(logits.cpu())
                if labels is not None:
                    all_labels.append(labels.cpu())
            
            # 检查预测结果
            if not all_preds:
                logger.error("评估未产生任何有效预测结果")
                return {}

            # 确保预测结果正确聚合
            eval_pred = EvalPrediction(
                predictions=torch.cat(all_preds).numpy(),
                label_ids=torch.cat(all_labels).numpy() if all_labels else None
            )
        
            # 计算指标
            metrics = self.compute_metrics(eval_pred)
            if batch_count > 0:
                metrics["loss"] = total_loss / batch_count
            
            
            return {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
    

        finally:
            # 清理进度条资源
            if self.eval_progress_bar is not None:
                self.eval_progress_bar.close()
                self.eval_progress_bar = None
        
    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        # 禁用内部进度条
        original_disable_tqdm = self.args.disable_tqdm
        self.args.disable_tqdm = True
        
        try:
            # 设置预测进度条
            self.predict_progress_bar = tqdm(
                total=len(self.get_test_dataloader(test_dataset)),
                desc="预测进度",
                leave=True
            )
            
            model = self._wrap_model(self.model, training=False)
            model.eval()
            
            all_preds = []
            all_labels = []
            
            for batch in self.get_test_dataloader(test_dataset):
                # 执行预测
                _, logits, labels = self.prediction_step(model, batch, prediction_loss_only=False)
                
                # 处理logits为元组的情况
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                if logits is not None:
                    all_preds.append(logits.cpu())
                if labels is not None:
                    all_labels.append(labels.cpu())
                    
                # 更新进度条
                self.predict_progress_bar.update(1)
            
            # 检查预测结果
            if not all_preds:
                logger.error("未产生任何有效预测结果")
                return PredictionOutput(predictions=None, label_ids=None, metrics={})
            
            # 返回标准格式的预测结果
            return PredictionOutput(
                predictions=torch.cat(all_preds).numpy(),
                label_ids=torch.cat(all_labels).numpy() if all_labels else None,
                metrics={}
            )
            
        finally:
            # 清理资源
            self.args.disable_tqdm = original_disable_tqdm
            if self.predict_progress_bar is not None:
                self.predict_progress_bar.close()
                self.predict_progress_bar = None
    # def evaluation_loop(self, *args, **kwargs):
    #     """重写评估循环，确保正确处理None值"""
    #     try:
    #         return super().evaluation_loop(*args, **kwargs)
    #     except TypeError as e:
    #         if "Unsupported types (<class 'NoneType'>)" in str(e):
    #             logger.warning("评估过程中遇到None值，已跳过处理")
    #             return None
    #         raise

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # 确保输出目录存在
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取模型状态并处理共享权重
        if state_dict is None:
            state_dict = self.model.state_dict()
            state_dict['base_model.lm_head.weight'] = state_dict['base_model.backbone.embedding.weight'].clone()
        
        # 保存模型权重
        if self.args.save_safetensors:
            save_file(
                state_dict, 
                os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                metadata={"format": "pt"}
            )
        else:
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        
        # 保存配置（兼容MambaConfig）
        if hasattr(self.model.config, 'save_pretrained'):
            self.model.config.save_pretrained(output_dir)
        else:
            # 手动保存配置
            config_dict = {k: v for k, v in self.model.config.__dict__.items() 
                        if not k.startswith('_') and not callable(v)}
            torch.save(config_dict, os.path.join(output_dir, 'config.pt'))
        
        # 修改tokenizer保存逻辑
        if hasattr(self, 'processing_class') and self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif hasattr(self, 'tokenizer') and self.tokenizer is not None:  # 保持向后兼容
            self.tokenizer.save_pretrained(output_dir)


   



