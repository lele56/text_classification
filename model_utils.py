import os
import sys
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from trainer_mamba import MambaForSentimentAnalysis
from matplotlib import rcParams
import logging
from safetensors.torch import load_file
logger = logging.getLogger(__name__)
def load_model(model_args, data_args, device):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    if "mamba" in model_args.model_name_or_path.lower():
        model = MambaForSentimentAnalysis(
            base_model_name=model_args.model_name_or_path,
            num_labels=data_args.num_labels
        )
        
        if os.path.isdir(model_args.model_name_or_path):
            # 优先尝试加载 pytorch_model.bin
            model_path = os.path.join(model_args.model_name_or_path, "pytorch_model.bin")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path)
            else:
                # 如果不存在则尝试加载 model.safetensors
                model_path = os.path.join(model_args.model_name_or_path, "model.safetensors")
                if os.path.exists(model_path):
                    state_dict = load_file(model_path)
                    
                    # 验证权重是否独立
                    if ('base_model.backbone.embedding.weight' in state_dict and 
                        'base_model.lm_head.weight' in state_dict and
                        not torch.equal(state_dict['base_model.backbone.embedding.weight'],
                                    state_dict['base_model.lm_head.weight'])):
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        raise ValueError("检测到权重共享问题，请检查模型保存逻辑")
            
            model.to(device)
    else:
        # 原有BERT模型加载逻辑保持不变
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=data_args.num_labels
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config
        )
        model.to(device)
    
    return tokenizer, model

def setup_environment():
    """初始化运行环境"""
    # 设置中文字体
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    rcParams['axes.unicode_minus'] = False
    
    # 初始化日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )