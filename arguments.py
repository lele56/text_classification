from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "模型路径或HuggingFace模型名称"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "缓存目录"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用快速tokenizer"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "模型版本"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "torch数据类型",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

@dataclass
class DataTrainingArguments:
    train_file: str = field(
        default=None, 
        metadata={"help": "训练数据路径"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "验证数据路径"}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "测试数据路径"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )
    num_labels: int = field(
        default=5,
        metadata={"help": "情感分类级别数(1-5)"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "是否覆盖缓存"}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "是否填充到最大长度"}
    )
