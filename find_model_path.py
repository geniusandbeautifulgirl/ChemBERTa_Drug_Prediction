from transformers import AutoTokenizer, AutoModel
import os
from pathlib import Path

# 先运行一次，让模型自动缓存到本地（你之前跑过，已经有了）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "seyonec/ChemBERTa-zinc250k-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 打印缓存路径
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
print(f"模型缓存根目录：{cache_dir}")
print(f"ChemBERTa模型完整路径：{cache_dir / 'models--seyonec--ChemBERTa-zinc250k-v1'}")