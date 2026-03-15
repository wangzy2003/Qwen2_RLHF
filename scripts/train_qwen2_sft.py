import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # 禁用 TensorFlow 支持
os.environ["TRANSFORMERS_NO_FLAX"] = "1" # 顺便禁用 Flax，精简依赖

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

class SFTJsonDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length: int = 512):
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.examples = []
        for item in data:
            prompt = item["prompt"]
            response = item["response"]
            # 简单拼接成单轮对话格式
            text = f"用户: {prompt}\n助手: {response}"
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
            )
            # 这里直接让模型学习整句，简单起见不区分 prompt/label
            tokenized["labels"] = tokenized["input_ids"].copy()
            self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


if __name__ == "__main__":
    # # 模型路径
    # model_path = r"D:/毕业论文/OpenRLHF/models/Qwen2-0.5B"
    # # 数据路径
    # train_dataset_path = r"D:/毕业论文/OpenRLHF/data/sft_data.json"
    # # 输出路径
    # output_dir = r"D:/毕业论文/OpenRLHF/qwen2_0.5b_sft_cpu"
    model_path = r"./models/Qwen2-0.5B"
    train_dataset_path = r"./data/sft_data.json"
    output_dir = r"./qwen2_0.5b_sft_cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # 自动选择设备：有 GPU 用 GPU，没有就用 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_dataset = SFTJsonDataset(train_dataset_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # 可以先用小 epoch 在 CPU 上调试
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=1000,
        save_total_limit=1,
        #no_cuda=not torch.cuda.is_available(),  # 没有 GPU 时强制走 CPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,  # 用这个替代 tokenizer
    )

    trainer.train()