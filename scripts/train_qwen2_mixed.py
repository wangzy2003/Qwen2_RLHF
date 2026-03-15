"""
SFT 与 RL（PPO）交替训练：同一模型先做若干步 SFT，再做若干步 PPO，循环多轮。
支持 CPU，但会较慢，每轮步数较小（如 sft_steps=2, ppo_steps=2, num_rounds=2）。
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import json
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------- 数据：SFT 用 prompt+response，PPO 只用 prompt ----------
class SFTJsonDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length: int = 512):
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.examples = []#存每条样本的tokenized结果
        for item in data:#遍历每条样本
            prompt = item["prompt"]#获取prompt
            response = item["response"]#获取response
            text = f"用户: {prompt}\n助手: {response}"#拼接成单轮对话格式
            tokenized = tokenizer(text, truncation=True, max_length=max_length)#分词、截断到 max_length，得到 input_ids 等
            tokenized["labels"] = tokenized["input_ids"].copy()#labels就是input_ids，因果 LM：标签=输入序列，用于计算下一个 token 的 loss
            self.examples.append(tokenized)#存入列表

    def __len__(self):
        return len(self.examples)#返回数据集长度

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}#按索引取一条样本，并把列表转成 tensor 字典（input_ids, attention_mask, labels）


class PromptDataset(Dataset):
    def __init__(self, json_path: str):#只给 PPO 用：每条样本就是一个 prompt 字符串
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))#只保留 prompt 字段，不读 response
        self.prompts = [item["prompt"] for item in data]#存每条样本的 prompt

    def __len__(self):
        return len(self.prompts)#返回数据集长度

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}#按索引取一条样本，返回字典（prompt）


# ---------- PPO 用：reward、生成、logprob、单步更新 ----------
def compute_reward(prompt: str, response: str) -> float:#输入 prompt 和 response 文本，输出一个标量奖励
    length = len(response)#计算 response 长度
    if length < 20:
        length_score = -1.0
    elif length > 200:
        length_score = -1.0
    else:
        length_score = 1.0
    keywords = ["RLHF", "SFT", "强化学习"]
    keyword_hits = sum(1 for kw in keywords if kw in response)#统计 response 中包含的关键词数量
    keyword_score = 0.5 * keyword_hits#关键词得分
    bad_words = ["脏话1", "脏话2"]
    bad_hits = sum(1 for bw in bad_words if bw in response)#统计 response 中包含的脏话数量
    bad_score = -1.0 * bad_hits#脏话得分
    return float(length_score + keyword_score + bad_score)#返回总得分


@torch.no_grad()
def generate_responses(model, tokenizer, prompts: List[str], device: torch.device,
                       max_input_length: int = 256, max_new_tokens: int = 64):#用当前模型对一批 prompt 生成回复
    model.eval()#模型评估模式
    all_input_ids, all_attention_mask, prompt_lens = [], [], []#存每个样本的生成序列、mask、以及 prompt 长度
    for prompt in prompts:#遍历每条 prompt
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length)#
        input_ids = enc["input_ids"]#获取 input_ids
        attention_mask = enc["attention_mask"]#获取 attention_mask
        prompt_len = input_ids.size(1)#获取 prompt 长度
        output_ids = model.generate(#生成回复
            input_ids=input_ids.to(device),#转成 device 上 tensor
            attention_mask=attention_mask.to(device),#转成 device 上 tensor
            max_new_tokens=max_new_tokens,#最大生成 tokens 数
            do_sample=True,#采样
            temperature=1.0,#温度
            top_p=0.9,#top-p 采样
        )
        all_input_ids.append(output_ids.cpu())#存生成序列
        all_attention_mask.append((output_ids != tokenizer.pad_token_id).long())#存 mask
        prompt_lens.append(prompt_len)#存 prompt 长度
    input_ids = torch.cat(all_input_ids, dim=0)#拼接成 batch tensor
    attention_mask = torch.cat(all_attention_mask, dim=0)#拼接成 batch tensor
    texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)#解码成文本
    return input_ids, attention_mask, prompt_lens, texts#返回 input_ids, attention_mask, prompt_lens, texts


def sequence_logprob(model, input_ids, attention_mask, prompt_lens: List[int], device: torch.device):
    model.eval()
    outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    batch_size, seq_len, _ = log_probs.size()
    logprobs_per_sample = []
    for i in range(batch_size):
        pl = prompt_lens[i]
        if pl >= seq_len - 1:
            logprobs_per_sample.append(torch.tensor(-10.0, device=device))
            continue
        resp_input_ids = input_ids[i, pl + 1 :].to(device)
        resp_log_probs = log_probs[i, pl : seq_len - 1, :]
        token_log_probs = resp_log_probs.gather(dim=-1, index=resp_input_ids.unsqueeze(-1)).squeeze(-1)
        logprobs_per_sample.append(token_log_probs.mean())
    return torch.stack(logprobs_per_sample, dim=0)


def ppo_update_step(policy_model, old_policy_model, tokenizer, prompts: List[str], device, optimizer,
                    kl_coef: float = 0.1, clip_range: float = 0.2) -> Dict[str, float]:
    with torch.no_grad():
        input_ids, attention_mask, prompt_lens, texts = generate_responses(
            policy_model, tokenizer, prompts, device=device)
    rewards = [compute_reward(p, t) for p, t in zip(prompts, texts)]
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    with torch.no_grad():
        logprob_old = sequence_logprob(old_policy_model, input_ids, attention_mask, prompt_lens, device)
    policy_model.train()
    logprob_new = sequence_logprob(policy_model, input_ids, attention_mask, prompt_lens, device)
    advantages = rewards_t - rewards_t.mean()
    std = advantages.std(unbiased=False)
    if std > 1e-8:
        advantages = advantages / std
    ratio = torch.exp(logprob_new - logprob_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    ppo_loss = -torch.mean(torch.min(unclipped, clipped))
    kl = torch.mean(logprob_old - logprob_new)
    loss = ppo_loss + kl_coef * kl
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
    optimizer.step()
    return {"loss": float(loss.item()), "reward_mean": float(rewards_t.mean().item()), "kl": float(kl.item())}


# ---------- 交替训练：单轮 SFT 阶段 / 单轮 PPO 阶段 ----------
def run_sft_phase(model, tokenizer, sft_dataloader, optimizer, device: torch.device, steps: int):
    """对当前模型做 steps 步 SFT 更新（就地更新 model）。"""
    model.train()
    sft_iter = iter(sft_dataloader)
    for step in range(steps):
        try:
            batch = next(sft_iter)
        except StopIteration:
            sft_iter = iter(sft_dataloader)
            batch = next(sft_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 1 == 0:
            print(f"  [SFT] step={step + 1}/{steps} loss={loss.item():.4f}")


def run_ppo_phase(policy_model, old_policy_model, tokenizer, ppo_dataloader, optimizer, device: torch.device, steps: int):
    """对当前策略做 steps 步 PPO 更新（就地更新 policy_model）。"""
    ppo_iter = iter(ppo_dataloader)
    for step in range(steps):
        try:
            batch = next(ppo_iter)
        except StopIteration:
            ppo_iter = iter(ppo_dataloader)
            batch = next(ppo_iter)
        prompts = batch["prompt"]
        stats = ppo_update_step(policy_model, old_policy_model, tokenizer, prompts, device, optimizer)
        if (step + 1) % 1 == 0:
            print(f"  [PPO] step={step + 1}/{steps} loss={stats['loss']:.4f} reward_mean={stats['reward_mean']:.4f} kl={stats['kl']:.4f}")


def main():
    base_model_path = r"./models/Qwen2-0.5B"
    sft_data_path = r"./data/sft_data.json"
    output_dir = r"./qwen2_0.5b_mixed_cpu"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    print(f"使用设备: {device}")

    # 从基座加载（若想从已有 SFT 开始，可改为 sft_checkpoint_path）
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
    old_policy_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
    old_policy_model.eval()

    sft_dataset = SFTJsonDataset(sft_data_path, tokenizer)
    sft_dataloader = DataLoader(sft_dataset, batch_size=1, shuffle=True, num_workers=0)
    ppo_dataset = PromptDataset(sft_data_path)
    ppo_dataloader = DataLoader(ppo_dataset, batch_size=1, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_rounds = 2
    sft_steps_per_round = 2
    ppo_steps_per_round = 2

    print("SFT 与 RL 交替训练（CPU，每轮步数较小，会较慢）")
    for r in range(num_rounds):
        print(f"--- Round {r + 1}/{num_rounds} ---")
        run_sft_phase(model, tokenizer, sft_dataloader, optimizer, device, sft_steps_per_round)
        old_policy_model.load_state_dict(model.state_dict())
        run_ppo_phase(model, old_policy_model, tokenizer, ppo_dataloader, optimizer, device, ppo_steps_per_round)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"交替训练完成，模型已保存到: {output_dir}")


if __name__ == "__main__":
    main()
