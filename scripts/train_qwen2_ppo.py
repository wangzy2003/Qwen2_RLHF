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


class PromptDataset(Dataset):
    """
    极简 PPO 用的 prompt 数据集。
    这里直接重用 SFT 的 json 格式，只取 prompt 字段：
    [
      {"prompt": "...", "response": "..."},
      ...
    ]
    """

    def __init__(self, json_path: str):
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.prompts = [item["prompt"] for item in data]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}


def compute_reward(prompt: str, response: str) -> float:
    """
    一个非常简单的规则奖励示例：
    - 回答长度适中给正分，太短/太长给负分
    - 命中某些关键字加分
    - 出现违禁词减分
    真实使用中请按你的任务需求修改。
    """
    length = len(response)
    if length < 20:
        length_score = -1.0
    elif length > 200:
        length_score = -1.0
    else:
        length_score = 1.0

    keywords = ["RLHF", "SFT", "强化学习"]
    keyword_hits = sum(1 for kw in keywords if kw in response)
    keyword_score = 0.5 * keyword_hits

    bad_words = ["脏话1", "脏话2"]
    bad_hits = sum(1 for bw in bad_words if bw in response)
    bad_score = -1.0 * bad_hits

    reward = length_score + keyword_score + bad_score
    return float(reward)


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_input_length: int = 256,
    max_new_tokens: int = 64,
):
    """
    用当前 policy 在 CPU 上对一批 prompt 生成回复。
    返回：
    - input_ids: 拼接后的 prompt+response token
    - attention_mask
    - prompt_lens: 每条样本的 prompt token 长度（用于切分 response）
    - texts: 解码后的完整文本
    """
    model.eval()

    all_input_ids = []
    all_attention_mask = []
    prompt_lens = []

    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        prompt_len = input_ids.size(1)

        output_ids = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
        )

        all_input_ids.append(output_ids.cpu())
        attn_mask = (output_ids != tokenizer.pad_token_id).long()
        all_attention_mask.append(attn_mask)
        prompt_lens.append(prompt_len)

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)
    texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    return input_ids, attention_mask, prompt_lens, texts


def sequence_logprob(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: List[int],
    device: torch.device,
) -> torch.Tensor:
    """
    计算每条样本在“回答部分”的平均 logprob。
    为简化，只对从 prompt 结束到序列末尾的 token 计算平均 logprob。
    """
    model.eval()
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
    )
    logits = outputs.logits  # [B, T, V]
    log_probs = F.log_softmax(logits, dim=-1)

    batch_size, seq_len, _ = log_probs.size()
    logprobs_per_sample = []

    for i in range(batch_size):
        pl = prompt_lens[i]
        # 回答部分 token（被预测的是从 pl 到 seq_len-1 的 token）
        if pl >= seq_len - 1:
            # 极端情况：没有生成新 token，给一个很小的 logprob
            logprobs_per_sample.append(torch.tensor(-10.0, device=device))
            continue

        resp_input_ids = input_ids[i, pl + 1 :].to(device)
        resp_log_probs = log_probs[i, pl : seq_len - 1, :]
        token_log_probs = resp_log_probs.gather(
            dim=-1, index=resp_input_ids.unsqueeze(-1)
        ).squeeze(-1)
        logprobs_per_sample.append(token_log_probs.mean())

    return torch.stack(logprobs_per_sample, dim=0)  # [B]


def ppo_update_step(
    policy_model,
    old_policy_model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    kl_coef: float = 0.1,
    clip_range: float = 0.2,
) -> Dict[str, float]:
    """
    对一小批 prompt 执行一次 PPO 更新（CPU 友好版）。
    """
    # 1) rollout
    with torch.no_grad():
        input_ids, attention_mask, prompt_lens, texts = generate_responses(
            policy_model,
            tokenizer,
            prompts,
            device=device,
        )

    # 2) 计算 reward
    rewards = []
    for p, full_text in zip(prompts, texts):
        # 这里简单用完整文本作为 response，实际可按需切 prompt/response
        r = compute_reward(p, full_text)
        rewards.append(r)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

    # 3) 计算 old / new logprob
    with torch.no_grad():
        logprob_old = sequence_logprob(
            old_policy_model, input_ids, attention_mask, prompt_lens, device
        )

    policy_model.train()
    logprob_new = sequence_logprob(
        policy_model, input_ids, attention_mask, prompt_lens, device
    )

    # 4) advantage：减去均值；batch 很小时避免除以接近 0 的 std
    advantages = rewards_t - rewards_t.mean()
    std = advantages.std(unbiased=False)
    if std > 1e-8:
        advantages = advantages / std

    # 5) PPO ratio
    ratio = torch.exp(logprob_new - logprob_old)

    # 6) PPO clipped loss
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    ppo_loss = -torch.mean(torch.min(unclipped, clipped))

    # 7) 简单 KL 惩罚
    kl = torch.mean(logprob_old - logprob_new)
    loss = ppo_loss + kl_coef * kl

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "ppo_loss": float(ppo_loss.item()),
        "kl": float(kl.item()),
        "reward_mean": float(rewards_t.mean().item()),
    }


def main():
    # 1. 模型与数据路径（完全基于 CPU）
    # 使用原始基座模型目录加载 tokenizer，
    # 使用 SFT checkpoint 目录加载权重，避免 tokenizer 相关依赖问题。
    base_model_path = r"./models/Qwen2-0.5B"
    sft_checkpoint_path = r"./qwen2_0.5b_sft_cpu/checkpoint-2"
    prompt_json_path = r"./data/sft_data.json"  # 这里直接重用现有数据中的 prompt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    policy_model = AutoModelForCausalLM.from_pretrained(sft_checkpoint_path).to(device)
    old_policy_model = AutoModelForCausalLM.from_pretrained(sft_checkpoint_path).to(device)
    old_policy_model.eval()

    dataset = PromptDataset(prompt_json_path)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # CPU 上小 batch 更稳
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)

    num_epochs = 1  # 先用很小的轮数在 CPU 上调试
    update_old_policy_every = 10  # 每多少个 batch 同步一次 old_policy

    global_step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            prompts = batch["prompt"]  # list[str] （batch_size 小）
            stats = ppo_update_step(
                policy_model,
                old_policy_model,
                tokenizer,
                prompts,
                device,
                optimizer,
            )
            global_step += 1

            if global_step % 1 == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={stats['loss']:.4f} "
                    f"reward_mean={stats['reward_mean']:.4f} "
                    f"kl={stats['kl']:.4f}"
                )

            if global_step % update_old_policy_every == 0:
                old_policy_model.load_state_dict(policy_model.state_dict())

    # 训练结束后保存 PPO 调整后的模型
    output_dir = "./qwen2_0.5b_ppo_cpu"
    os.makedirs(output_dir, exist_ok=True)
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"PPO 训练完成，模型已保存到: {output_dir}")


if __name__ == "__main__":
    main()

