"""
SFT 与 RL（PPO）交替训练：同一模型先做若干步 SFT，再做若干步 PPO，循环多轮。
支持 CPU，但会较慢，每轮步数较小（如 sft_steps=2, ppo_steps=2, num_rounds=2）。
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import json
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM


# ---------- 自动敏感词：下载词库 + 子串匹配（非「理解语义」的 AI 判别）----------
_SENSITIVE_LEXICON: Optional[Tuple[str, Any]] = None

# 公开词库镜像（任一成功即可）；离线时请自行放置 data/sensitive_words_zh.txt
_SENSITIVE_WORDLIST_URLS = (
    "https://raw.githubusercontent.com/observerss/textfilter/master/sensitive_words.txt",
    "https://cdn.jsdelivr.net/gh/observerss/textfilter@master/sensitive_words.txt",
)


def _project_root_path() -> Path:
    return Path(__file__).resolve().parent.parent


def _download_sensitive_word_file(dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in _SENSITIVE_WORDLIST_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Qwen2-SFT-CPU/1.0)"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            if len(raw.strip()) < 80:
                continue
            dest.write_text(raw, encoding="utf-8")
            return True
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return False


def prepare_sensitive_lexicon(project_root: Path) -> None:
    """
    若 data/sensitive_words_zh.txt 不存在或过小，尝试自动下载开源词库。
    仍失败则仅用英文粗口检测（better-profanity，可选安装）+ 环境变量 REWARD_BAD_WORDS。
    """
    global _SENSITIVE_LEXICON
    _SENSITIVE_LEXICON = None
    path = project_root / "data" / "sensitive_words_zh.txt"
    if not path.exists() or path.stat().st_size < 80:
        if _download_sensitive_word_file(path):
            print(f"已自动下载中文敏感词库: {path}")
        else:
            print(
                "提示: 未能联网下载敏感词库。可手动下载词表保存为 data/sensitive_words_zh.txt ，"
                "或 pip install better-profanity 以启用英文粗口检测。"
            )


def _load_sensitive_lexicon() -> Tuple[str, Any]:
    """返回 (mode, payload)：aho | naive | none"""
    global _SENSITIVE_LEXICON
    if _SENSITIVE_LEXICON is not None:
        return _SENSITIVE_LEXICON

    path = _project_root_path() / "data" / "sensitive_words_zh.txt"
    words: List[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            w = line.strip().split("#")[0].strip()
            if len(w) >= 2:
                words.append(w)
    # 去重并限制规模，避免无 ahocorasick 时子串匹配过慢
    words = list(dict.fromkeys(words))[:6000]

    if not words:
        _SENSITIVE_LEXICON = ("none", None)
        return _SENSITIVE_LEXICON

    try:
        import ahocorasick as ah  # type: ignore

        automaton = ah.Automaton()
        for w in words:
            automaton.add_word(w, w)
        automaton.make_automaton()
        _SENSITIVE_LEXICON = ("aho", automaton)
    except ImportError:
        _SENSITIVE_LEXICON = ("naive", frozenset(words))

    return _SENSITIVE_LEXICON


def _count_sensitive_lexicon_hits(text: str) -> int:
    mode, data = _load_sensitive_lexicon()
    if mode == "none":
        return 0
    if mode == "naive":
        return sum(1 for w in data if w in text)
    seen = set()
    for _, w in data.iter(text):
        seen.add(w)
    return len(seen)


def _english_profanity_hit(text: str) -> bool:
    try:
        from better_profanity import profanity

        return profanity.contains_profanity(text)
    except ImportError:
        return False


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


def sft_collate_fn(batch, pad_token_id: int = 0):
    """将多条 SFT 样本 pad 到同一长度再 stack，供 batch_size>1 使用。"""
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    labels = [x["labels"] for x in batch]
    max_len = max(x.size(0) for x in input_ids)
    pad_input = [F.pad(x, (0, max_len - x.size(0)), value=pad_token_id) for x in input_ids]
    pad_attn = [F.pad(x, (0, max_len - x.size(0)), value=0) for x in attention_mask]
    pad_labels = [F.pad(x, (0, max_len - x.size(0)), value=-100) for x in labels]
    return {
        "input_ids": torch.stack(pad_input),
        "attention_mask": torch.stack(pad_attn),
        "labels": torch.stack(pad_labels),
    }


# ---------- PPO 用：reward、生成、logprob、单步更新 ----------
def compute_reward(prompt: str, response: str) -> float:
    """
    启发式奖励。旧版强依赖「RLHF/SFT」等关键词，与 COIG 等开放域数据严重错配。

    当前设计（可按实验调权重）：
      - 平滑长度：过短/过长渐近惩罚，避免硬阈值抖动
      - 重复：单字占比过高视为刷屏，轻罚
      - 中文：回复中含一定比例汉字时小幅奖励（中文指令数据）
      - 敏感词：自动加载 data/sensitive_words_zh.txt（首次可联网下载开源词表）；可选 pip install better-profanity 检英文粗口
      - 额外：环境变量 REWARD_BAD_WORDS=词1,词2 与自动词库叠加

    说明：子串匹配≠语义理解，仍有漏报/误报；严肃场景请用审核 API 或专用分类模型。
    """
    text = (response or "").strip()
    if not text:
        return -2.0

    n = len(text)

    # --- 1) 平滑长度分：理想段内给正分，向外渐弱 ---
    lo, hi = 24, 1200  # 字符级；可按 max_new_tokens 调 hi
    if n < 8:
        length_score = -1.0
    elif n < lo:
        length_score = -0.5 + 0.5 * (n - 8) / (lo - 8)  # [-0.5, 0) 附近
    elif n <= hi:
        length_score = 0.35
    else:
        overflow = min(n - hi, 2000)
        length_score = 0.35 - 0.35 * (overflow / 2000.0)

    # --- 2) 重复惩罚：最高频字符占比 ---
    if n >= 16:
        top_freq = Counter(text).most_common(1)[0][1]
        top_ratio = top_freq / n
        if top_ratio > 0.35:
            repeat_penalty = -0.6
        elif top_ratio > 0.22:
            repeat_penalty = -0.25
        else:
            repeat_penalty = 0.0
    else:
        repeat_penalty = 0.0

    # --- 3) 中文占比小奖（prompt 含中文时略增权，避免英文任务被误伤）---
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    cjk_ratio = cjk / n
    prompt_cjk = any("\u4e00" <= c <= "\u9fff" for c in prompt)
    if prompt_cjk and cjk_ratio >= 0.08:
        lang_bonus = min(0.15, 0.15 * (cjk_ratio / 0.3))
    else:
        lang_bonus = 0.0

    # --- 4) 敏感词 / 粗口：自动词库 + 环境变量追加 ---
    lex_hits = _count_sensitive_lexicon_hits(text)
    bad_score = -0.45 * min(lex_hits, 12)
    if _english_profanity_hit(text):
        bad_score -= 0.55
    extra_bad = [w.strip() for w in os.environ.get("REWARD_BAD_WORDS", "").split(",") if w.strip()]
    bad_score -= 1.0 * sum(1 for bw in extra_bad if bw in text)

    # --- 5) 可选领域关键词小奖（不写环境变量则为 0）---
    domain_kw = [w.strip() for w in os.environ.get("REWARD_DOMAIN_KEYWORDS", "").split(",") if w.strip()]
    domain_bonus = 0.08 * sum(1 for kw in domain_kw if kw in text)

    return float(length_score + repeat_penalty + lang_bonus + bad_score + domain_bonus)


@torch.no_grad()
def generate_responses_with_logprobs(
    model, tokenizer, prompts: List[str], device: torch.device,
    max_input_length: int = 256, max_new_tokens: int = 64,
):
    """生成回复并记录逐 token 的旧策略 log 概率（无需 old_policy）。"""
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id or 0
    encoded = [tokenizer(p, return_tensors="pt", truncation=True, max_length=max_input_length) for p in prompts]
    max_pl = max(enc["input_ids"].size(1) for enc in encoded)
    prompt_lens = [enc["input_ids"].size(1) for enc in encoded]
    input_ids = []
    attention_mask = []
    for enc in encoded:
        ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)
        if ids.size(0) < max_pl:
            ids = F.pad(ids, (0, max_pl - ids.size(0)), value=pad_id)
            attn = F.pad(attn, (0, max_pl - attn.size(0)), value=0)
        input_ids.append(ids)
        attention_mask.append(attn)
    input_ids = torch.stack(input_ids).to(device)
    attention_mask = torch.stack(attention_mask).to(device)

    all_token_logprobs = []
    for _ in range(max_new_tokens):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        next_tokens = torch.multinomial(log_probs.exp(), num_samples=1)
        token_log_probs = log_probs.gather(dim=-1, index=next_tokens).squeeze(-1)
        all_token_logprobs.append(token_log_probs)
        input_ids = torch.cat([input_ids, next_tokens], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens, device=device)], dim=1)

    # [B, T]，T=max_new_tokens；逐 token 的旧策略 logprob
    old_token_logprobs = torch.stack(all_token_logprobs, dim=1)
    texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return input_ids, attention_mask, prompt_lens, texts, old_token_logprobs


def sequence_token_logprobs(model, input_ids, attention_mask, gen_len: int, device: torch.device):
    """对固定已生成序列，返回逐 token 的当前策略 logprob: [B, gen_len]。"""
    model.eval()
    outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    _, seq_len, _ = log_probs.size()
    if gen_len <= 0 or gen_len >= seq_len:
        raise ValueError(f"invalid gen_len={gen_len} for seq_len={seq_len}")

    # 生成段统一位于序列尾部：目标 token 在 [start, seq_len-1]，对应 logits 位置 [start-1, seq_len-2]
    start = seq_len - gen_len
    target_ids = input_ids[:, start:seq_len].to(device)             # [B, T]
    step_log_probs = log_probs[:, start - 1 : seq_len - 1, :]       # [B, T, V]
    token_log_probs = step_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs


def ppo_update_step(policy_model, tokenizer, prompts: List[str], device, optimizer,
                    kl_coef: float = 0.1, clip_range: float = 0.2) -> Dict[str, float]:
    with torch.no_grad():
        input_ids, attention_mask, prompt_lens, texts, old_token_logprobs = generate_responses_with_logprobs(
            policy_model, tokenizer, prompts, device=device)
    rewards = [compute_reward(p, t) for p, t in zip(prompts, texts)]
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    policy_model.train()
    gen_len = old_token_logprobs.size(1)
    new_token_logprobs = sequence_token_logprobs(policy_model, input_ids, attention_mask, gen_len, device)
    # PPO 比率沿用样本级均值 logprob，避免改动过大；KL 使用逐 token 更标准近似
    logprob_old = old_token_logprobs.mean(dim=1)
    logprob_new = new_token_logprobs.mean(dim=1)
    advantages = rewards_t - rewards_t.mean()
    std = advantages.std(unbiased=False)
    if std > 1e-8:
        advantages = advantages / std
    ratio = torch.exp(logprob_new - logprob_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    ppo_loss = -torch.mean(torch.min(unclipped, clipped))
    kl = torch.mean(old_token_logprobs - new_token_logprobs)
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


def run_ppo_phase(policy_model, tokenizer, ppo_dataloader, optimizer, device: torch.device, steps: int):
    """对当前策略做 steps 步 PPO 更新（就地更新 policy_model）。"""
    ppo_iter = iter(ppo_dataloader)
    for step in range(steps):
        try:
            batch = next(ppo_iter)
        except StopIteration:
            ppo_iter = iter(ppo_dataloader)
            batch = next(ppo_iter)
        prompts = batch["prompt"]
        stats = ppo_update_step(policy_model, tokenizer, prompts, device, optimizer)
        if (step + 1) % 1 == 0:
            print(f"  [PPO] step={step + 1}/{steps} loss={stats['loss']:.4f} reward_mean={stats['reward_mean']:.4f} kl={stats['kl']:.4f}")


def main():
    # 使用绝对路径，避免新版 huggingface_hub 把本地路径当 repo_id 校验报错
    _root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prepare_sensitive_lexicon(Path(_root))

    base_model_path = os.path.join(_root, "models", "Qwen2-1.5B")
    sft_data_path = os.path.join(_root, "data", "sft_data_coig.json")
    output_dir = os.path.join(_root, "qwen2_1.5b_mixed_coig")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() // 2))
    print(f"使用设备: {device}")

    # 从基座加载，只保留一个 policy（生成时存 log π_old，无需 old_policy）
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = Qwen2ForCausalLM.from_pretrained(base_model_path).to(device)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id or 0
    batch_size = 10
    sft_dataset = SFTJsonDataset(sft_data_path, tokenizer)
    sft_dataloader = DataLoader(
        sft_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: sft_collate_fn(b, pad_token_id=pad_token_id),
    )
    ppo_dataset = PromptDataset(sft_data_path)
    ppo_dataloader = DataLoader(ppo_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # ---------- 训练规模 ----------
    # num_rounds：SFT→PPO 循环次数（例如 3 表示 SFT→PPO→SFT→PPO→SFT→PPO）
    # sft_steps_per_round / ppo_steps_per_round：每轮里各跑多少步（每步用 batch_size 条数据）
    #
    # 显存：只要 batch_size、序列长、模型不变，增大步数/轮数不会提高「峰值显存」，
    #       只会线性增加总训练时间。若仍 OOM，应减小 batch 或 max_length，而不是减轮数。
    num_rounds = 5
    sft_steps_per_round = 100
    ppo_steps_per_round = 20

    # 若希望「每轮 SFT / PPO 各扫一遍当前 JSON 全量」，可改用下面两行（注释掉上面两个 per_round 数字）：
    # sft_steps_per_round = max(1, (len(sft_dataset) + batch_size - 1) // batch_size)
    # ppo_steps_per_round = max(1, (len(ppo_dataset) + batch_size - 1) // batch_size)

    print("SFT 与 RL 交替训练（单 policy，生成时存 log π_old，全程 GPU）")
    print(f"num_rounds={num_rounds}, sft_steps/round={sft_steps_per_round}, ppo_steps/round={ppo_steps_per_round}")
    for r in range(num_rounds):
        print(f"--- Round {r + 1}/{num_rounds} ---")
        run_sft_phase(model, tokenizer, sft_dataloader, optimizer, device, sft_steps_per_round)
        run_ppo_phase(model, tokenizer, ppo_dataloader, optimizer, device, ppo_steps_per_round)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"交替训练完成，模型已保存到: {output_dir}")


if __name__ == "__main__":
    main()
