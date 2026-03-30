"""
在已训练模型上抽样生成，计算简易指标并保存 PNG 图表，便于论文插图。

依赖: pip install matplotlib
可选: pip install rouge-score  (计算 ROUGE-L，需数据里有 reference)

用法示例:
  python scripts/eval_plot.py --model qwen2_1.5b_mixed_coig --data data/sft_data_coig.json --out figures/eval_run1
  python scripts/eval_plot.py --model models/Qwen2-0.5B --max-samples 30 --greedy
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def load_items(path: Path, max_samples: int, seed: int) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if max_samples > 0 and len(data) > max_samples:
        rng = random.Random(seed)
        data = rng.sample(data, max_samples)
    return data


def distinct_n(tokens: List[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def char_repeat_ratio(text: str) -> float:
    if not text:
        return 0.0
    from collections import Counter

    top = Counter(text).most_common(1)[0][1]
    return top / len(text)


def generate_one(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    greedy: bool,
) -> str:
    full_prompt = f"用户: {prompt.strip()}\n助手:"
    enc = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    eos_id = tokenizer.eos_token_id
    gen_kw = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or eos_id,
        eos_token_id=eos_id,
        repetition_penalty=1.5,
        no_repeat_ngram_size=6,
    )
    with torch.no_grad():
        if greedy:
            out = model.generate(**gen_kw, do_sample=False)
        else:
            out = model.generate(
                **gen_kw,
                do_sample=True,
                temperature=0.45,
                top_p=0.9,
            )
    gen_ids = out[:, input_ids.size(1) :]
    text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    for stop in ("\n用户:", "\n用户："):
        if stop in text:
            text = text.split(stop, 1)[0].strip()
    return text


def try_rouge_l(preds: List[str], refs: List[str]) -> Optional[List[float]]:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for p, r in zip(preds, refs):
        scores.append(s.score(r, p)["rougeL"].fmeasure)
    return scores


def plot_all(
    out_dir: Path,
    lengths: List[int],
    d1: List[float],
    d2: List[float],
    rep_r: List[float],
    rouge_l: Optional[List[float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 中文字体：Windows 常见黑体；失败则用默认（图例仍可用英文）
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 生成长度分布
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=min(20, max(5, len(lengths) // 3)), color="steelblue", edgecolor="white")
    ax.set_title("生成长度分布（字符数）")
    ax.set_xlabel("字符长度")
    ax.set_ylabel("样本数")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_lengths_hist.png", dpi=150)
    plt.close(fig)

    # 2) 多样性 / 重复 条形图
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["Distinct-1", "Distinct-2", "1-最大字频占比"]
    means = [
        sum(d1) / len(d1),
        sum(d2) / len(d2),
        sum(rep_r) / len(rep_r),
    ]
    ax.bar(names, means, color=["#4c78a8", "#f58518", "#e45756"])
    ax.set_ylim(0, 1.05)
    ax.set_title("多样性（越高越好）与字符重复度（越低越好）")
    fig.tight_layout()
    fig.savefig(out_dir / "eval_distinct_repeat.png", dpi=150)
    plt.close(fig)

    # 3) ROUGE-L（若有）
    if rouge_l:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["ROUGE-L 均值"], [sum(rouge_l) / len(rouge_l)], color="#54a24b")
        ax.set_ylim(0, 1.0)
        ax.set_title("与参考答案 ROUGE-L（F1）")
        fig.tight_layout()
        fig.savefig(out_dir / "eval_rougeL.png", dpi=150)
        plt.close(fig)

    # 4) 汇总四宫格
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(lengths, bins=min(20, max(5, len(lengths) // 3)), color="steelblue", edgecolor="white")
    axes[0, 0].set_title("生成长度分布")
    axes[0, 1].bar(names, means, color=["#4c78a8", "#f58518", "#e45756"])
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_title("Distinct / 重复")
    if rouge_l:
        axes[1, 0].bar(["ROUGE-L"], [sum(rouge_l) / len(rouge_l)], color="#54a24b")
        axes[1, 0].set_ylim(0, 1.0)
        axes[1, 0].set_title("ROUGE-L")
    else:
        axes[1, 0].text(0.5, 0.5, "未安装 rouge-score\n或未计算", ha="center", va="center")
        axes[1, 0].set_axis_off()
    axes[1, 1].plot(range(1, len(lengths) + 1), lengths, marker="o", markersize=3, linestyle="-")
    axes[1, 1].set_title("各样本生成长度（按评测顺序）")
    axes[1, 1].set_xlabel("样本序号")
    axes[1, 1].set_ylabel("字符长度")
    fig.suptitle("模型评测汇总", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_summary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="模型目录（默认项目根下 qwen2_1.5b_mixed_coig）")
    ap.add_argument("--data", type=str, default="data/sft_data_coig.json", help="含 prompt / 可选 response 的 JSON")
    ap.add_argument("--out", type=str, default="figures/eval", help="图表与 jsonl 输出目录")
    ap.add_argument("--max-samples", type=int, default=40, help="抽样条数，0 表示全量（较慢）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--greedy", action="store_true", help="贪心解码（默认识别为采样）")
    args = ap.parse_args()

    root = _root_dir()
    model_dir = Path(args.model) if args.model else root / "qwen2_1.5b_mixed_coig"
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    data_path = root / args.data if not Path(args.data).is_absolute() else Path(args.data)
    out_dir = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if not data_path.exists():
        raise SystemExit(f"数据不存在: {data_path}")

    items = load_items(data_path, args.max_samples, args.seed)
    if not items:
        raise SystemExit("数据为空")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = {"local_files_only": True, "trust_remote_code": True}
    print(f"设备: {device}  模型: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), **kw)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kw).to(device)
    model.eval()

    preds: List[str] = []
    refs: List[str] = []
    rows: List[Dict[str, Any]] = []

    for i, item in enumerate(items):
        prompt = item.get("prompt", "") or ""
        ref = (item.get("response") or "").strip()
        print(f"[{i+1}/{len(items)}] 生成中…")
        gen = generate_one(model, tokenizer, prompt, device, args.max_new_tokens, args.greedy)
        preds.append(gen)
        refs.append(ref)
        toks = tokenizer.tokenize(gen)
        d1 = distinct_n([t.replace("▁", "") for t in toks], 1) if len(toks) >= 1 else 0.0
        d2 = distinct_n([t.replace("▁", "") for t in toks], 2) if len(toks) >= 2 else 0.0
        rows.append(
            {
                "prompt": prompt[:200],
                "reference": ref[:500],
                "prediction": gen[:2000],
                "len_gen": len(gen),
                "distinct_1": d1,
                "distinct_2": d2,
                "char_repeat_ratio": char_repeat_ratio(gen),
            }
        )

    lengths = [r["len_gen"] for r in rows]
    d1s = [r["distinct_1"] for r in rows]
    d2s = [r["distinct_2"] for r in rows]
    rep_rs = [r["char_repeat_ratio"] for r in rows]

    rouge_l = try_rouge_l(preds, refs) if any(refs) else None

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 文本摘要
    summary = {
        "n": len(rows),
        "mean_len": sum(lengths) / len(lengths),
        "mean_distinct_1": sum(d1s) / len(d1s),
        "mean_distinct_2": sum(d2s) / len(d2s),
        "mean_char_repeat_ratio": sum(rep_rs) / len(rep_rs),
        "mean_rougeL": sum(rouge_l) / len(rouge_l) if rouge_l else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("指标:", json.dumps(summary, ensure_ascii=False, indent=2))

    try:
        plot_all(out_dir, lengths, d1s, d2s, rep_rs, rouge_l)
    except ImportError:
        print("未安装 matplotlib，跳过出图。请执行: pip install matplotlib")

    print(f"已写入: {pred_path}  图表目录: {out_dir}")


if __name__ == "__main__":
    main()
