"""
把 BAAI 的 COIG-PC 指令数据转成本项目 SFT/PPO 共用的 JSON：
  [{"prompt": "...", "response": "..."}, ...]

用法（先安装: pip install datasets）:
  # 先试小集（每任务约 200 条，总体积小）
  python scripts/prepare_coig_sft.py --dataset BAAI/COIG-PC-Lite --max-samples 5000

  # 常用高质量全集子集（COIG-PC-core，约 74 万条，建议限条数）
  python scripts/prepare_coig_sft.py --dataset BAAI/COIG-PC-core --max-samples 20000

  # 指定输出路径
  python scripts/prepare_coig_sft.py --dataset BAAI/COIG-PC-core --max-samples 100000 \\
      --output data/sft_data_coig.json

字段映射:
  prompt  = instruction + (若有 input 则换行拼上 input)
  response = output

许可证与引用见各数据集在 Hugging Face 上的说明页（如 BAAI/COIG-PC-core）。
"""
import argparse
import json
import os
from pathlib import Path


def row_to_pair(row: dict) -> dict | None:
    inst = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    out = (row.get("output") or "").strip()
    if not out:
        return None
    if inp:
        prompt = f"{inst}\n{inp}".strip()
    else:
        prompt = inst
    if not prompt:
        return None
    return {"prompt": prompt, "response": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="BAAI/COIG-PC-core",
        help="HF 数据集名，如 BAAI/COIG-PC-core 或 BAAI/COIG-PC-Lite",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=20_000, help="最多导出条数（全量很大，务必限制）")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 json 路径，默认 项目根/data/sft_data_coig.json",
    )
    parser.add_argument("--seed", type=int, default=42, help="打乱抽样用的随机种子")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("请先安装: pip install datasets")

    _root = Path(__file__).resolve().parent.parent
    out_path = Path(args.output) if args.output else _root / "data" / "sft_data_coig.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"加载 {args.dataset} (split={args.split}) ...")
    ds = load_dataset(args.dataset, split=args.split, trust_remote_code=True)

    # 打乱后取前 N 条，避免只用到数据集中最前面单一任务
    if len(ds) > args.max_samples:
        ds = ds.shuffle(seed=args.seed).select(range(args.max_samples))
        print(f"已随机抽样 {args.max_samples} 条（seed={args.seed}）")
    else:
        print(f"全量共 {len(ds)} 条，小于 max-samples，全部使用")

    pairs = []
    for row in ds:
        p = row_to_pair(row)
        if p:
            pairs.append(p)

    out_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {len(pairs)} 条 -> {out_path}")
    print("训练时在 train_qwen2_mixed.py 里把 sft_data_path 指到该文件即可。")


if __name__ == "__main__":
    main()
