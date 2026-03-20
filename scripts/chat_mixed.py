"""
命令行对话：加载 mixed 模型，输入问题后生成回答并打印。
用法：python scripts/chat_mixed.py
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # 项目根 = scripts 的上一级，与 train_qwen2_mixed 的 output_dir 同级（切勿少一层 dirname）
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.abspath(
        os.path.expanduser(os.environ.get("CHAT_MODEL_DIR", os.path.join(_root, "qwen2_1.5b_mixed")))
    )
    kw = {"local_files_only": True, "trust_remote_code": True}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"加载模型与分词器: {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, **kw)
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kw).to(device)
    model.eval()

    print("输入问题后回车生成回答，输入 quit 或 exit 退出。\n")
    while True:
        try:
            prompt = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,  # 惩罚已出现过的 token
                no_repeat_ngram_size=3,  # 禁止重复出现相同的 3 个连续词
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # 只保留新生成的部分
        gen_ids = output_ids[:, input_ids.size(1) :]
        reply = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        print("模型:", reply)
        print()


if __name__ == "__main__":
    main()
