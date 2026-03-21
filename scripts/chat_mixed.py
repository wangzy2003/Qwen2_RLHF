"""
命令行对话：加载 mixed 模型，输入问题后生成回答并打印。
用法：python scripts/chat_mixed.py

推理格式与 train_qwen2_mixed.py 的 SFT 一致：「用户: 问题」换行「助手:」再接生成内容。
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _truncate_repeated_spans(text: str, min_span: int = 14) -> str:
    """若同一段子串在文中重复出现，在第二次出现前截断，减轻 SFT/助手 式死循环。"""
    if len(text) < min_span * 3:
        return text
    for span in range(min(len(text) // 2, 80), min_span - 1, -1):
        for i in range(len(text) - 2 * span):
            chunk = text[i : i + span]
            if len(chunk.strip()) < min_span // 2:
                continue
            j = text.find(chunk, i + span)
            if j != -1 and j < i + span + 80:
                return text[: i + span].rstrip()
    return text


def strip_optional_user_prefix(text: str) -> str:
    """去掉用户误输入的「你：」「用户:」等前缀，避免与模板重复。"""
    t = text.strip()
    for p in ("你：", "你:", "用户：", "用户:"):
        if t.startswith(p):
            t = t[len(p) :].lstrip()
    return t


def main():
    # 项目根 = scripts 的上一级，与 train_qwen2_mixed 的 output_dir 同级（切勿少一层 dirname）
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.abspath(
        os.path.expanduser(
            os.environ.get("CHAT_MODEL_DIR", os.path.join(_root, "qwen2_1.5b_mixed_coig"))
        )
    )
    kw = {"local_files_only": True, "trust_remote_code": True}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_greedy = os.environ.get("CHAT_GREEDY", "").strip() in ("1", "true", "yes")

    print(f"加载模型与分词器: {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, **kw)
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kw).to(device)
    model.eval()

    print("输入问题后回车生成回答，输入 quit 或 exit 退出。")
    print("若出现复读 SFT/助手 等：设环境变量 CHAT_GREEDY=1 试贪心解码；或换用仅 SFT 的 checkpoint。\n")
    while True:
        try:
            prompt = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break

        user_content = strip_optional_user_prefix(prompt)
        # 与 SFT 完全一致：模型只负责续写「助手:」之后的内容
        full_prompt = f"用户: {user_content}\n助手:"

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
            max_new_tokens=160,
            pad_token_id=tokenizer.pad_token_id or eos_id,
            eos_token_id=eos_id,
            repetition_penalty=1.5,
            no_repeat_ngram_size=6,
        )
        with torch.no_grad():
            if use_greedy:
                output_ids = model.generate(**gen_kw, do_sample=False)
            else:
                output_ids = model.generate(
                    **gen_kw,
                    do_sample=True,
                    temperature=0.45,
                    top_p=0.9,
                )

        # 只保留新生成的部分
        gen_ids = output_ids[:, input_ids.size(1) :]
        reply = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        reply = _truncate_repeated_spans(reply)
        # 若模型又编出下一轮「用户:」，截断，避免整段胡话
        for stop in ("\n用户:", "\n用户："):
            if stop in reply:
                reply = reply.split(stop, 1)[0].strip()
        print("模型:", reply)
        print()


if __name__ == "__main__":
    main()
