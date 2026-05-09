import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer


def process(path: str):
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

    with open(path, 'r', encoding="utf-8") as f:
        poems = json.load(f)

    total_lines = 0
    total_tokens = 0

    for poem in poems:
        raw_text = poem["content"]

        lines = raw_text.splitlines()

        for line in lines:
            total_tokens += len(tokenizer(line)["input_ids"])

        total_lines += len(lines)

    return total_tokens / total_lines


def main():
    path = input("输入要处理的诗歌文件路径：")
    print(process(path))


if __name__ == "__main__":
    main()
