import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

    path = input("输入文件路径：")
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    results = []
    for entry in raw_data:
        text = f"{entry["title"]}\n{entry["content"]}"
        tokens = tokenizer(text)["input_ids"]

        if len(tokens) < 512:
            results.append({"title": entry["title"], "content": entry["content"]})

    with open("dataset_tmp.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()