import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

    path = input("输入文件夹路径：")
    root_dir = Path(path)

    results = []
    total_tokens = int(input("输入预期的token数："))

    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue

                    entry = json.loads(line)
                    text = f"{entry["title"]}\n{entry["text"]}"
                    tokens = tokenizer(text)["input_ids"]

                    if len(tokens) >= 512:
                        continue

                    if len(tokens) < 64:
                        continue

                    results.append({"title": entry["title"], "content": entry["text"]})
                    total_tokens -= len(tokens)

                    if total_tokens <= 0:
                        break
        except UnicodeDecodeError:
            print("编码错误，跳过：", file_path)
            continue
        except Exception as e:
            print("打开失败：", file_path, e)
            continue

        if total_tokens <= 0:
            break

    with open("wiki_dataset_tmp.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
