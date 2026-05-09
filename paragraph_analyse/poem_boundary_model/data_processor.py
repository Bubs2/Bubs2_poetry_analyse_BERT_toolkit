import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer


def main():
    """
        要求的原始 content 格式:
        - 段间用空行分隔（\n\n）
        - 段内按单换行分行

        输出:
        - title
        - content: 插入 [B] 后的训练文本
        - labels: 每个 [B] 对应的 0/1 标签
    """

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[B]"]})

    path = input("输入文件路径：")
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    results = []
    for entry in raw_data:
        content = entry["content"]
        labels = []

        idx = 0
        while idx < len(content):
            if content[idx] == '\n':
                if content[idx + 1] == '\n':
                    labels.append(1.0)
                    idx += 1
                else:
                    labels.append(0.0)
            idx += 1

        content = content.replace("\n\n", "\n").replace("\n", "\n[B]\n")

        encoded = tokenizer.encode(content, add_special_tokens=True)
        if len(encoded) < 512:
            results.append({"title": entry["title"], "content": content, "labels": labels})

    with open("./dataset_tmp.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()