import json

from random import shuffle


def process(path: str):
    with open(path, 'r', encoding="utf-8") as f:
        poems = json.load(f)

    for poem in poems:
        raw_text = poem["content"]

        lines = raw_text.splitlines()
        shuffle(lines)

        poem["content"] = "\n".join(lines).strip()

    return poems


def main():
    path = input("输入要处理的诗歌文件路径：")
    poems = process(path)

    with open("shuffled_dataset_tmp.json", 'w', encoding='utf-8') as f:
        json.dump(poems, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()