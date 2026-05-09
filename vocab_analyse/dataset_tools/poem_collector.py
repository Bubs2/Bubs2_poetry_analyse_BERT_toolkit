import json


def process(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chunks = []
    last_idx = 0
    for idx, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line: continue

        next_2_lines = [l.strip() for l in lines[idx + 1: idx + 3]]

        is_content_end = clean_line.startswith("选自")
        is_sep = clean_line.startswith('（')
        is_eof = (idx + 1 == len(lines))
        is_followed_by_empty = not is_eof and not any(next_2_lines)

        if is_content_end:
            raw_text = "".join(lines[last_idx:idx]).strip()
            chunks.append(raw_text)

        if is_sep and (is_eof or is_followed_by_empty):
            last_idx = idx + 1

    print(chunks[1])
    return chunks


def main():
    path = input("输入要处理的诗歌文件路径：")
    chunks: list[str] = process(path)

    results = []
    for item in chunks:
        title_end_pos = item.find('\n')
        title = item[:title_end_pos]
        content_start_pos = item.find('\n', title_end_pos + 2) + 1
        content = item[content_start_pos:]
        results.append({"title": title, "content": content})

    with open("poems_tmp.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()