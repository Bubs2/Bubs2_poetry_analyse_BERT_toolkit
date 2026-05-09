import json

from enum import Enum


def process(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chunks = []

    class State(Enum):
        Skip = -1
        Gap = 0
        Title = 1
        Content = 2
        Comment = 3

    state: State = State.Gap

    title_idx = 0
    content_idx = 0
    comment_idx = 0

    for idx, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line: continue

        is_eof = (idx + 1 == len(lines))

        if state is State.Gap:
            title_idx = idx
            state = State.Title

        match state:
            case State.Title:
                if "原诗略" in clean_line:
                    state = State.Skip
                    continue
                if is_eof:
                    break
                if not lines[idx + 1].strip():
                    state = State.Content
                    content_idx = idx + 3 # 作者不进入标题/内容
            case State.Content:
                if clean_line.startswith("选自"):
                    state = State.Comment
                    comment_idx = idx + 1
            case State.Comment:
                next_3_lines = [l.strip() for l in lines[idx + 1: idx + 4]]
                is_followed_by_empty = not is_eof and not any(next_3_lines)
                if is_eof or is_followed_by_empty:
                    state = State.Gap
                    chunks.append({
                        "title": "".join(lines[title_idx:content_idx-2]).strip(),
                        "content": "".join(lines[content_idx:comment_idx-1]).strip(),
                        "comment": "".join(lines[comment_idx:idx+1]).strip()
                    })
            case State.Skip:
                next_3_lines = [l.strip() for l in lines[idx + 1: idx + 4]]
                is_followed_by_empty = not is_eof and not any(next_3_lines)
                if is_eof or is_followed_by_empty:
                    state = State.Gap

    print(chunks[0])
    return chunks


def main():
    path = input("输入要处理的诗歌文件路径：")
    chunks: list[str] = process(path)

    with open("poems_tmp.json", 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()