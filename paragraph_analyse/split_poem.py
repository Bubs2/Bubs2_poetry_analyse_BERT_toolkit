import re
from enum import Enum


class SplitMode(Enum):
    Line = 0
    Paragraph = 1

def split_poem(text: str, mode: SplitMode = SplitMode.Line, print_markdown: bool = False):
    """
    返回 [(segment_text, start_char, end_char), ...]
    """
    segments = []
    if mode == SplitMode.Line: pattern = r'(.*?)(?:\n|$)'
    else: pattern = r'(.*?)(?:\n\s*\n|$)'

    for match in re.finditer(pattern, text, re.S):
        seg = match.group(1)
        if seg.strip():
            start = match.start(1)
            end = match.start(1) + len(seg)
            segments.append((seg, start, end))

    if print_markdown: print_as_markdown(segments, "L" if mode == SplitMode.Line else "P")
    return segments

def print_as_markdown(segments, prefix: str = ""):
    sheet = ["| 序号 | 文本 |", "|---:|---|"]
    for i, (seg, start, end) in enumerate(segments, 1):
        content = seg.replace("\n", "<br>").replace("|", r"\|")
        sheet.append(f"| {prefix}{i} | {content} |")
    print("\n".join(sheet))