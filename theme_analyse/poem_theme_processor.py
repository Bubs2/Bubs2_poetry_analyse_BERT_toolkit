import json
from random import sample

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam


def clean_ebook(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = []
    skipped_empty_line = False
    idx = 0
    while idx < len(lines):
        if not lines[idx].strip():
            if skipped_empty_line: result.append(lines[idx])
            else: skipped_empty_line = True

            idx += 1
            continue

        skipped_empty_line = False
        result.append(lines[idx])

        if "【诗人小传】" in lines[idx]:
            result = result[:-3]
            idx += 2

        idx += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in result: f.write(line)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=20),
    reraise=True
)
def process_single_poem(client, model, messages, tools):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        timeout=180
    )

    message = response.choices[0].message

    if message.tool_calls:
        return json.loads(message.tool_calls[0].function.arguments)

    if message.content:
        raise ValueError(f"模型错误输出至content：{message.content}")

    raise ValueError(f"模型无输出。")



def process(input_file, output_file, url, key, model):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "process_poem",
                "description": "将给出的诗歌文本整理为给定形式。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "诗歌的标题，不加书名号，使用中文标点符号。",
                        },
                        "author": {
                            "type": "string",
                            "description": "诗歌的作者。",
                        },
                        "content": {
                            "type": "string",
                            "description": "诗歌的正文，不包括标题和作者。要求保留空行和任何诗歌原来的形式，但去除诗尾的时间记录。使用中文标点符号，非markdown，每行一个换行符。",
                        },
                        "comment": {
                            "type": "string",
                            "description": "诗歌的评论文章。要求去除末尾的评论人姓名。使用中文标点符号。",
                        },
                        "style_tag": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "符合诗歌艺术风格的tag，依据评论文章提取。要求最好为两个字，所有tag的词性最好保持一致。",
                        },
                        "emotion_tag": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "符合诗歌所传达的情感的tag，依据评论文章提取。要求最好为两个字，所有tag的词性最好保持一致。",
                        },
                        "theme_tag": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "符合诗歌思想主旨的tag，依据评论文章提取。要求最好为两个字，所有tag的词性最好保持一致。",
                        },
                        "negative_tag": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 9,
                            "maxItems": 9,
                            "description": "与本诗歌无关，但是常用于诗歌分析的tag。要求最好为两个字。",
                        },
                    },
                    "required": ["title", "author", "content", "comment", "style_tag", "emotion_tag", "theme_tag", "negative_tag"]
                },
            }
        },
    ]
    formatted_tools = cast(List[ChatCompletionToolParam], cast(object, tools))

    client = OpenAI(api_key=key, base_url=url)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chunks = []
    last_idx = 0
    for idx, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line: continue

        next_2_lines = [l.strip() for l in lines[idx + 1: idx + 3]]

        is_sep = clean_line.startswith('（')
        is_eof = (idx + 1 == len(lines))
        is_followed_by_empty = not is_eof and not all(next_2_lines)

        if is_sep and (is_eof or is_followed_by_empty):
            raw_text = "\n".join(lines[last_idx:idx+1]).strip()
            chunks.append(raw_text)
            last_idx = idx + 1

    tasks = sample(chunks, 100)

    results = []
    max_workers = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        poem_tasks = {executor.submit(
            process_single_poem,
            client,
            model,
            cast(List[ChatCompletionMessageParam], cast(object, [{"role": "user", "content": text}])),
            formatted_tools
        ): text for text in tasks}

        for future in as_completed(poem_tasks):
            try:
                data = future.result()
                if data: results.append(data)
            except Exception as e:
                print(f"处理失败：{e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=20),
    reraise=True
)
def regenerate_single_poem(client, model, entry, tools):
    keys_to_keep = ["title", "author", "content", "comment"]
    text = json.dumps({k: v for k, v in entry.items() if k in keys_to_keep}, ensure_ascii=False)

    response = client.chat.completions.create(
        model=model,
        messages=cast(List[ChatCompletionMessageParam], cast(object, [{"role": "user", "content": text}])),
        tools=tools,
        timeout=180
    )

    message = response.choices[0].message

    if message.tool_calls:
        return entry | json.loads(message.tool_calls[0].function.arguments)

    if message.content:
        raise ValueError(f"模型错误输出至content：{message.content}")

    raise ValueError(f"模型无输出。")


def regenerate_tags(input_file, output_file, url, key, model):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "process_poem",
                "description": "根据给出的诗歌信息，生成给定形式的标签。你需要注意，标签不允许重复出现，并且标签不应该出现在诗歌原文中。请仔细检查标签是否在原文中出现过。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "style_tag": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "符合诗歌艺术风格的tag，依据评论文章提取。要求最好为两个字，所有tag的词性最好保持一致。",
                        },
                        "emotion_tag": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "符合诗歌所传达的情感的tag，依据评论文章提取。要求最好为两个字，所有tag的词性最好保持一致。",
                        },
                        "theme_tag": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "符合诗歌思想主旨的tag，依据评论文章提取。要求最好为两个字，所有tag的词性最好保持一致。",
                        },
                        "negative_tag": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 9,
                            "maxItems": 9,
                            "description": "与本诗歌无关，但是常用于现当代诗歌分析的tag。要求最好为两个字。",
                        },
                    },
                    "required": ["style_tag", "emotion_tag", "theme_tag", "negative_tag"]
                },
            }
        },
    ]
    formatted_tools = cast(List[ChatCompletionToolParam], cast(object, tools))

    client = OpenAI(api_key=key, base_url=url)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    max_workers = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        poem_tasks = {executor.submit(
            regenerate_single_poem,
            client,
            model,
            entry,
            formatted_tools
        ): entry for entry in data}

        for future in as_completed(poem_tasks):
            try:
                result = future.result()
                if result: results.append(result)
            except Exception as e:
                print(f"处理失败：{e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)



regenerate_tags(
    "poem_theme_dataset_original.json",
    "poem_theme_dataset_tmp.json",
    "https://api.deepseek.com",
    "key-here",
    "deepseek-reasoner"
)
