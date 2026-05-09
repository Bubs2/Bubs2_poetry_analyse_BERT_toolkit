import json
from string import Template

from embedding import EmbOutput, Embedding
from sklearn.metrics.pairwise import cosine_similarity

emb = Embedding("BAAI/bge-m3", EmbOutput.Official)


def process(input_file, output_file, query_prompt, passage_prompt):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []
    evaluation = {
        'style_tag': {
            'count': 0,
            'top_3': 0.0,
            'top_6': 0.0,
            'top_9': 0.0,
            'rank': 0.0,
            'best': ["", 1000.0],
        },
        'emotion_tag': {
            'count': 0,
            'top_3': 0.0,
            'top_6': 0.0,
            'top_9': 0.0,
            'rank': 0.0,
            'best': ["", 1000.0],
        },
        'theme_tag': {
            'count': 0,
            'top_3': 0.0,
            'top_6': 0.0,
            'top_9': 0.0,
            'rank': 0.0,
            'best': ["", 1000.0],
        },
        'negative_tag': {
            'count': 0,
            'top_3': 0.0,
            'top_6': 0.0,
            'top_9': 0.0,
            'rank': 0.0,
            'best': ["", 1000.0],
        },
    }
    result.append(evaluation)

    query_template = Template(query_prompt)
    passage_template = Template(passage_prompt)

    for entry in data:
        title = entry['title']
        content = entry['content']
        comment = entry['comment']
        passage = passage_template.safe_substitute(title=title, content=content, comment=comment)

        content_vec = emb.get_vector(passage).reshape(1, -1)

        style_scores = {tag: float(cosine_similarity(content_vec, emb.get_vector(query_template.safe_substitute(tag=tag)).reshape(1, -1))[0][0])
                        for tag in entry['style_tag']}

        emotion_scores = {tag: float(cosine_similarity(content_vec, emb.get_vector(query_template.safe_substitute(tag=tag)).reshape(1, -1))[0][0])
                          for tag in entry['emotion_tag']}

        theme_scores = {tag: float(cosine_similarity(content_vec, emb.get_vector(query_template.safe_substitute(tag=tag)).reshape(1, -1))[0][0])
                        for tag in entry['theme_tag']}

        negative_scores = {tag: float(cosine_similarity(content_vec, emb.get_vector(query_template.safe_substitute(tag=tag)).reshape(1, -1))[0][0])
                           for tag in entry['negative_tag']}

        merged_scores = style_scores | emotion_scores | theme_scores | negative_scores
        sorted_scores = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)

        poem_evaluation = {
            'style_tag': [0, 0],
            'emotion_tag': [0, 0],
            'theme_tag': [0, 0],
            'negative_tag': [0, 0],
        }

        for idx, (tag, score) in enumerate(sorted_scores):
            for category, stats in evaluation.items():
                if not tag in entry[category]: continue
                poem_evaluation[category][0] += 1
                poem_evaluation[category][1] += idx + 1
                stats['count'] += 1
                stats['top_3'] += idx < 3
                stats['top_6'] += idx < 6
                stats['top_9'] += idx < 9
                stats['rank'] += idx + 1

        for category, stats in poem_evaluation.items():
            if stats[0] == 0: continue
            if stats[1] / stats[0] >= evaluation[category]['best'][1]: continue
            evaluation[category]['best'][0] = title
            evaluation[category]['best'][1] = stats[1] / stats[0]

        result.append({
            "title": title,
            "style_scores": style_scores,
            "emotion_scores": emotion_scores,
            "theme_scores": theme_scores,
            "negative_scores": negative_scores,
            "rank": sorted_scores
        })

    for category, stats in evaluation.items():
        if stats['count'] == 0: continue
        stats['top_3'] /= stats['count']
        stats['top_6'] /= stats['count']
        stats['top_9'] /= stats['count']
        stats['rank'] /= stats['count']

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"结果已成功写入{output_file}")


def main():
    input_file = input("请输入待处理文件名：")
    output_file = input("请输入处理结果文件名：")
    query_prompt = input("请输入模板（使用${tag}作为占位符）：")
    content_prompt = input("请输入模板（使用${title}，${content}，${comment}作为占位符）：")

    process(input_file, output_file, query_prompt, content_prompt)


if __name__ == '__main__':
    main()