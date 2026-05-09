import json

from embedding import Embedding
from sklearn.metrics.pairwise import cosine_similarity

emb = Embedding("BAAI/bge-m3")


def process(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []

    for entry in data:
        title = entry['title']
        content = entry['content']

        content_vec = emb.get_vector(f"{title}\n\n{content}").reshape(1, -1)

        scores = {comment: float(cosine_similarity(content_vec, emb.get_vector(comment).reshape(1, -1))[0][0])
                  for comment in entry['comment']}

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        result.append({
            "title": title,
            "scores": scores,
            "rank": sorted_scores
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"结果已成功写入{output_file}")


def main():
    input_file = input("请输入待处理文件名：")
    output_file = input("请输入处理结果文件名：")

    process(input_file, output_file)


if __name__ == '__main__':
    main()
