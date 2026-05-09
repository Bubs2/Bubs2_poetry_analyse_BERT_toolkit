import json

from string import Template
from sklearn.metrics.pairwise import cosine_similarity
from embedding import Embedding
from reranker import Reranker


def get_rank_embedding(emb: Embedding, target: str, dataset, prompt: str):
    rank = []
    target_vector = emb.get_vector(target).reshape(1, -1)

    template = Template(prompt)
    for item in dataset:
        title = item['title']
        content = item['content']
        comment = item['comment']
        text = template.safe_substitute(title=title, content=content, comment=comment)

        item_vector = emb.get_vector(text).reshape(1, -1)
        score = float(cosine_similarity(target_vector, item_vector)[0][0])
        rank.append({"title": item["title"], "score": score})

    rank.sort(key=lambda x: x["score"], reverse=True)

    return rank


def get_rank_reranker(reranker: Reranker, target: str, dataset, prompt: str):
    rank = []

    template = Template(prompt)
    for item in dataset:
        title = item['title']
        content = item['content']
        comment = item['comment']
        text = template.safe_substitute(title=title, content=content, comment=comment)

        score = reranker.get_relevance_score(target, text)
        rank.append({"title": item["title"], "score": score})

    rank.sort(key=lambda x: x["score"], reverse=True)

    return rank


def main():
    target_path = input("输入要查询的诗歌路径（txt）：")
    with open(target_path, 'r', encoding="utf-8") as f:
        target = f.read()

    dataset_path = input("输入诗歌数据库路径（json）：")
    with open(dataset_path, 'r', encoding="utf-8") as f:
        dataset = json.load(f)

    prompt = input("请输入诗歌数据库模板（使用${title}，${content}，${comment}作为占位符）：")

    method = input("输入E使用Embedding，输入R使用Reranker：")
    if method == "E":
        emb = Embedding("BAAI/bge-m3")
        rank = get_rank_embedding(emb, target, dataset, prompt)
    elif method == "R":
        reranker = Reranker("BAAI/bge-reranker-v2-m3")
        rank = get_rank_reranker(reranker, target, dataset, prompt)
    else:
        raise ValueError("Invalid input")

    with open("rank_tmp.json", 'w', encoding='utf-8') as f:
        json.dump(rank, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()