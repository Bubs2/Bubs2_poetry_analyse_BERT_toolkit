from split_poem import SplitMode
from embedding import Embedding
from analyse_helper import ExtractMode, get_all_segment_vectors
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_and_print(
    mode_str, text_vector,
    segment_texts, vectors,
    top_k = 5
):
    n = len(segment_texts)
    if n == 1: return

    similarity = [[idx, cosine_similarity(text_vector, vector.reshape(1, -1))[0][0]] for idx, vector in enumerate(vectors)]
    similarity.sort(key=lambda x: x[1], reverse=True)

    print("与全诗相似度最高")
    for i, sim in similarity[:top_k]:
        print(f"[{mode_str}{i+1}] 相似度={sim / (n - 1):.4f}")
        print(f"  {mode_str}{i+1}: {segment_texts[i]}")
        print()


def main():
    emb = Embedding("BAAI/bge-m3")

    path = input("输入待分析文件：")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    mode_str = input("输入分割模式（L为行，P为段）：").strip().upper()
    extract_mode = ExtractMode.Span
    split_mode = SplitMode.Line if mode_str == "L" else SplitMode.Paragraph

    text_vector = emb.get_vector(text).reshape(1, -1)
    segment_texts, vectors = get_all_segment_vectors(emb, extract_mode, split_mode, text)

    evaluate_and_print(mode_str, text_vector, segment_texts, vectors)


if __name__ == '__main__':
    main()
