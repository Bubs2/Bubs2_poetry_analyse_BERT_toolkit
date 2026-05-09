from split_poem import SplitMode
from embedding import EmbOutput, Embedding
from analyse_helper import ExtractMode, get_all_segment_vectors
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_and_print(
    mode_str,
    segment_texts, sim_matrix,
    top_k = 5
):
    n = len(segment_texts)

    similarity = [[idx, sim_matrix[0][idx]] for idx in range(n)]
    similarity.sort(key=lambda x: x[1], reverse=True)

    print("相似度最高")
    for i, sim in similarity[:top_k]:
        print(f"[{mode_str}{i+1}] 平均相似度={sim / (n - 1):.4f}")
        print(f"  {mode_str}{i+1}: {segment_texts[i]}")
        print()


def main():
    emb = Embedding("BAAI/bge-m3")

    path = input("输入待分析文件：")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    mode_str = input("输入分割模式（L为行，P为段）：").strip().upper()
    extract_mode = ExtractMode.Prompt
    split_mode = SplitMode.Line if mode_str == "L" else SplitMode.Paragraph

    segment_texts, vectors = get_all_segment_vectors(emb, extract_mode, split_mode, text)
    target_text = input("输入查询语句：")
    target_vec = emb.get_vector(target_text)
    sim_matrix = cosine_similarity(target_vec.reshape(1, -1), vectors)

    evaluate_and_print(mode_str, segment_texts, sim_matrix)


if __name__ == '__main__':
    main()