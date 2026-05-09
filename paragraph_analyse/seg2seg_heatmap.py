import seaborn as sns
import matplotlib.pyplot as plt

from split_poem import SplitMode
from embedding import Embedding
from analyse_helper import ExtractMode, get_all_segment_vectors
from sklearn.metrics.pairwise import cosine_similarity


def plot_heatmap(mode_str, sim_matrix, title="配对相似度热力图"):
    labels = [f"{mode_str}{i + 1}" for i in range(len(sim_matrix[0]))]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        sim_matrix,
        annot=False,
        cmap="coolwarm",
        xticklabels=labels,
        yticklabels=labels,
        square=True
    )
    plt.title(title)
    plt.tight_layout()

    plt.show()

    return sim_matrix


def evaluate_and_print(
    mode_str,
    segment_texts, sim_matrix,
    avg_top_k = 3, pair_top_k = 5
):
    n = len(segment_texts)
    if n == 1: return

    average_similarity = [[idx, 0] for idx in range(n)]
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            average_similarity[i][1] += sim_matrix[i, j]
            average_similarity[j][1] += sim_matrix[i, j]
            pairs.append((i, j, sim_matrix[i, j]))

    average_similarity.sort(key=lambda x: x[1], reverse=True)
    pairs.sort(key=lambda x: x[2], reverse=True)

    print("平均相似度最高")
    for i, sim in average_similarity[:avg_top_k]:
        print(f"[{mode_str}{i+1}] 平均相似度={sim / (n - 1):.4f}")
        print(f"  {mode_str}{i+1}: {segment_texts[i]}")
        print()
    print("配对相似度最高")
    for i, j, sim in pairs[:pair_top_k]:
        print(f"[{mode_str}{i+1} - {mode_str}{j+1}] 相似度={sim:.4f}")
        print(f"  {mode_str}{i+1}: {segment_texts[i]}")
        print(f"  {mode_str}{j+1}: {segment_texts[j]}")
        print()


def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    emb = Embedding("BAAI/bge-m3")

    path = input("输入待分析文件：")
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    mode_str = input("输入分割模式（L为行，P为段）：").strip().upper()
    extract_mode = ExtractMode.Span
    split_mode = SplitMode.Line if mode_str == "L" else SplitMode.Paragraph

    segment_texts, vectors = get_all_segment_vectors(emb, extract_mode, split_mode, text)
    sim_matrix = cosine_similarity(vectors)

    evaluate_and_print(mode_str, segment_texts, sim_matrix)

    plot_heatmap(mode_str, sim_matrix)


if __name__ == '__main__':
    main()