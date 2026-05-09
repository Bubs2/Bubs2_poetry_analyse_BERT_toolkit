import pandas as pd
import matplotlib.pyplot as plt

def calc_overlap_by_ratio(file1, file2, key_column='index', ratios=None):
    """
    按比例计算两个文件前 k 条数据的重合数，并计算 k^2 / N

    参数:
        file1: 第一个 CSV 文件
        file2: 第二个 CSV 文件
        key_column: 用来判断重合的列名
        ratios: 比例列表，如 [0.1, 0.2, ..., 1.0]

    返回:
        DataFrame，包含每个比例下的结果
    """
    if ratios is None:
        ratios = [i / 10 for i in range(1, 11)]  # 10% ~ 100%

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    if key_column not in df1.columns or key_column not in df2.columns:
        raise KeyError(f"列名 {key_column} 不存在于输入文件中")

    # N：两个文件中 key 的总唯一数
    N = len(set(df1[key_column]).union(set(df2[key_column])))

    # 为了保证两边按相同比例取样，使用较短文件长度作为基准
    base_len = min(len(df1), len(df2))

    results = []

    for ratio in ratios:
        sample_k = max(1, int(base_len * ratio))  # 每个文件取前 sample_k 条

        sample1 = df1.head(sample_k)
        sample2 = df2.head(sample_k)

        set1 = set(sample1[key_column])
        set2 = set(sample2[key_column])

        overlap = len(set1 & set2)

        # 按你的公式计算 k^2 / N
        k2_over_N = (sample_k ** 2) / N if N > 0 else 0

        results.append({
            'ratio': ratio,
            'sample_k': sample_k,
            'overlap': overlap,
            'k2_over_N': k2_over_N
        })

    return pd.DataFrame(results)

def plot_results(result_df, out_file='overlap_chart.png'):
    """
    把结果画成图表
    """
    x = result_df['ratio'] * 100

    plt.figure(figsize=(9, 5))
    plt.plot(x, result_df['overlap'], marker='o', linewidth=2, label='实际重合数 overlap')
    plt.plot(x, result_df['k2_over_N'], marker='s', linewidth=2, label='k^2 / N')

    plt.xticks(x, [f'{int(v)}%' for v in x])
    plt.xlabel('样本比例')
    plt.ylabel('数量')
    plt.title('不同样本比例下的重合数与 k^2 / N')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.show()

def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    result_df = calc_overlap_by_ratio(
        'results.csv',
        'shuffled_results.csv',
        key_column='index'
    )

    print(result_df)

    # 保存结果表
    result_df.to_csv('overlap_ratio_results.csv', index=False, encoding='utf-8-sig')

    # 画图
    plot_results(result_df, out_file='overlap_chart.png')

if __name__ == "__main__":
    main()