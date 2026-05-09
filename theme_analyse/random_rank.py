import json
import random
from copy import deepcopy


def simulate_one_round(sample_size=100):
    """
    单次随机排序模拟：
    - sample_size 个样本
    - 每个样本标签组数量固定为 3-3-3-9
    """
    evaluation = {
        'style_tag': {
            'count': 0,
            'top_3': 0,
            'top_6': 0,
            'top_9': 0,
            'rank': 0,
        },
        'emotion_tag': {
            'count': 0,
            'top_3': 0,
            'top_6': 0,
            'top_9': 0,
            'rank': 0,
        },
        'theme_tag': {
            'count': 0,
            'top_3': 0,
            'top_6': 0,
            'top_9': 0,
            'rank': 0,
        },
        'negative_tag': {
            'count': 0,
            'top_3': 0,
            'top_6': 0,
            'top_9': 0,
            'rank': 0,
        },
    }

    # 每个样本固定 3-3-3-9
    tag_pool = (
            ['style_tag'] * 3 +
            ['emotion_tag'] * 3 +
            ['theme_tag'] * 3 +
            ['negative_tag'] * 9
    )

    for _ in range(sample_size):
        shuffled = tag_pool[:]
        random.shuffle(shuffled)  # 随机排序 18 个标签

        for idx, category in enumerate(shuffled):
            stats = evaluation[category]
            stats['count'] += 1
            stats['top_3'] += int(idx < 3)
            stats['top_6'] += int(idx < 6)
            stats['top_9'] += int(idx < 9)
            stats['rank'] += idx + 1  # rank 从 1 开始

    # 归一化
    for category, stats in evaluation.items():
        if stats['count'] == 0:
            continue
        stats['top_3'] /= stats['count']
        stats['top_6'] /= stats['count']
        stats['top_9'] /= stats['count']
        stats['rank'] /= stats['count']

    return evaluation


def average_results(results_list):
    """
    对多轮模拟结果求平均
    """
    avg = deepcopy(results_list[0])

    for category in avg:
        for metric in avg[category]:
            avg[category][metric] = 0

    n = len(results_list)

    for result in results_list:
        for category in result:
            for metric in result[category]:
                avg[category][metric] += result[category][metric]

    for category in avg:
        for metric in avg[category]:
            avg[category][metric] /= n

    return avg


def simulate_random_baseline(sample_size=100, rounds=1000, output_file=None, seed=42):
    """
    多轮随机模拟，得到更稳定的随机基线
    """
    random.seed(seed)

    all_results = []
    for _ in range(rounds):
        result = simulate_one_round(sample_size=sample_size)
        all_results.append(result)

    avg_result = average_results(all_results)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(avg_result, f, ensure_ascii=False, indent=4)

    return avg_result



baseline = simulate_random_baseline(
    sample_size=100,
    rounds=1000,
    output_file="random_baseline_100samples.json",
    seed=42
)