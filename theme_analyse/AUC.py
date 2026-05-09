import json

input_file = input("输入要计算AUC的文件名（json）：")

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

poem_aucs = []

for entry in data:
    title = entry.get("title", "unknown")

    # 1. 取正例和负例标签名
    positive_tags = []
    positive_tags += list(entry.get("style_scores", {}).keys())
    positive_tags += list(entry.get("emotion_scores", {}).keys())
    positive_tags += list(entry.get("theme_scores", {}).keys())
    negative_tags = list(entry.get("negative_scores", {}).keys())

    # 2. 从 rank 列表构造标签 -> 排名（1-based rank）
    # rank 列表已经是按分数从高到低排好的
    rank_list = entry.get("rank", [])
    tag_to_rank = {}
    for idx, item in enumerate(rank_list, start=1):
        tag = item[0]
        tag_to_rank[tag] = idx

    # 3. 提取正例和负例的排名
    positive_ranks = []
    negative_ranks = []

    for tag in positive_tags:
        if tag in tag_to_rank:
            positive_ranks.append(tag_to_rank[tag])

    for tag in negative_tags:
        if tag in tag_to_rank:
            negative_ranks.append(tag_to_rank[tag])

    if not positive_ranks or not negative_ranks:
        print(f"{title}: 缺少正例或负例，跳过")
        continue

    # 4. 精确计算 AUC
    # rank 越小越好，所以 pos_rank < neg_rank 记为成功
    total = 0
    success = 0
    tie = 0

    for pr in positive_ranks:
        for nr in negative_ranks:
            total += 1
            if pr < nr:
                success += 1
            elif pr == nr:
                tie += 1

    auc = (success + 0.5 * tie) / total
    poem_aucs.append((title, auc))

# 5. 输出结果
if not poem_aucs:
    print("没有可计算的样本")
else:
    mean_auc = sum(x[1] for x in poem_aucs) / len(poem_aucs)

    print("\n每首诗的 AUC：")
    for title, auc in poem_aucs:
        print(f"{title}: {auc:.4f}")

    print(f"\n总样本数: {len(poem_aucs)}")
    print(f"平均 AUC: {mean_auc:.4f}")