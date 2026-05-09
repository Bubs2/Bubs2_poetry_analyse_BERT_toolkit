import json
import math
import random


def collect_all_labels(samples):
    all_labels = []
    for sample in samples:
        all_labels.extend(sample["labels"])
    return all_labels


def compute_binary_metrics(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 长度不一致。")

    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)

    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_binary_baselines(train_samples, eval_samples, eps=1e-12, threshold=0.5, seed=42):
    # 收集所有边界标签
    train_labels = collect_all_labels(train_samples)
    eval_labels = collect_all_labels(eval_samples)

    if len(train_labels) == 0:
        raise ValueError("训练集没有任何 labels，无法计算 baseline。")
    if len(eval_labels) == 0:
        raise ValueError("验证集没有任何 labels，无法计算 baseline。")

    # 确保标签为 int
    train_labels = [int(y) for y in train_labels]
    eval_labels = [int(y) for y in eval_labels]

    # 训练集正类比例 π
    pi = sum(train_labels) / len(train_labels)

    # 数值稳定
    pi_clamped = min(max(pi, eps), 1 - eps)

    # =========================
    # 1) 随机猜测 baseline: p=0.5
    # =========================
    random_bce = math.log(2)
    random_acc = 0.5
    random_true_conf = 0.5

    # 理论随机标签预测：每个位置以 0.5 概率预测为 1
    # 为了可复现，这里使用模拟方式得到 precision/recall/F1
    rng = random.Random(seed)
    random_eval_pred = [1 if rng.random() < 0.5 else 0 for _ in eval_labels]
    random_metrics_eval = compute_binary_metrics(eval_labels, random_eval_pred)

    # 也可以用理论期望：
    # 若真实正类比例为 r，则随机 p=0.5 的期望 precision ≈ r, recall ≈ 0.5,
    # F1 ≈ 2*r*0.5/(r+0.5)，但有限样本下模拟更直观。

    # =========================
    # 2) 训练集先验 baseline: 常数预测 p=π
    # =========================
    prior_bce_train = -(
        pi_clamped * math.log(pi_clamped) +
        (1 - pi_clamped) * math.log(1 - pi_clamped)
    )

    eval_pos_rate = sum(eval_labels) / len(eval_labels)
    eval_neg_rate = 1 - eval_pos_rate

    prior_bce_eval = -(
        eval_pos_rate * math.log(pi_clamped) +
        eval_neg_rate * math.log(1 - pi_clamped)
    )

    # 先验概率经过 threshold 转成类别
    prior_pred_class = 1 if pi >= threshold else 0

    prior_train_pred = [prior_pred_class for _ in train_labels]
    prior_eval_pred = [prior_pred_class for _ in eval_labels]

    prior_metrics_train = compute_binary_metrics(train_labels, prior_train_pred)
    prior_metrics_eval = compute_binary_metrics(eval_labels, prior_eval_pred)

    # =========================
    # 3) 多数类 baseline
    # =========================
    majority_class = 1 if pi >= 0.5 else 0

    majority_train_pred = [majority_class for _ in train_labels]
    majority_eval_pred = [majority_class for _ in eval_labels]

    majority_metrics_train = compute_binary_metrics(train_labels, majority_train_pred)
    majority_metrics_eval = compute_binary_metrics(eval_labels, majority_eval_pred)

    # 先验模型的“预测类别置信度”
    pred_conf_baseline = max(pi, 1 - pi)

    # 先验模型在训练集上的平均 true confidence
    avg_true_conf_train = pi * pi + (1 - pi) * (1 - pi)

    # 先验模型在验证集上的平均 true confidence
    avg_true_conf_eval = eval_pos_rate * pi + eval_neg_rate * (1 - pi)

    results = {
        "train_label_count": len(train_labels),
        "eval_label_count": len(eval_labels),
        "train_pos_rate_pi": pi,
        "eval_pos_rate": eval_pos_rate,

        "random_bce_baseline": random_bce,
        "random_acc_baseline": random_acc,
        "random_true_conf_baseline": random_true_conf,
        "random_metrics_eval": random_metrics_eval,

        "prior_bce_baseline_train": prior_bce_train,
        "prior_bce_baseline_eval": prior_bce_eval,
        "prior_pred_class_from_threshold": prior_pred_class,
        "prior_metrics_train": prior_metrics_train,
        "prior_metrics_eval": prior_metrics_eval,

        "majority_class_from_train": majority_class,
        "majority_metrics_train": majority_metrics_train,
        "majority_metrics_eval": majority_metrics_eval,

        "prior_pred_conf_baseline": pred_conf_baseline,
        "prior_avg_true_conf_train": avg_true_conf_train,
        "prior_avg_true_conf_eval": avg_true_conf_eval,
    }

    return results


def print_metric_block(name, metrics):
    print(f"---- {name} ----")
    print(f"Accuracy:  {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall:    {metrics['recall']:.6f}")
    print(f"F1:        {metrics['f1']:.6f}")
    print(f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']} | TN: {metrics['tn']}")
    print()

def print_baselines(results):
    print("===== Baseline Statistics =====")
    print(f"Train label count: {results['train_label_count']}")
    print(f"Eval label count:  {results['eval_label_count']}")
    print(f"Train positive rate (pi): {results['train_pos_rate_pi']:.6f}")
    print(f"Eval positive rate:       {results['eval_pos_rate']:.6f}")
    print()

    print("---- Random Baseline (p=0.5) ----")
    print(f"BCE loss baseline:   {results['random_bce_baseline']:.6f}")
    print(f"Accuracy baseline:   {results['random_acc_baseline']:.6f}")
    print(f"True conf baseline:  {results['random_true_conf_baseline']:.6f}")
    print()

    print_metric_block(
        "Random Baseline Metrics on Eval",
        results["random_metrics_eval"]
    )

    print("---- Prior Baseline (predict constant p=pi from train) ----")
    print(f"BCE baseline on train: {results['prior_bce_baseline_train']:.6f}")
    print(f"BCE baseline on eval:  {results['prior_bce_baseline_eval']:.6f}")
    print(f"Prior predicted class by threshold: {results['prior_pred_class_from_threshold']}")
    print(f"Pred-class confidence baseline: {results['prior_pred_conf_baseline']:.6f}")
    print(f"Avg true confidence on train:   {results['prior_avg_true_conf_train']:.6f}")
    print(f"Avg true confidence on eval:    {results['prior_avg_true_conf_eval']:.6f}")
    print()

    print_metric_block(
        "Prior Baseline Metrics on Train",
        results["prior_metrics_train"]
    )

    print_metric_block(
        "Prior Baseline Metrics on Eval",
        results["prior_metrics_eval"]
    )

    print("---- Majority-Class Baseline ----")
    print(f"Predicted majority class: {results['majority_class_from_train']}")
    print()

    print_metric_block(
        "Majority Baseline Metrics on Train",
        results["majority_metrics_train"]
    )

    print_metric_block(
        "Majority Baseline Metrics on Eval",
        results["majority_metrics_eval"]
    )


with open("./dataset.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

random.seed(42)
random.shuffle(samples)

split_idx = int(len(samples) * 0.8)
train_samples = samples[:split_idx]
eval_samples = samples[split_idx:]

print(f"Total samples: {len(samples)}")
print(f"Train samples: {len(train_samples)}")
print(f"Eval samples: {len(eval_samples)}")

baseline_results = compute_binary_baselines(train_samples, eval_samples)
print_baselines(baseline_results)