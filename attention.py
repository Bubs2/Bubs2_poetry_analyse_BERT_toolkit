import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 加上 attn_implementation="eager" 消除警告，强制输出标准 Attention 矩阵
model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
model.eval()


def get_word_token_indices(text, target_word, occurrence=1):
    """
    辅助函数：找到目标词在句子中的 Token 索引。
    """
    inputs = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = inputs['offset_mapping']

    start_char = -1
    for _ in range(occurrence):
        start_char = text.find(target_word, start_char + 1)
        if start_char == -1:
            raise ValueError(f"未找到第 {occurrence} 个 '{target_word}'")

    end_char = start_char + len(target_word)

    target_indices = []
    for idx, (token_start, token_end) in enumerate(offsets):
        if token_start == token_end:
            continue
        if token_end > start_char and token_start < end_char:
            target_indices.append(idx)

    return target_indices


def get_all_heads_attention(text, source_word, target_word, source_occurrence=1, target_occurrence=1):
    """
    计算 source_word 对 target_word 在所有 Attention Head 上的独立注意力权重。
    返回一个包含 num_heads 个元素的 numpy 数组。
    """
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 取最后一层 Attention 矩阵
    # 形状: (batch_size, num_heads, seq_len, seq_len) -> (num_heads, seq_len, seq_len)
    last_layer_attn = outputs.attentions[-1].squeeze(0)
    num_heads = last_layer_attn.shape[0]

    source_indices = get_word_token_indices(text, source_word, source_occurrence)
    target_indices = get_word_token_indices(text, target_word, target_occurrence)

    # 使用网格索引提取 tokens 对应的子矩阵
    grid_x, grid_y = torch.meshgrid(torch.tensor(source_indices), torch.tensor(target_indices), indexing='ij')

    # 提取所有头在指定 tokens 上的子矩阵
    # 形状: (num_heads, len(source_indices), len(target_indices))
    sub_matrices = last_layer_attn[:, grid_x, grid_y]

    # 在 tokens 维度（dim=1和dim=2）上取平均，保留 num_heads 维度
    # 结果形状: (num_heads,)
    scores_per_head = torch.mean(sub_matrices, dim=(1, 2)).numpy()

    return scores_per_head


# ================= 测试与分析 =================
text = """
面朝大海，春暖花开
从明天起，做一个幸福的人
喂马、劈柴，周游世界
从明天起，关心粮食和蔬菜
我有一所房子，面朝大海，春暖花开
从明天起，和每一个亲人通信
告诉他们我的幸福
那幸福的闪电告诉我的
我将告诉每一个人
给每一条河每一座山取一个温暖的名字
陌生人，我也为你祝福
愿你有一个灿烂的前程
愿你有情人终成眷属
愿你在尘世获得幸福
我只愿面朝大海，春暖花开
"""
source = "春暖花开"
source_occurrence = 1

print(f"分析文本: {text}\n")
print(f"观察源: '{source}' 的注意力分布 (共 24 个 Head)\n")

targets = ["幸福", "幸福", "幸福", "幸福", "闪电"]
targets_occurrence = [1, 2, 3, 4, 1]

for target, target_occurrence in zip(targets, targets_occurrence):
    # 获取 24 个头的得分数组
    scores = get_all_heads_attention(text, source_word=source, target_word=target,
                                     source_occurrence=source_occurrence,
                                     target_occurrence=target_occurrence)

    # 找出得分最高的 Top 3 个头及其索引
    top_3_indices = np.argsort(scores)[-3:][::-1]
    top_3_scores = scores[top_3_indices]

    print(f"--- 目标: '{target}' ---")
    print(f"24个头的完整得分数组:\n{np.round(scores, 4)}")
    print(f"【最关注该词的 Top 3 Head】:")
    for rank, (idx, score) in enumerate(zip(top_3_indices, top_3_scores)):
        print(f"  第 {idx:02d} 号 Head : 权重 {score:.4f}")
    print("-" * 40)