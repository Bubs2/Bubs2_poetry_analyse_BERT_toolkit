import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModel.from_pretrained(model_name, local_files_only=True)
model.eval()


def get_word_vector(text, target_word=None, context_mode=True):
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)

    offsets = inputs['offset_mapping'][0].numpy()

    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        last_hidden_states = outputs.last_hidden_state.squeeze(0)

    # --- 概念模式 (非语境) ---
    if not context_mode:
        # 取除 [CLS] 和 [SEP] 外的平均值
        return torch.mean(last_hidden_states, dim=0).numpy()

    # --- 语境模式 ---
    # 1. 在原始字符串中找到 target_word 的字符位置
    start_char = text.find(target_word)
    if start_char == -1:
        raise ValueError(f"错误：在文本中未找到目标词 '{target_word}'")
    end_char = start_char + len(target_word)

    # 2. 遍历 offsets，找到所有与目标词字符区间有重叠的 tokens
    target_token_indices = []

    for idx, (token_start, token_end) in enumerate(offsets):
        # 过滤掉特殊 token（它们的 offset 通常是 (0,0) 或者是整个句子长度，视具体模型而定）
        if token_start == token_end:
            continue

        # 判断重叠逻辑：
        # 如果 token 的范围在目标词的范围内，或者与目标词有交集，则选中
        # 这里的逻辑是：token的结束点 > 目标词起点 AND token的起始点 < 目标词终点
        if token_end > start_char and token_start < end_char:
            target_token_indices.append(idx)

    if not target_token_indices:
        raise ValueError(f"无法将目标词 '{target_word}' 映射到 Token 列表。")

    # print(f"调试信息: '{target_word}' 对应的 Token 索引为: {target_token_indices}")

    # 3. 提取这些 token 的向量并取平均
    target_vectors = last_hidden_states[target_token_indices]
    final_vector = torch.mean(target_vectors, dim=0).numpy()

    return final_vector


context = "苹果很甜。"
target_image = "苹果"

concepts = ["苹果", "水果", "食物", "科技", "电子产品", "公司", "红色", "甜味"]

try:
    vec_target = get_word_vector(context, target_image, context_mode=True)
except ValueError as e:
    print(e)
    exit()

vec_normal = get_word_vector(target_image, context_mode=False)

vec_concepts = {c: get_word_vector(c, context_mode=False) for c in concepts}

print(f"--- 分析 '{target_image}' 在语境中的倾向 (Model: {model_name}) ---")
print(f"{'Concept':<12} | {'Cosine Similarity':<10}")
print("-" * 35)

context_sims = [cosine_similarity(vec_target.reshape(1, -1), vec_concepts[concept].reshape(1, -1))[0][0] for concept in concepts]

for sim, concept in zip(context_sims, concepts):
    print(f"{concept:<12} | {sim:.4f}")

print(f"--- 分析 '{target_image}' 不在语境中的倾向 (Model: {model_name}) ---")
print(f"{'Concept':<12} | {'Cosine Similarity':<10}")
print("-" * 35)

normal_sims = [cosine_similarity(vec_normal.reshape(1, -1), vec_concepts[concept].reshape(1, -1))[0][0] for concept in concepts]

for sim, concept in zip(normal_sims, concepts):
    print(f"{concept:<12} | {sim:.4f}")

print(f"--- 分析 '{target_image}' 的语义保留率 (Model: {model_name}) ---")
print(f"{'Concept':<12} | {'Similarity Reservation':<10}")
print("-" * 35)

for context_sim, normal_sim, concept in zip(context_sims, normal_sims, concepts):
    print(f"{concept:<12} | {context_sim / normal_sim:.4f}")