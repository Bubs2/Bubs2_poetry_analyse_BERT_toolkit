import json
from string import Template

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

model.to(device)
model.eval()


def get_vector(text, target_word=None, context_mode=True, layer=-1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True, add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    offsets = inputs['offset_mapping'][0].cpu().numpy()

    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_states = outputs.hidden_states[layer].squeeze(0)

    # --- 概念模式 (非语境) ---
    if not context_mode:
        # 取除 [CLS] 和 [SEP] 外的平均值
        return torch.mean(hidden_states[1:-1], dim=0).cpu().numpy()

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
    target_vectors = hidden_states[target_token_indices]
    final_vector = torch.mean(target_vectors, dim=0).cpu().numpy()

    return final_vector


def process(input_file, output_file, query_prompt, passage_prompt):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []
    pos_neg_correct_count = 0
    pos_adv_correct_count = 0

    query_template = Template(query_prompt)
    passage_template = Template(passage_prompt)

    for entry in data:
        title = entry['title']
        content = entry['content']
        comment = entry['comment']

        content_vec = get_vector(
            passage_template.safe_substitute(title=title, content=content, comment=comment),
            context_mode=False).reshape(1, -1)

        pos_sims = {tag : float(cosine_similarity
                    (content_vec,
                     get_vector(query_template.safe_substitute(tag=tag), context_mode=False).reshape(1, -1))[0][0])
                    for tag in entry['positive_tag']}
        avg_pos = sum(pos_sims.values()) / len(pos_sims)

        neg_sims = {tag : float(cosine_similarity
                    (content_vec,
                     get_vector(query_template.safe_substitute(tag=tag), context_mode=False).reshape(1, -1))[0][0])
                    for tag in entry['negative_tag']}
        avg_neg = sum(neg_sims.values()) / len(neg_sims)

        adv_sims = {tag : float(cosine_similarity
                    (content_vec,
                     get_vector(query_template.safe_substitute(tag=tag), context_mode=False).reshape(1, -1))[0][0])
                    for tag in entry['adversarial_tag']}
        avg_adv = sum(adv_sims.values()) / len(adv_sims)

        result.append({
            "title": title,
            "pos_sims": pos_sims,
            "avg_pos": avg_pos,
            "neg_sims": neg_sims,
            "avg_neg": avg_neg,
            "neg_is_correct": avg_pos > avg_neg,
            "adv_sims": adv_sims,
            "avg_adv": avg_adv,
            "adv_is_correct": avg_pos > avg_adv,
        })
        pos_neg_correct_count += avg_pos > avg_neg
        pos_adv_correct_count += avg_pos > avg_adv

    print(f"entry_num:{len(data)}")
    print(f"pos_neg_correct_count: {pos_neg_correct_count}")
    print(f"pos_adv_correct_count: {pos_adv_correct_count}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"结果已成功写入{output_file}")


input_file_ = input("请输入待处理文件名：")
output_file_ = input("请输入处理结果文件名：")
query_prompt_ = input("请输入模板（使用${tag}作为占位符）：")
content_prompt_ = input("请输入模板（使用${title}，${content}，${comment}作为占位符）：")

process(input_file_, output_file_, query_prompt_, content_prompt_)