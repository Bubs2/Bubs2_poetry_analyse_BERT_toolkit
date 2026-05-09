import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from string import Template

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)

model.to(device)
model.eval()


def get_relevance_score(text1, text2):
    # Cross-Encoder 的标准输入是将两个文本拼接在一起
    inputs = tokenizer(text1, text2, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # 模型直接输出 logits，代表相关性强度
        logits = model(**inputs).logits
        score = logits.view(-1).float().cpu().item()
    return score


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

        pos_scores = {tag : float(get_relevance_score
                            (query_template.safe_substitute(tag=tag),
                             passage_template.safe_substitute(title=title, content=content, comment=comment)))
                      for tag in entry['positive_tag']}
        avg_pos = sum(pos_scores.values()) / len(pos_scores)

        neg_scores = {tag: float(get_relevance_score
                            (query_template.safe_substitute(tag=tag),
                             passage_template.safe_substitute(title=title, content=content, comment=comment)))
                      for tag in entry['negative_tag']}
        avg_neg = sum(neg_scores.values()) / len(neg_scores)

        adv_scores = {tag: float(get_relevance_score
                            (query_template.safe_substitute(tag=tag),
                             passage_template.safe_substitute(title=title, content=content, comment=comment)))
                      for tag in entry['adversarial_tag']}
        avg_adv = sum(adv_scores.values()) / len(adv_scores)

        merged_scores = {**pos_scores, **neg_scores, **adv_scores}
        sorted_scores = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)

        result.append({
            "title": title,
            "pos_scores": pos_scores,
            "avg_pos": avg_pos,
            "neg_scores": neg_scores,
            "avg_neg": avg_neg,
            "neg_is_correct": avg_pos > avg_neg,
            "adv_scores": adv_scores,
            "avg_adv": avg_adv,
            "adv_is_correct": avg_pos > avg_adv,
            "rank": sorted_scores
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