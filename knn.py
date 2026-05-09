import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

# 1. 加载模型和分词器
model_name = "hfl/chinese-roberta-wwm-ext"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval() # 设置为评估模式

# 2. 获取 BERT 的全词表静态向量矩阵 (大小为 [21128, 768])
# 这是我们的“日常词典义参照系”
vocab_embeddings = model.embeddings.word_embeddings.weight.detach()

# 3. 输入诗歌文本
# "月亮是天空溃烂的伤口，向人间滴落白色的血。"
# "今天的月亮很圆。"
poem_text = "今天的月亮很圆。"
target_word = "月亮"

inputs = tokenizer(poem_text, return_tensors="pt")
input_ids = inputs['input_ids'][0]

# 动态查找 "月亮" 在 token 列表中的位置
target_ids = tokenizer.encode(target_word, add_special_tokens=False)
indices = []
for i in range(len(input_ids)):
    if input_ids[i:i+len(target_ids)].tolist() == target_ids:
        indices = list(range(i, i+len(target_ids)))
        break

with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state[0]

# 对 "月" 和 "亮" 的向量取平均，得到完整的“月亮”语义
if indices:
    target_vector = last_hidden_states[indices].mean(dim=0)
else:
    raise ValueError("未在文本中找到目标词，请检查分词。")

# 4. 执行 KNN 之前：做一个简单的去中心化（可选，但能显著提高准确度）
mean_vocab = vocab_embeddings.mean(dim=0)
centered_target = target_vector - mean_vocab
centered_vocab = vocab_embeddings - mean_vocab

similarities = F.cosine_similarity(centered_target.unsqueeze(0), centered_vocab)

# 获取相似度最高的 Top 10 个词的索引
K = 10
top_k_scores, top_k_indices = torch.topk(similarities, k=K)

# 5. 打印结果
print(f"诗歌语境中，与目标意象最接近的 {K} 个词汇是：")
for score, idx in zip(top_k_scores, top_k_indices):
    word = tokenizer.convert_ids_to_tokens(idx.item())
    print(f"词汇: {word}, 相似度: {score.item():.4f}")