import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModelForSequenceClassification

__all__ = ["Reranker"]


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str):
        return self.model.encode(text)

    def decode(self, tokens):
        return self.model.decode(tokens)

    def get_relevance_score(self, text1: str, text2: str):
        # Cross-Encoder 的标准输入是将两个文本拼接在一起
        inputs = self.tokenizer(text1, text2, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # 模型直接输出 logits，代表相关性强度
            logits = self.model(**inputs).logits
            score = logits.view(-1).float().cpu().item()
        return score