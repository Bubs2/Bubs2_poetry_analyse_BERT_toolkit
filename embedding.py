import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from enum import Enum
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel


__all__ = ["EmbOutput", "Embedding"]


class EmbOutput(Enum):
    Official = 0
    MeanPooling = 1


class Embedding:
    def __init__(self, model_name: str, output: EmbOutput = EmbOutput.MeanPooling):
        self.model_name = model_name
        self.output = output

        match output:
            case EmbOutput.Official:
                self.embedding = _EmbeddingOfficial(model_name)
            case EmbOutput.MeanPooling:
                self.embedding = _EmbeddingMeanPooling(model_name)

    def encode(self, text: str):
        return self.embedding.encode(text)

    def decode(self, tokens):
        return self.embedding.decode(tokens)

    def get_vector(self, text: str, span: tuple[int, int]=None):
        """
        非语境模式：返回 text 的 token 平均
        语境模式：输入整段文本 text，以及目标片段的字符区间 [start_char, end_char)
        返回该片段在整段文本语境中的向量（token平均）

        Official Embedding 暂不支持 span 参数，使用 prompt 工程替代。
        """
        return self.embedding.get_vector(text, span)


class _EmbeddingOfficial:
    def __init__(self, model_name: str):
        self.model = BGEM3FlagModel(model_name, use_fp16=torch.cuda.is_available())

    def encode(self, text: str):
        raise NotImplementedError("Official Embedding currently not support encode.")

    def decode(self, tokens):
        raise NotImplementedError("Official Embedding currently not support decode.")

    def get_vector(self, text: str, span: tuple[int, int]=None):
        if span is not None:
            raise NotImplementedError("Official Embedding does not support span. Please use prompt formatting instead.")
        return self.model.encode([text])["dense_vecs"][0]


class _EmbeddingMeanPooling:
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str):
        return self.tokenizer(text, return_offsets_mapping=True)

    def decode(self, tokens):
        return self.tokenizer.convert_ids_to_tokens(tokens["input_ids"])

    def get_vector(self, text: str, span: tuple[int, int]=None):
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True
        )

        offset_mapping = encoded.pop("offset_mapping")[0].cpu().numpy()
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            hidden_states = outputs.hidden_states[-1].squeeze(0)

        if span is None:
            # 非语境模式，直接取除特殊 tokens 外的平均值
            return torch.mean(hidden_states[1:-1], dim=0).cpu().numpy()

        target_token_indices = []
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                continue
            if tok_end > span[0] and tok_start < span[1]:
                target_token_indices.append(idx)

        if not target_token_indices:
            raise ValueError(f"区间 [{span[0]}, {span[1]}) 没有映射到任何 token")

        span_vec = hidden_states[target_token_indices].mean(dim=0).cpu().numpy()
        return span_vec