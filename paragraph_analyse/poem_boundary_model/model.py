import torch
import torch.nn as nn
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModel

class PoemBoundaryModel(nn.Module):
    def __init__(self, model_name, tokenizer, dropout=0.1):
        super().__init__()

        self.tokenizer = tokenizer
        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.encoder.resize_token_embeddings(len(tokenizer))

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, boundary_positions, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = outputs.last_hidden_state  # [B, T, H]

        batch_logits = []
        batch_losses = []

        loss_fct = nn.BCEWithLogitsLoss()

        for i in range(input_ids.size(0)):
            positions = boundary_positions[i]  # [num_boundaries]
            states = last_hidden_state[i, positions, :]  # [num_boundaries, H]
            states = self.dropout(states)
            logits = self.classifier(states).squeeze(-1)  # [num_boundaries]

            batch_logits.append(logits)

            if labels is not None:
                label = labels[i].to(logits.device)
                loss = loss_fct(logits, label)
                batch_losses.append(loss)

        output = {"logits": batch_logits}

        if labels is not None:
            output["loss"] = torch.stack(batch_losses).mean()

        return output

    def get_vector(self, text: str, span: tuple[int, int]=None):
        device = next(self.parameters()).device

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True
        )

        offset_mapping = encoded.pop("offset_mapping")[0].cpu().numpy()
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.encoder(**encoded)
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