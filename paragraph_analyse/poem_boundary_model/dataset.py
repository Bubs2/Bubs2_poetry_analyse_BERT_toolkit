import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def collate_fn(batch, pad_token_id):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    boundary_positions = [item["boundary_positions"] for item in batch]
    labels = [item["labels"] for item in batch]
    titles = [item["title"] for item in batch]

    return {
        "title": titles,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "boundary_positions": boundary_positions,
        "labels": labels
    }


class PoemBoundaryDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.boundary_token_id = tokenizer.convert_tokens_to_ids("[B]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        content = sample["content"]
        labels = sample["labels"]

        encoded = self.tokenizer(
            content,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        boundary_positions = (input_ids == self.boundary_token_id).nonzero(as_tuple=True)[0]

        if len(boundary_positions) != len(labels):
            raise ValueError(
                f"边界数量不匹配: token中有 {len(boundary_positions)} 个[B]，"
                f"但标签有 {len(labels)} 个。可能是截断导致。标题: {sample.get('title', 'UNKNOWN')}"
            )

        return {
            "title": sample.get("title", "UNKNOWN"),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "boundary_positions": boundary_positions,
            "labels": torch.tensor(labels, dtype=torch.float)
        }

