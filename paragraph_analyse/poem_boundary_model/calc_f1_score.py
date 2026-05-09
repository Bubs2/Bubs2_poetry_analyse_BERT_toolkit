import json
import random
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import PoemBoundaryDataset, collate_fn
from model import PoemBoundaryModel


def evaluate_boundary_metrics_sklearn(model, dataloader, device, threshold=0.5):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            boundary_positions = [x.to(device) for x in batch["boundary_positions"]]
            labels = [x.to(device) for x in batch["labels"]]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_positions=boundary_positions,
                labels=labels
            )

            batch_logits = outputs["logits"]

            for logits, label in zip(batch_logits, labels):
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).long()

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(label.long().cpu().tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0)
    }


def main():
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

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[B]"]})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PoemBoundaryModel("BAAI/bge-m3", tokenizer).to(device)
    checkpoint = torch.load("best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_dataset = PoemBoundaryDataset(eval_samples, tokenizer, max_length=512)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )

    print(evaluate_boundary_metrics_sklearn(model, eval_loader, device))


if __name__ == "__main__":
    main()