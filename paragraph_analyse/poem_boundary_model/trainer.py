import json
import random
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import PoemBoundaryDataset, collate_fn
from model import PoemBoundaryModel

model_name = "BAAI/bge-m3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({"additional_special_tokens": ["[B]"]})

with open("./dataset.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

# ===== 1. 划分训练集 / 验证集 =====
random.seed(42)
random.shuffle(samples)

split_idx = int(len(samples) * 0.8)
train_samples = samples[:split_idx]
eval_samples = samples[split_idx:]

print(f"Total samples: {len(samples)}")
print(f"Train samples: {len(train_samples)}")
print(f"Eval samples: {len(eval_samples)}")

# ===== 2. 构建 dataset =====
train_dataset = PoemBoundaryDataset(train_samples, tokenizer, max_length=512)
eval_dataset = PoemBoundaryDataset(eval_samples, tokenizer, max_length=512)

# ===== 3. 构建 dataloader =====
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PoemBoundaryModel(model_name, tokenizer).to(device)
optimizer = AdamW([
    {"params": model.encoder.parameters(), "lr": 2e-5},
    {"params": model.classifier.parameters(), "lr": 5e-5},
], weight_decay=1e-4)

checkpoint = torch.load("checkpoint.pt", map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"]
best_eval_loss = checkpoint["best_eval_loss"]

# ===== 4. 训练 + 验证 =====
for epoch in range(5):
    # ---- train ----
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
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

        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ---- eval ----
    model.eval()
    total_eval_loss = 0.0

    with torch.no_grad():
        for batch in eval_loader:
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

            loss = outputs["loss"]
            total_eval_loss += loss.item()

    avg_eval_loss = total_eval_loss / len(eval_loader)

    if avg_eval_loss < best_eval_loss:
        best_eval_loss = avg_eval_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_eval_loss": best_eval_loss,
        }, "best_checkpoint.pt")
        print("Best checkpoint saved.")

    print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, eval_loss={avg_eval_loss:.4f}")