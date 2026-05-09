import json
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import PoemBoundaryModel
from dataset import PoemBoundaryDataset, collate_fn


def predict_boundaries(model, tokenizer, samples, device, max_length=512, threshold=0.5):
    dataset = PoemBoundaryDataset(samples, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )

    results = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            boundary_positions = [x.to(device) for x in batch["boundary_positions"]]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                boundary_positions=boundary_positions,
                labels=None
            )

            logits_list = outputs["logits"]

            for positions, logits in zip(boundary_positions, logits_list):
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).long()

                results.append({
                    "boundary_positions": positions.cpu().tolist(),
                    "probs": probs.cpu().tolist(),
                    "preds": preds.cpu().tolist()
                })

    return results


def format_poem(samples, results):
    poems = ""
    for sample, result in zip(samples, results):
        lines = sample["content"].split("\n[B]\n")
        poems += f"{sample["title"]}\n\n{lines[0]}"

        for idx, line in enumerate(lines[1:]):
            if result["preds"][idx]:
                poems += f"\n\n{line}"
            else:
                poems += f"\n{line}"

        poems += "\n\n\n"

    return poems


def main():
    model_name = "BAAI/bge-m3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[B]"]})

    model = PoemBoundaryModel(model_name, tokenizer).to(device)
    checkpoint = torch.load("best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    with open("testset.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    results = predict_boundaries(model, tokenizer, samples, device, threshold=0.5)

    with open("result_tmp.txt", 'w', encoding='utf-8') as f:
        f.write(format_poem(samples, results))


if __name__ == "__main__":
    main()