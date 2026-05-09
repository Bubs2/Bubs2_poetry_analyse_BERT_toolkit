import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from helper import compute_poem_losses_batched


def main():
    model_name = "hfl/chinese-bert-wwm-ext"
    path = input("输入待评估的诗歌数据集路径：")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    special_token_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id,
                         tokenizer.pad_token_id, tokenizer.mask_token_id}

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    records = []

    for poem in tqdm(dataset, desc="Evaluating poems"):
        title = poem.get("title", "")
        content = poem.get("content", "")

        encoded = tokenizer(
            f"{title}\n{content}",
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        token_ids = input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        single_losses, same_losses, valid_positions = compute_poem_losses_batched(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_token_ids=special_token_ids,
            batch_size=12,
        )

        for pos in valid_positions:
            token_id = token_ids[pos]
            token = tokens[pos]

            records.append({
                "title": title,
                "position": pos,
                "token": token,
                "token_id": token_id,
                "single_mask_loss": single_losses[pos],
                "same_token_mask_loss": same_losses[token_id],
            })

    df = pd.DataFrame(records)

    df.to_csv(
        "results_temp.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f"
    )

    print(df.head())

    single_loss_mean = df["single_mask_loss"].mean()
    same_loss_mean = df["same_token_mask_loss"].mean()

    print("单 token mask 平均 loss:", single_loss_mean)
    print("同 token mask 平均 loss:", same_loss_mean)


if __name__ == "__main__":
    main()
