import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F


def compute_poem_losses_batched(
        model,
        tokenizer,
        input_ids,
        attention_mask,
        special_token_ids,
        batch_size=32,

        top_k=5
):
    """
    返回：
    single_losses: dict[pos] = loss
    same_losses: dict[token_id] = loss
    valid_positions: list[int]
    """

    device = input_ids.device
    mask_token_id = tokenizer.mask_token_id

    token_ids = input_ids[0].tolist()
    seq_len = int(attention_mask.sum().item())

    valid_positions = [
        pos for pos in range(seq_len)
        if token_ids[pos] not in special_token_ids
    ]

    single_losses = {}

    # =========================
    # 1. 单 token mask 批量计算
    # =========================
    for start in range(0, len(valid_positions), batch_size):
        batch_positions = valid_positions[start:start + batch_size]
        bsz = len(batch_positions)

        batch_input_ids = input_ids.repeat(bsz, 1)
        batch_attention_mask = attention_mask.repeat(bsz, 1)

        row_idx = torch.arange(bsz, device=device)
        pos_tensor = torch.tensor(batch_positions, device=device)

        target_ids = batch_input_ids[row_idx, pos_tensor].clone()

        batch_input_ids[row_idx, pos_tensor] = mask_token_id

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask
                    ).logits
            else:
                logits = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                ).logits

        # logits[row_idx, pos_tensor] shape: [bsz, vocab_size]
        losses = F.cross_entropy(
            logits[row_idx, pos_tensor].float(),
            target_ids,
            reduction="none"
        )

        for pos, loss in zip(batch_positions, losses.detach().cpu().tolist()):
            single_losses[pos] = loss

    # =========================
    # 2. same token mask 批量计算
    # =========================
    unique_token_ids = sorted(set(token_ids[pos] for pos in valid_positions))
    same_losses = {}

    for start in range(0, len(unique_token_ids), batch_size):
        batch_token_ids = unique_token_ids[start:start + batch_size]
        bsz = len(batch_token_ids)

        batch_input_ids = input_ids.repeat(bsz, 1)
        batch_attention_mask = attention_mask.repeat(bsz, 1)

        labels = torch.full_like(batch_input_ids, -100)

        for i, tid in enumerate(batch_token_ids):
            target_positions = (
                    (input_ids[0] == tid) &
                    attention_mask[0].bool()
            )

            batch_input_ids[i, target_positions] = mask_token_id
            labels[i, target_positions] = tid

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask
                    ).logits
            else:
                logits = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                ).logits

        # loss_per_pos shape: [bsz, seq_len]
        loss_per_pos = F.cross_entropy(
            logits.transpose(1, 2).float(),
            labels,
            ignore_index=-100,
            reduction="none"
        )

        valid_counts = (labels != -100).sum(dim=1)
        losses = loss_per_pos.sum(dim=1) / valid_counts

        for tid, loss in zip(batch_token_ids, losses.detach().cpu().tolist()):
            same_losses[tid] = loss

    return single_losses, same_losses, valid_positions
