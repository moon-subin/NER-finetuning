# src/data/collate.py
# 데이터로더/콜레이트
import torch

def pad_batch(batch, pad_id: int):
    maxlen = max(len(x["input_ids"]) for x in batch)
    input_ids, masks, labels = [], [], []
    for ex in batch:
        L = len(ex["input_ids"])
        pad = maxlen - L
        input_ids.append(torch.cat([ex["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        masks.append(torch.cat([ex["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        # labels: CRF ignore = -1
        labels.append(torch.cat([ex["labels"], torch.full((pad,), -1, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(masks),
        "labels": torch.stack(labels),
    }
