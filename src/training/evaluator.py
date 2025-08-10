# src/training/evaluator.py
# dev/test 평가
from typing import List, Tuple
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from ..training.metrics import seq_f1

def wp_align_decode(paths_wp: List[List[int]], align, id2label: List[str]) -> List[str]:
    n_tok = (align >= 0).max().item() + 1 if (align >= 0).any() else 0
    tok_labels = ["O"] * n_tok
    for pos, lab_id in enumerate(paths_wp):
        t_idx = int(align[pos].item())
        if 0 <= t_idx < n_tok and tok_labels[t_idx] == "O":
            tok_labels[t_idx] = id2label[int(lab_id)]
    return tok_labels

def evaluate(model, dataloader, tokenizer: AutoTokenizer, id2label: List[str], device="cpu"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval"):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            gold = batch["labels"].cpu().tolist()
            paths = model(input_ids, attn)  # list(list)
            for b in range(len(paths)):
                # build align: we didn't keep align in dataset; use first-subword-only heuristic
                # labels already include O for subwords; so compare at wordpiece-level by mapping to tokens:
                # simple: take gold and predicted where gold!=-1 and compare positionally
                g = [gid for gid in gold[b] if gid != -1]
                p = paths[b][:len(g)]
                # map ids to labels
                y_true.append([id2label[gid] for gid in g])
                y_pred.append([id2label[pid] for pid in p])
    return {"micro_f1": seq_f1(y_true, y_pred, average="micro")}
