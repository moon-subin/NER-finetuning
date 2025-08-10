# src/training/train.py
# 학습 루프(평가/저장 포함)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KoELECTRA + CRF NER 학습 루프 (best dev F1 저장)
- seqeval로 dev micro-F1 측정
- labels.txt / label_map.json / config / tokenizer 저장
"""
import os
import io
import json
import argparse
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from seqeval.metrics import f1_score

# 모델
from ..models.koelectra_crf import ElectraCRF

# 재현성
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# --------- 데이터 유틸 (간단 내장 버전) ---------
def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    sents, tags = [], []
    cur_x, cur_y = [], []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur_x:
                    sents.append(cur_x); tags.append(cur_y)
                    cur_x, cur_y = [], []
                continue
            parts = line.split()
            if len(parts) == 1:
                tok, lab = parts[0], "O"
            else:
                tok, lab = parts[0], parts[1]
            cur_x.append(tok); cur_y.append(lab)
    if cur_x:
        sents.append(cur_x); tags.append(cur_y)
    return sents, tags

def read_labels(path: str) -> List[str]:
    with io.open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

class ConllDataset(Dataset):
    def __init__(self, sents, tags, tokenizer, label2id, max_len=256):
        self.sents = sents; self.tags = tags
        self.tk = tokenizer; self.label2id = label2id
        self.max_len = max_len

    def __len__(self): return len(self.sents)

    def __getitem__(self, idx):
        tokens = self.sents[idx]
        labels = self.tags[idx]

        wp_ids = [self.tk.cls_token_id]
        wp_labels = [self.label2id["O"]]

        for tok, lab in zip(tokens, labels):
            pieces = self.tk.tokenize(tok) or [self.tk.unk_token]
            ids = self.tk.convert_tokens_to_ids(pieces)
            wp_ids.extend(ids)
            lab_id = self.label2id.get(lab, self.label2id["O"])
            wp_labels.append(lab_id)
            for _ in pieces[1:]:
                wp_labels.append(self.label2id["O"])

        wp_ids.append(self.tk.sep_token_id)
        wp_labels.append(self.label2id["O"])

        if len(wp_ids) > self.max_len:
            wp_ids = wp_ids[:self.max_len]
            wp_labels = wp_labels[:self.max_len]

        attn = [1] * len(wp_ids)
        return {
            "input_ids": torch.tensor(wp_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(wp_labels, dtype=torch.long),
        }

def pad_batch(batch, pad_id: int):
    maxlen = max(len(x["input_ids"]) for x in batch)
    input_ids, masks, labels = [], [], []
    for ex in batch:
        L = len(ex["input_ids"]); pad = maxlen - L
        input_ids.append(torch.cat([ex["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        masks.append(torch.cat([ex["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        labels.append(torch.cat([ex["labels"], torch.full((pad,), -1, dtype=torch.long)]))  # -1 mask
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(masks),
        "labels": torch.stack(labels),
    }

# --------- 평가 ---------
def eval_f1(model: nn.Module, loader: DataLoader, id2label_list: List[str], device) -> float:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            paths = model(batch["input_ids"], batch["attention_mask"])  # list[list[int]]
            for b, path in enumerate(paths):
                # gold_wp: -1 제외
                gold_wp = [gid.item() for gid in batch["labels"][b] if gid.item() != -1]
                pred_wp = path[:len(gold_wp)]
                y_true.append([id2label_list[g] for g in gold_wp])
                y_pred.append([id2label_list[p] for p in pred_wp])
    return f1_score(y_true, y_pred, average="micro")

# --------- 메인 ---------
def train_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="monologg/koelectra-small-discriminator")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # labels / 매핑
    labels = read_labels(args.labels)
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in enumerate(labels)}

    # tokenizer / config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    config.id2label = {int(i): l for i, l in id2label.items()}
    config.label2id = {l: int(i) for l, i in label2id.items()}

    # data
    tr_x, tr_y = read_conll(args.train)
    dv_x, dv_y = read_conll(args.dev)

    train_ds = ConllDataset(tr_x, tr_y, tokenizer, label2id, max_len=args.max_len)
    dev_ds   = ConllDataset(dv_x, dv_y, tokenizer, label2id, max_len=args.max_len)
    collate = lambda b: pad_batch(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=True,  collate_fn=collate, num_workers=0)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.bsz, shuffle=False, collate_fn=collate, num_workers=0)

    # model / opt
    model = ElectraCRF(args.model_name, num_labels=len(labels)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item()
            if step % 50 == 0:
                print(f"[epoch {epoch}] step {step} loss {total/step:.4f}")

        dev_f1 = eval_f1(model, dev_loader, labels, device)
        print(f"[epoch {epoch}] train_loss={total/len(train_loader):.4f} dev_micro_f1={dev_f1:.4f}")

        # best만 저장
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.state_dict(), os.path.join(args.out_dir, "pytorch_model.bin"))
            with io.open(os.path.join(args.out_dir, "best.json"), "w", encoding="utf-8") as f:
                json.dump({"dev_micro_f1": float(dev_f1), "epoch": epoch}, f, ensure_ascii=False, indent=2)
            print(f"[save] new best at epoch {epoch}: F1={dev_f1:.4f}")

    # 부가 파일 저장
    with io.open(os.path.join(args.out_dir, "labels.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(labels))

    with io.open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"id2label": config.id2label, "label2id": config.label2id}, f, ensure_ascii=False, indent=2)

    config.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"[DONE] saved to {args.out_dir}")
