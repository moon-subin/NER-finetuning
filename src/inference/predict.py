# src/inference/predict.py
# 단일/배치 추론 → BIO
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KoELECTRA + CRF 추론
- dev/test용 CoNLL 입력 → BIO 예측
- --with_conf: softmax(logits) 기반 토큰 단위 confidence
- --include_gold: 입력이 라벨을 포함하면 gold도 같이 저장
"""
import os
import io
import json
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig

from ..models.koelectra_crf import ElectraCRF

# --------- 파일 로딩 ---------
def read_conll_tokens_and_gold(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    CoNLL: 토큰만 또는 '토큰 공백 라벨' 형식 모두 지원
    반환: (tokens_list, gold_list or [])
    """
    sents, gold = [], []
    cur_x, cur_y = [], []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur_x:
                    sents.append(cur_x); gold.append(cur_y if cur_y else [])
                    cur_x, cur_y = [], []
                continue
            parts = line.split()
            if len(parts) == 1:
                tok, lab = parts[0], None
            else:
                tok, lab = parts[0], parts[1]
            cur_x.append(tok)
            if lab is not None:
                cur_y.append(lab)
    if cur_x:
        sents.append(cur_x); gold.append(cur_y if cur_y else [])
    return sents, gold

def tokens_to_wp(tokenizer, tokens: List[str], max_len: int):
    ids = [tokenizer.cls_token_id]
    align = [-1]
    for i, tok in enumerate(tokens):
        pieces = tokenizer.tokenize(tok) or [tokenizer.unk_token]
        pids = tokenizer.convert_tokens_to_ids(pieces)
        ids.extend(pids)
        align.extend([i] * len(pids))
    ids.append(tokenizer.sep_token_id)
    align.append(-1)
    if len(ids) > max_len:
        ids = ids[:max_len]
        align = align[:max_len]
    attn = [1] * len(ids)
    return torch.tensor(ids).long(), torch.tensor(attn).long(), torch.tensor(align).long()

def decode_wp(labels_idx: List[int], align: torch.Tensor, id2label: List[str]) -> List[str]:
    # 첫 sub-token 위치의 라벨만 사용
    lab_by_token = ["O"] * (int(align.max().item()) + 1 if (align >= 0).any() else 0)
    for pos, wid in enumerate(labels_idx):
        tok_idx = int(align[pos].item())
        if 0 <= tok_idx < len(lab_by_token):
            lab_by_token[tok_idx] = id2label[wid]
    return lab_by_token

# --------- labels 로더 ---------
def load_labels_from_model_dir(model_dir: str) -> List[str]:
    # 1) label_map.json
    p1 = os.path.join(model_dir, "label_map.json")
    if os.path.exists(p1):
        with io.open(p1, "r", encoding="utf-8") as f:
            m = json.load(f)
        id2 = m.get("id2label") or {}
        if id2:
            return [id2[str(i)] for i in range(len(id2))]

    # 2) labels.txt
    p2 = os.path.join(model_dir, "labels.txt")
    if os.path.exists(p2):
        with io.open(p2, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    # 3) config.json
    cfg = AutoConfig.from_pretrained(model_dir)
    if getattr(cfg, "id2label", None):
        id2 = cfg.id2label
        return [id2[i] for i in range(len(id2))]

    raise RuntimeError(f"labels not found under {model_dir}")

# --------- 메인 ---------
def predict_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_conll", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--with_conf", action="store_true")
    ap.add_argument("--include_gold", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # labels / tokenizer / config
    labels = load_labels_from_model_dir(args.model_dir)
    num_labels = len(labels)
    id2label = labels

    config = AutoConfig.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    backbone_name = getattr(config, "_name_or_path", None) or args.model_dir

    # 모델 로드
    model = ElectraCRF(backbone_name, num_labels=num_labels).to(device)
    state = torch.load(os.path.join(args.model_dir, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # 입력 로드
    sents, gold = read_conll_tokens_and_gold(args.input_conll)

    rows = []
    with torch.no_grad():
        for i, tokens in enumerate(sents):
            input_ids, attn, align = tokens_to_wp(tokenizer, tokens, max_len=args.max_len)
            input_ids = input_ids.unsqueeze(0).to(device)
            attn = attn.unsqueeze(0).to(device)

            # backbone + classifier logits
            outputs = model.backbone(input_ids=input_ids, attention_mask=attn)
            seq_out = model.dropout(outputs.last_hidden_state)
            logits  = model.classifier(seq_out)  # [1, T_wp, C]

            # CRF decode
            paths = model.crf.decode(logits, mask=attn.bool())
            wp_pred = paths[0]

            # confidences: softmax(logits)에서 예측 라벨 확률
            probs = F.softmax(logits[0], dim=-1)  # [T_wp, C]
            L = min(len(wp_pred), probs.size(0))
            wp_conf = [float(probs[t, wp_pred[t]].item()) for t in range(L)]

            # WP → token 정렬 (첫 sub-token만 사용)
            preds_token = decode_wp(wp_pred, align, id2label)
            conf_token = []
            seen = set()
            for pos in range(align.size(0)):
                idx = int(align[pos].item())
                if idx >= 0 and idx not in seen and pos < len(wp_conf):
                    conf_token.append(wp_conf[pos]); seen.add(idx)
            while len(conf_token) < len(preds_token): conf_token.append(0.5)
            if len(conf_token) > len(preds_token): conf_token = conf_token[:len(preds_token)]

            row = {"tokens": tokens, "labels": preds_token}
            if args.with_conf:
                row["confidences"] = conf_token
            if args.include_gold and i < len(gold) and gold[i]:
                m = min(len(gold[i]), len(preds_token))
                row["gold_labels"] = gold[i][:m]
            rows.append(row)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with io.open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] wrote {args.out_jsonl}  ({len(rows)} examples)")

if __name__ == "__main__":
    predict_main()
