# scripts/predict_with_conf.py
import os, io, json, argparse
from typing import List, Tuple, Dict, Any

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

try:
    from torchcrf import CRF
except Exception as e:
    raise RuntimeError("torchcrf가 필요합니다. pip install torchcrf (가능하면 --upgrade)")

# ----------------------------
# Model
# ----------------------------
class ElectraCRF(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        # 최신 torchcrf는 batch_first 지원. 구버전이면 업그레이드 권장.
        self.crf = CRF(num_labels, batch_first=True)

# ----------------------------
# Utils
# ----------------------------
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

def load_labels(model_dir: str) -> List[str]:
    label_map_path = os.path.join(model_dir, "label_map.json")
    labels_txt_path = os.path.join(model_dir, "labels.txt")
    if os.path.exists(label_map_path):
        with io.open(label_map_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        id2label = m.get("id2label") or {}
        if id2label:
            return [id2label[str(i)] for i in range(len(id2label))]
    with io.open(labels_txt_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def tokens_to_wp(tokenizer, tokens: List[str], max_len: int):
    ids = [tokenizer.cls_token_id]
    align = [-1]  # CLS
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
    return (torch.tensor(ids).long(),
            torch.tensor(attn).long(),
            torch.tensor(align).long())

def wp_to_token_labels_and_conf(
    wp_label_ids: List[int],
    wp_conf: List[float],
    align: torch.Tensor,
    id2label: List[str]
):
    # 각 원토큰의 "첫 subword" 위치의 라벨/확률만 사용
    n_tok = (align >= 0).max().item() + 1 if (align >= 0).any() else 0
    tok_labels = ["O"] * n_tok
    tok_confs  = [0.0] * n_tok
    for pos, lab_id in enumerate(wp_label_ids):
        t_idx = int(align[pos].item())
        if 0 <= t_idx < n_tok and tok_labels[t_idx] == "O":
            tok_labels[t_idx] = id2label[int(lab_id)]
            tok_confs[t_idx]  = float(wp_conf[pos])
    return tok_labels, tok_confs

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ----------------------------
# Inference with confidences (CRF marginal -> fallback softmax)
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input_conll", required=True)   # dev.conll 권장
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = load_labels(args.model_dir)
    num_labels = len(labels)
    id2label = labels

    config = AutoConfig.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    backbone_name = getattr(config, "_name_or_path", None) or args.model_dir

    model = ElectraCRF(args.model_dir, num_labels=num_labels).to(device)
    state = torch.load(os.path.join(args.model_dir, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    sents, golds = read_conll(args.input_conll)

    rows = []
    softmax = nn.Softmax(dim=-1)

    with torch.no_grad():
        for tokens, gold in zip(sents, golds):
            input_ids, attn, align = tokens_to_wp(tokenizer, tokens, max_len=args.max_len)
            input_ids = input_ids.unsqueeze(0).to(device)   # [1, T]
            attn      = attn.unsqueeze(0).to(device)
            mask      = attn.bool()

            # emissions
            outputs = model.backbone(input_ids=input_ids, attention_mask=attn)
            seq_out = model.dropout(outputs.last_hidden_state)
            logits  = model.classifier(seq_out)                 # [1, T, C]

            # CRF decode (WP 단위 라벨 id 시퀀스)
            wp_paths = model.crf.decode(logits, mask=mask)      # List[List[int]]
            wp_ids   = wp_paths[0]                               # [T]

            # confidences: 1) CRF marginal이 있으면 그걸로, 2) 없으면 softmax로 폴백
            wp_conf = []
            use_marginal = hasattr(model.crf, "compute_marginal_probabilities")
            if use_marginal:
                marg = model.crf.compute_marginal_probabilities(logits, mask=mask)  # [1, T, C]
                marg = marg[0]  # [T, C]
                for pos, lab_id in enumerate(wp_ids):
                    if pos < marg.shape[0]:
                        wp_conf.append(float(marg[pos, int(lab_id)].item()))
                    else:
                        wp_conf.append(1.0)
            else:
                probs = softmax(logits[0])  # [T, C]
                for pos, lab_id in enumerate(wp_ids):
                    if pos < probs.shape[0]:
                        wp_conf.append(float(probs[pos, int(lab_id)].item()))
                    else:
                        wp_conf.append(1.0)

            # WP → token 정렬
            pred_labels_tok, conf_tok = wp_to_token_labels_and_conf(wp_ids, wp_conf, align, id2label)

            # 길이 보정
            n = min(len(tokens), len(pred_labels_tok), len(gold))
            rows.append({
                "tokens": tokens[:n],
                "model_labels": pred_labels_tok[:n],
                "confidences": conf_tok[:n],
                "gold_labels": gold[:n],
            })

    write_jsonl(args.out_jsonl, rows)
    print(f"[DONE] wrote {args.out_jsonl}  ({len(rows)} examples)")

if __name__ == "__main__":
    main()
