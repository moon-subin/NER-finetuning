# scripts/predict_text.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
한 줄 텍스트 → NER 예측(+confidence) → thresholds + 규칙 적용 → 최종 fields 출력

예)
  python scripts/predict_text.py ^
    --model_dir outputs/models/koelectra_crf ^
    --text "2025년 6월 26일 저녁 7시 유인원 단독 콘서트 Our Nature, 예매 30,000원 현매 35,000원" ^
    --thresholds outputs/thresholds.json ^
    --artists_csv data/lexicon/artists.csv ^
    --venues_csv data/lexicon/venues.csv ^
    --save_jsonl outputs/predictions/custom_text.jsonl
"""

import os, sys, io, re, json, argparse

# ── 프로젝트 루트 경로를 sys.path에 주입 (항상 src.* 임포트 가능)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig

from src.models.koelectra_crf import ElectraCRF
from src.rules.regex_postrules import merge_model_and_rules, schema_guard, load_lexicons


def simple_tokenize(text: str):
    """
    한국 공연 공지 특화 간이 토크나이저:
    - 가격: 30,000원 / 45,000원
    - 날짜: 2025년 / 9월 / 21일 / (일)
    - 시간: 오전/오후/낮/밤/저녁/새벽 + '숫자시(분)' 또는 12:30
    - @handle, 영문/한글 단어, 구두점
    """
    text = re.sub(r"\s+", " ", text.strip())

    price = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:원)?"
    weekday_paren = r"\([월화수목금토일]\)"
    time_units = r"(?:(?:오전|오후|낮|밤|저녁|새벽)\s*\d{1,2}시(?:\s*\d{1,2}분)?|\d{1,2}:\d{2}|\d{1,2}시(?:\s*\d{1,2}분)?)"
    date_units = r"(?:\d{4}년|\d{1,2}월|\d{1,2}일)"

    token_re = re.compile(
        rf"{weekday_paren}|{time_units}|{date_units}|{price}|@[A-Za-z0-9_.]+|[A-Za-z]+|[가-힣]+|\d+|[^\sA-Za-z0-9가-힣]",
        re.UNICODE
    )

    toks = token_re.findall(text)

    # '35,000' '원' → '35,000원' 결합
    merged = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if i + 1 < len(toks) and toks[i+1] == "원" and re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)", t):
            merged.append(t + "원")
            i += 2
            continue
        merged.append(t)
        i += 1
    return merged


def load_labels_from_model_dir(model_dir: str):
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
        # id2가 dict일 수 있으므로 index 순서 보장
        if isinstance(id2, dict):
            return [id2[str(i)] if str(i) in id2 else id2[i] for i in range(len(id2))]
        return [id2[i] for i in range(len(id2))]
    raise RuntimeError(f"labels not found under {model_dir}")


def tokens_to_wp(tokenizer, tokens, max_len: int):
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
        ids   = ids[:max_len]
        align = align[:max_len]
    attn = [1] * len(ids)
    return torch.tensor(ids).long(), torch.tensor(attn).long(), torch.tensor(align).long()


def decode_wp(label_ids, align, id2label):
    # align의 최초 등장 sub-token 위치의 라벨만 취함
    max_tok = int(align.max().item()) if (align >= 0).any() else -1
    lab_by_token = ["O"] * (max_tok + 1 if max_tok >= 0 else 0)
    seen = set()
    for pos, wid in enumerate(label_ids):
        tok_idx = int(align[pos].item())
        if 0 <= tok_idx < len(lab_by_token) and tok_idx not in seen:
            lab_by_token[tok_idx] = id2label[wid]
            seen.add(tok_idx)
    return lab_by_token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--thresholds", default=None, help="thresholds.json 경로(없으면 규칙만 적용)")
    ap.add_argument("--artists_csv", default=None)
    ap.add_argument("--venues_csv", default=None)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--save_jsonl", default=None, help="RAW 예측(jsonl) 저장 경로(옵션)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # labels / tokenizer / config / model
    labels = load_labels_from_model_dir(args.model_dir)
    id2label = labels
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(args.model_dir)
    # 토크나이저가 model_dir에 없으면 backbone 이름으로 재시도
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:
        backbone_name = getattr(config, "_name_or_path", None) or "monologg/koelectra-base-v3-discriminator"
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    backbone_name = getattr(config, "_name_or_path", None) or args.model_dir
    model = ElectraCRF(backbone_name, num_labels=num_labels).to(device)

    state_path = os.path.join(args.model_dir, "pytorch_model.bin")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"model weights not found: {state_path}")
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # tokenize input text
    tokens = simple_tokenize(args.text)
    input_ids, attn, align = tokens_to_wp(tokenizer, tokens, max_len=args.max_len)
    input_ids = input_ids.unsqueeze(0).to(device)
    attn      = attn.unsqueeze(0).to(device)

    # forward + logits
    with torch.no_grad():
        outputs = model.backbone(input_ids=input_ids, attention_mask=attn)
        seq_out = model.dropout(outputs.last_hidden_state)
        logits  = model.classifier(seq_out)  # [1, T_wp, C]
        paths   = model.crf.decode(logits, mask=attn.bool())
        wp_pred = paths[0]  # list[int]

        probs  = F.softmax(logits[0], dim=-1)  # [T_wp, C]
        L      = min(len(wp_pred), probs.size(0))
        wp_conf = [float(probs[t, wp_pred[t]].item()) for t in range(L)]

    # WP → token (첫 sub-token만 사용)
    pred_labels = decode_wp(wp_pred, align, id2label)

    # 토큰별 confidence 정렬
    conf_token  = []
    seen=set()
    for pos in range(align.size(0)):
        idx = int(align[pos].item())
        if idx >= 0 and idx not in seen and (pos < len(wp_conf)):
            conf_token.append(wp_conf[pos]); seen.add(idx)
    # 길이 보정
    while len(conf_token) < len(pred_labels): conf_token.append(0.5)
    if len(conf_token) > len(pred_labels): conf_token = conf_token[:len(pred_labels)]

    # RAW 저장(option) — apply_rules.py에서 재사용 가능
    if args.save_jsonl:
        os.makedirs(os.path.dirname(args.save_jsonl) or ".", exist_ok=True)
        with io.open(args.save_jsonl, "w", encoding="utf-8") as f:
            ex = {
                "tokens": tokens,
                "model_labels": pred_labels,
                "confidences": conf_token,
                "text": args.text,
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # rules + thresholds 적용
    thresholds = {}
    if args.thresholds and os.path.exists(args.thresholds):
        with io.open(args.thresholds, "r", encoding="utf-8") as f:
            got = json.load(f)
        thresholds = got.get("thresholds", got)  # {"DATE":0.5,...} or plain dict

    lex = load_lexicons(args.artists_csv, args.venues_csv)
    merged = merge_model_and_rules(tokens, pred_labels, conf_token, thresholds, lexicons=lex)
    clean  = schema_guard(merged)

    # 출력
    print("\n=== INPUT TEXT ===")
    print(args.text)
    print("\n=== TOKENS ===")
    print(tokens)
    print("\n=== MODEL PRED (token-level) ===")
    print(list(zip(tokens, pred_labels, [round(c,3) for c in conf_token])))

    print("\n=== FINAL FIELDS (after thresholds + rules) ===")
    print(json.dumps(clean, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
