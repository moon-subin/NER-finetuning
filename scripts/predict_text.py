# scripts/predict_text.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
한 줄 텍스트 → NER 예측(+conf) → thresholds + 규칙 적용 → 최종 fields 출력
사용 예:
  python scripts/predict_text.py ^
    --model_dir outputs/checkpoints/best_model ^
    --text "2025년 6월 26일 저녁 7시 유인원 단독 콘서트 Our Nature, 예매 30,000원 현매 35,000원" ^
    --thresholds outputs/thresholds.json ^
    --artists_csv data/lexicon/artists.csv ^
    --venues_csv data/lexicon/venues.csv
"""
import os, sys, io, re, json, argparse
sys.path.append(os.path.abspath("."))  # allow "src.*" imports

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
    - 시간: 오후 6시 / 낮 12시 / 7:30
    - @handle, 영문/한글 단어, 구두점
    """
    text = re.sub(r"\s+", " ", text.strip())

    price = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:원)?"
    # 요일 괄호 토큰
    weekday_paren = r"\([월화수목금토일]\)"
    # 시간: 오전/오후/낮/밤/저녁/새벽 + '숫자시(분)'
    time_units = r"(?:(?:오전|오후|낮|밤|저녁|새벽)\s*\d{1,2}시(?:\s*\d{1,2}분)?|\d{1,2}:\d{2}|\d{1,2}시(?:\s*\d{1,2}분)?)"
    date_units = r"(?:\d{4}년|\d{1,2}월|\d{1,2}일)"

    token_re = re.compile(
        rf"{weekday_paren}|{time_units}|{date_units}|{price}|@[A-Za-z0-9_.]+|[A-Za-z]+|[가-힣]+|\d+|[^\sA-Za-z0-9가-힣]",
        re.UNICODE
    )

    toks = token_re.findall(text)

    # '35,000' '원' → '35,000원'으로 결합
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
    lab_by_token = ["O"] * (int(align.max().item()) + 1 if (align >= 0).any() else 0)
    for pos, wid in enumerate(label_ids):
        tok_idx = int(align[pos].item())
        if 0 <= tok_idx < len(lab_by_token):
            lab_by_token[tok_idx] = id2label[wid]
    return lab_by_token


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--thresholds", default=None, help="thresholds.json 경로(없으면 규칙만 적용)")
    ap.add_argument("--artists_csv", default=None)
    ap.add_argument("--venues_csv", default=None)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # labels / tokenizer / config / model
    labels = load_labels_from_model_dir(args.model_dir)
    id2label = labels
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    backbone_name = getattr(config, "_name_or_path", None) or args.model_dir

    model = ElectraCRF(backbone_name, num_labels=num_labels).to(device)
    state = torch.load(os.path.join(args.model_dir, "pytorch_model.bin"), map_location=device)
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
    conf_token  = []
    seen=set()
    for pos in range(align.size(0)):
        idx = int(align[pos].item())
        if idx >= 0 and idx not in seen and pos < len(wp_conf):
            conf_token.append(wp_conf[pos]); seen.add(idx)
    while len(conf_token) < len(pred_labels): conf_token.append(0.5)
    if len(conf_token) > len(pred_labels): conf_token = conf_token[:len(pred_labels)]

    # rules + thresholds 적용
    thresholds = {}
    if args.thresholds and os.path.exists(args.thresholds):
        with io.open(args.thresholds, "r", encoding="utf-8") as f:
            got = json.load(f)
        thresholds = got.get("thresholds", got)  # 파일 구조 2가지 케이스 지원

    lex = load_lexicons(args.artists_csv, args.venues_csv)
    merged = merge_model_and_rules(tokens, pred_labels, conf_token, thresholds, lexicons=lex)
    final_doc = schema_guard(merged)

    # 출력
    print("\n=== INPUT TEXT ===")
    print(args.text)
    print("\n=== TOKENS ===")
    print(tokens)
    print("\n=== MODEL PRED (token-level) ===")
    # 짧게 앞부분만 보여주고 싶으면 슬라이스 가능
    print(list(zip(tokens, pred_labels, [round(c,3) for c in conf_token])))

    print("\n=== FINAL FIELDS (after thresholds + rules) ===")
    print(json.dumps(final_doc.get("fields", {}), ensure_ascii=False, indent=2))

    # 원하면 full json도 보기
    # print("\n=== FULL DOC ===")
    # print(json.dumps(final_doc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
