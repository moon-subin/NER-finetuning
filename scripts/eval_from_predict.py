# scripts/eval_from_predict.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
통합 평가 스크립트
- 모드 A (BIO/seqeval): CoNLL 정답 vs 모델 예측 → micro/macro-F1 + per-entity report
- 모드 B (필드): gold_jsonl의 text에 대해 rules on/off 예측 비교 → 필드별 P/R/F1, macro-F1, confusion

필요:
  pip install seqeval transformers torch

예시:
  # A) BIO 평가지표
  python scripts/eval_from_predict.py `
    --model_dir outputs/models/koelectra_crf `
    --gold_conll data/processed/dev.conll `
    --labels data/processed/labels.txt `
    --max_len 256

  # B) 필드 평가 (rules on/off 비교)
  python scripts/eval_from_predict.py `
    --model_dir outputs/models/koelectra_crf `
    --gold_jsonl data/processed/field_gold.jsonl `
    --thresholds outputs/thresholds.json `
    --artists_csv data/lexicon/artists.csv `
    --venues_csv data/lexicon/venues.csv `
    --max_len 256
"""

import sys, os, io, re, json, argparse
from typing import List, Tuple, Dict, Optional, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# 모델/룰
from src.models.koelectra_crf import ElectraCRF
from src.rules.postrules import merge_model_and_rules, schema_guard, load_lexicons

# ─────────────────────────────────────────────────────────────────────────────
# 공통: 모델/토크나이저 로드
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(model_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        backbone_name = model_dir
    except Exception:
        backbone_name = getattr(config, "_name_or_path", None) or "monologg/koelectra-base-v3-discriminator"
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    # labels
    labels_path = os.path.join(model_dir, "labels.txt")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.txt not found in {model_dir}")
    with io.open(labels_path, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in enumerate(labels)}
    # model
    model = ElectraCRF(backbone_name, num_labels=len(labels)).to(device)
    state_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"model weights not found: {state_path}")
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tokenizer, labels, id2label, label2id, device

# ─────────────────────────────────────────────────────────────────────────────
# 모드 A: BIO/seqeval 평가 (CoNLL)
# ─────────────────────────────────────────────────────────────────────────────
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
            tok, lab = (parts[0], parts[1]) if len(parts) > 1 else (parts[0], "O")
            cur_x.append(tok); cur_y.append(lab)
    if cur_x:
        sents.append(cur_x); tags.append(cur_y)
    return sents, tags

def predict_labels_for_tokens(model, tokenizer, tokens: List[str], labels: List[str], device, max_len=256) -> List[str]:
    """CoNLL 토큰 시퀀스 그대로 받아 WordPiece→CRF 디코딩→첫 sub-token 라벨만 사용."""
    # WP ids/labels 정렬 (gold 라벨은 필요 없지만 길이/정렬 동일하게)
    wp_ids = [tokenizer.cls_token_id]
    align = [-1]
    for i, tok in enumerate(tokens):
        pieces = tokenizer.tokenize(tok) or [tokenizer.unk_token]
        ids = tokenizer.convert_tokens_to_ids(pieces)
        wp_ids.extend(ids)
        align.extend([i] * len(pieces))
    wp_ids.append(tokenizer.sep_token_id); align.append(-1)
    if len(wp_ids) > max_len:
        wp_ids = wp_ids[:max_len]; align = align[:max_len]
    attn = [1] * len(wp_ids)

    input_ids = torch.tensor(wp_ids).long().unsqueeze(0).to(device)
    attn_t    = torch.tensor(attn).long().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model.backbone(input_ids=input_ids, attention_mask=attn_t)
        seq_out = model.dropout(outputs.last_hidden_state)
        logits  = model.classifier(seq_out)   # [1, T_wp, C]
        paths   = model.crf.decode(logits, mask=attn_t.bool())
        wp_pred = paths[0]

    # WP → token 첫 sub-token 위치 label만 사용
    max_tok = max([a for a in align if a >= 0], default=-1)
    pred_by_tok = ["O"] * (max_tok + 1 if max_tok >= 0 else 0)
    seen = set()
    for pos, lid in enumerate(wp_pred):
        tok_idx = int(align[pos]) if pos < len(align) else -1
        if 0 <= tok_idx < len(pred_by_tok) and tok_idx not in seen:
            pred_by_tok[tok_idx] = labels[lid]
            seen.add(tok_idx)
    return pred_by_tok

def run_seqeval(model_dir: str, gold_conll: str, labels_path: str, max_len=256):
    model, tokenizer, labels, id2label, label2id, device = load_model_and_tokenizer(model_dir)
    sents, gold = read_conll(gold_conll)

    y_true, y_pred = [], []
    for toks, glabs in zip(sents, gold):
        preds = predict_labels_for_tokens(model, tokenizer, toks, labels, device, max_len=max_len)
        # 길이 차이가 생기면 짧은 쪽 기준으로 자르기
        L = min(len(glabs), len(preds))
        y_true.append(glabs[:L]); y_pred.append(preds[:L])

    P = precision_score(y_true, y_pred, average="micro")
    R = recall_score(y_true, y_pred, average="micro")
    F_micro = f1_score(y_true, y_pred, average="micro")
    F_macro = f1_score(y_true, y_pred, average="macro")
    print("===== BIO / seqeval =====")
    print(f"micro-Precision={P:.4f} micro-Recall={R:.4f} micro-F1={F_micro:.4f}")
    print(f"macro-F1={F_macro:.4f}")
    print("\n-- classification report --")
    print(classification_report(y_true, y_pred, digits=4))

# ─────────────────────────────────────────────────────────────────────────────
# 모드 B: 필드 평가 (rules on/off 비교)
# gold_jsonl 포맷 예:
# {"text": "...", "TITLE":[], "DATE":[...], "TIME":[...], ...}
# ─────────────────────────────────────────────────────────────────────────────
FIELD_KEYS = ["TITLE","DATE","TIME","PRICE","PRICE_ONSITE","PRICE_TYPE",
              "LINEUP","INSTAGRAM","VENUE","V_INSTA","TICKET_OPEN_DATE","TICKET_OPEN_TIME"]

def simple_tokenize_for_eval(text: str) -> List[str]:
    """predict_text.py와 유사: 날짜/시간/가격/핸들/해시태그/한/영/숫자/구두점"""
    text = re.sub(r"\s+", " ", (text or "").strip())
    price = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\s*(?:원|won))?"
    weekday_paren = r"\((?:[월화수목금토일]|Mon|Tue|Wed|Thu|Fri|Sat|Sun|MON|TUE|WED|THU|FRI|SAT|SUN)\)"
    time_units = (
        r"(?:(?:오전|오후|낮|밤|저녁)\s*\d{1,2}시(?:\s*\d{1,2}분)?"
        r"|\d{1,2}:\d{2}(?:\s*(?:am|pm|AM|PM))?"
        r"|\d{1,2}\s*(?:am|pm|AM|PM)"
        r"|\d{1,2}시(?:\s*\d{1,2}분)?)"
    )
    date_units = (
        r"(?:\d{4}년|\d{1,2}월|\d{1,2}일"
        r"|\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}"
        r"|\d{1,2}\s*\.\s*\d{1,2}"
        r"|\d{4}/\d{1,2}/\d{1,2}"
        r"|\d{1,2}/\d{1,2})"
    )
    token_re = re.compile(
        rf"{weekday_paren}|{time_units}|{date_units}|{price}|@[A-Za-z0-9_.]+|#[`\s#@]+|[A-Za-z]+|[가-힣]+|\d+|[`\sA-Za-z0-9가-힣]",
        re.UNICODE
    )
    toks = token_re.findall(text)
    # 금액 결합
    merged, i = [], 0
    while i < len(toks):
        t = toks[i]
        if i + 1 < len(toks) and toks[i+1].lower() in ("원","won"):
            if re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)", t):
                merged.append(t + "원"); i += 2; continue
        merged.append(t); i += 1
    return merged

def tokens_to_wp(tokenizer, tokens, max_len: int):
    ids = [tokenizer.cls_token_id]; align = [-1]
    for i, tok in enumerate(tokens):
        pieces = tokenizer.tokenize(tok) or [tokenizer.unk_token]
        pids = tokenizer.convert_tokens_to_ids(pieces)
        ids.extend(pids); align.extend([i] * len(pids))
    ids.append(tokenizer.sep_token_id); align.append(-1)
    if len(ids) > max_len:
        ids = ids[:max_len]; align = align[:max_len]
    attn = [1] * len(ids)
    return torch.tensor(ids).long(), torch.tensor(attn).long(), torch.tensor(align).long()

def decode_wp_to_token_labels(wp_pred: List[int], align, id2label: List[str]) -> List[str]:
    max_tok = int(align.max().item()) if (align >= 0).any() else -1
    lab_by_token = ["O"] * (max_tok + 1 if max_tok >= 0 else 0)
    seen = set()
    for pos, lid in enumerate(wp_pred):
        tok_idx = int(align[pos].item())
        if 0 <= tok_idx < len(lab_by_token) and tok_idx not in seen:
            lab_by_token[tok_idx] = id2label[lid]; seen.add(tok_idx)
    return lab_by_token

def _bio_to_spans(tokens: List[str], labels: List[str]) -> List[Tuple[str,int,int]]:
    spans = []; cur_type, s = None, None
    for i, lab in enumerate(labels):
        if not lab or lab == 'O':
            if cur_type is not None: spans.append((cur_type, s, i)); cur_type, s = None, None
            continue
        tag, etype = (lab.split('-',1)+[None])[:2] if '-' in lab else (lab, None)
        if tag == 'B':
            if cur_type is not None: spans.append((cur_type, s, i))
            cur_type, s = etype, i
        elif tag == 'I':
            if cur_type != etype:
                if cur_type is not None: spans.append((cur_type, s, i))
                cur_type, s = etype, i
        else:
            if cur_type is not None: spans.append((cur_type, s, i))
            cur_type, s = None, None
    if cur_type is not None: spans.append((cur_type, s, len(labels)))
    return spans

def fields_from_bio(tokens: List[str], labels: List[str]) -> Dict[str, List[str]]:
    fields = {k: [] for k in FIELD_KEYS}
    spans = _bio_to_spans(tokens, labels)
    def join(a,b): return ' '.join(tokens[a:b]).strip()
    for etype, s, e in spans:
        seg = join(s,e)
        if not seg: continue
        if etype in fields:
            fields[etype].append(seg)
        elif etype == 'DATE': fields['DATE'].append(seg)
        elif etype == 'TIME': fields['TIME'].append(seg)
        elif etype == 'TITLE': fields['TITLE'].append(seg)
    return fields

# ── 필드 정규화(간단): 공백/대소문자/통화/조사/해시태그 등
JOSA_TAIL = re.compile(r'(와|과|이|가|는|은|도|를|을|과의|와의)$')
def norm_field_item(field: str, v: str) -> str:
    s = re.sub(r'\s+', ' ', (v or '')).strip()
    if field in ('PRICE','PRICE_ONSITE'):
        s = s.replace('won','원').replace('Won','원')
        # 금액만 추출
        m = re.search(r'(\d{1,3}(?:,\d{3})+|\d+)\s*원', s)
        if m:
            val = int(m.group(1).replace(',', ''))
            return f"{val:,}원"
    if field in ('LINEUP','VENUE','TITLE'):
        s = s.lstrip('#')
        s = re.sub(r'[\"\'“”‘’]', '', s)
        s = JOSA_TAIL.sub('', s)
    return s

def setify_field(field: str, arr: List[str]) -> Set[str]:
    return {norm_field_item(field, x) for x in (arr or []) if norm_field_item(field, x)}

def prf(tp, fp, fn):
    P = tp / (tp+fp) if (tp+fp)>0 else 0.0
    R = tp / (tp+fn) if (tp+fn)>0 else 0.0
    F = 2*P*R/(P+R) if (P+R)>0 else 0.0
    return P,R,F

def run_field_eval(model_dir: str, gold_jsonl: str, thresholds_path: Optional[str], artists_csv: Optional[str],
                   venues_csv: Optional[str], max_len=256):
    # 모델/토크나이저
    model, tokenizer, labels, id2label_list, label2id, device = load_model_and_tokenizer(model_dir)
    id2label = labels

    # thresholds/lexicons
    thresholds = {}
    if thresholds_path and os.path.exists(thresholds_path):
        with io.open(thresholds_path, "r", encoding="utf-8") as f:
            got = json.load(f)
        thresholds = got.get("thresholds", got)
    lex = load_lexicons(artists_csv, venues_csv)

    # 집계
    agg = {k: {"tp":0,"fp":0,"fn":0} for k in FIELD_KEYS}
    confusion = {}  # (pred_type -> true_type) 카운트
    on_off_scores = {k: {"on":{"tp":0,"fp":0,"fn":0},"off":{"tp":0,"fp":0,"fn":0}} for k in FIELD_KEYS}

    with io.open(gold_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            gold_obj = json.loads(line)
            text = gold_obj.get("text","")
            tokens = simple_tokenize_for_eval(text)
            # 모델 추론
            input_ids, attn, align = tokens_to_wp(tokenizer, tokens, max_len=max_len)
            input_ids = input_ids.unsqueeze(0).to(device)
            attn_t    = attn.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model.backbone(input_ids=input_ids, attention_mask=attn_t)
                seq_out = model.dropout(outputs.last_hidden_state)
                logits  = model.classifier(seq_out)
                paths   = model.crf.decode(logits, mask=attn_t.bool())
                wp_pred = paths[0]
            pred_labels_tok = decode_wp_to_token_labels(wp_pred, align, id2label)

            # rules OFF
            fields_off = fields_from_bio(tokens, pred_labels_tok)
            fields_off = schema_guard({"text": " ".join(tokens), "tokens": tokens, **fields_off})

            # rules ON
            # (confidence/thresholds는 생략해도 무방; 필요하면 softmax 계산해서 넘길 수 있음)
            fields_on  = merge_model_and_rules(tokens, pred_labels_tok, confidences=None,
                                               thresholds=thresholds, lexicons=lex)
            fields_on  = schema_guard(fields_on)

            # per-field 집계 (on/off 모두)
            for mode, pred in (("on", fields_on), ("off", fields_off)):
                for k in FIELD_KEYS:
                    gset = setify_field(k, gold_obj.get(k, []))
                    pset = setify_field(k, pred.get(k, []))
                    tp = len(gset & pset)
                    fp = len(pset - gset)
                    fn = len(gset - pset)
                    on_off_scores[k][mode]["tp"] += tp
                    on_off_scores[k][mode]["fp"] += fp
                    on_off_scores[k][mode]["fn"] += fn

            # confusion: 잘못 예측된 문자열이 다른 필드로 들어간 경우 카운트
            # (rules ON 기준으로 집계)
            for pk in FIELD_KEYS:
                pset = setify_field(pk, fields_on.get(pk, []))
                for pv in pset:
                    # 정답에서는 어떤 필드에 있었나?
                    true_type = None
                    for tk in FIELD_KEYS:
                        if pv in setify_field(tk, gold_obj.get(tk, [])):
                            true_type = tk; break
                    if true_type and true_type != pk:
                        confusion.setdefault(pk, {}).setdefault(true_type, 0)
                        confusion[pk][true_type] += 1

    # 출력: per-field P/R/F1 (on/off), macro-F1, confusion
    print("===== FIELD-LEVEL EVAL (rules ON vs OFF) =====")
    macro_on, macro_off = [], []
    header = f"{'FIELD':18} | {'ON_P':>6} {'ON_R':>6} {'ON_F1':>6} || {'OFF_P':>6} {'OFF_R':>6} {'OFF_F1':>6}"
    print(header); print("-"*len(header))
    for k in FIELD_KEYS:
        on = on_off_scores[k]["on"]; off = on_off_scores[k]["off"]
        P_on,R_on,F_on = prf(on["tp"], on["fp"], on["fn"])
        P_off,R_off,F_off = prf(off["tp"], off["fp"], off["fn"])
        macro_on.append(F_on); macro_off.append(F_off)
        print(f"{k:18} | {P_on:6.3f} {R_on:6.3f} {F_on:6.3f} || {P_off:6.3f} {R_off:6.3f} {F_off:6.3f}")
    print("-"*len(header))
    print(f"MACRO-F1 (rules ON) : {sum(macro_on)/len(macro_on):.4f}")
    print(f"MACRO-F1 (rules OFF): {sum(macro_off)/len(macro_off):.4f}")

    # confusion 표
    if confusion:
        print("\n-- Per-entity confusion (rules ON, predicted → true) --")
        # 정렬 출력
        for pk in sorted(confusion.keys()):
            row = [(tk, cnt) for tk, cnt in confusion[pk].items()]
            row.sort(key=lambda x: -x[1])
            cells = ", ".join([f"{tk}:{cnt}" for tk, cnt in row])
            print(f"{pk:18} -> {cells}")

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_len", type=int, default=256)

    # 모드 A (BIO)
    ap.add_argument("--gold_conll", default=None, help="BIO 평가용 CoNLL 파일 경로")
    ap.add_argument("--labels", default=None, help="labels.txt (생략시 model_dir/labels.txt 사용)")

    # 모드 B (필드)
    ap.add_argument("--gold_jsonl", default=None, help='필드 평가용 jsonl (각 줄에 {"text":..., FIELD: [...]})')
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--artists_csv", default=None)
    ap.add_argument("--venues_csv", default=None)

    args = ap.parse_args()

    ran = False
    if args.gold_conll:
        run_seqeval(args.model_dir, args.gold_conll, args.labels or os.path.join(args.model_dir, "labels.txt"), max_len=args.max_len)
        ran = True
    if args.gold_jsonl:
        run_field_eval(args.model_dir, args.gold_jsonl, args.thresholds, args.artists_csv, args.venues_csv, max_len=args.max_len)
        ran = True
    if not ran:
        raise SystemExit("하나 이상 입력 필요: --gold_conll (BIO) 또는 --gold_jsonl (필드)")

if __name__ == "__main__":
    main()
