# scripts/predict_text.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
텍스트 파일 → NER 예측(+confidence) → (옵션) thresholds + 규칙 적용 → 최종 fields 출력

예)
python scripts/predict_text.py `
>> --model_dir outputs/models/koelectra_crf `
>> --text_file input.txt `
>> --thresholds outputs/thresholds.json `
>> --artists_csv data/lexicon/artists.csv `
>> --venues_csv data/lexicon/venues.csv `
>> --rules on `
>> --out outputs/pred.json `
>> --quiet `
>> --strip_internal
"""

import os, sys, io, re, json, argparse

# ── 프로젝트 루트 경로를 sys.path에 주입 (항상 src.* 임포트 가능)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch, warnings
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, logging as hf_logging

from src.models.koelectra_crf import ElectraCRF
from src.rules.postrules import merge_model_and_rules, schema_guard, load_lexicons

def _silence_lib_logs():
    # 라이브러리 경고/로그 최소화
    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    

# ─────────────────────────────────────────────────────────────────────────────
# 간이 토크나이저 (공연 공지 특화)
# ─────────────────────────────────────────────────────────────────────────────
def simple_tokenize(text: str):
    text = re.sub(r"\s+", " ", (text or "").strip())

    price = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\s*(?:원|won|KRW))?"
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

    # 통화 기호(₩/￦)와 KRW 단어를 분리 토큰으로 인식
    token_re = re.compile(
        rf"{weekday_paren}|{time_units}|{date_units}|{price}|[₩￦]|KRW|@[A-Za-z0-9_.]+|#[^\s#@]+|[A-Za-z]+|[가-힣]+|\d+|[^\sA-Za-z0-9가-힣]",
        re.UNICODE
    )
    toks = token_re.findall(text)

    merged = []
    i = 0
    while i < len(toks):
        t = toks[i]

        # 앞에 통화기호가 오고 뒤에 숫자면 결합: '₩' '35,000' -> '35,000'
        if t in ('₩','￦') and i + 1 < len(toks) and re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)", toks[i+1]):
            merged.append(toks[i+1])
            i += 2
            continue

        # '35,000' + ('원'|'won'|'KRW') -> '35,000'
        if i + 1 < len(toks) and toks[i+1].upper() in ("원".upper(), "WON", "KRW"):
            if re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)", t):
                merged.append(t)
                i += 2
                continue

        # 'krw' + '35000' -> '30,000'
        if t.upper() in ("WON", "KRW") and i + 1 < len(toks) and re.fullmatch(r"\d+", toks[i+1]):
            merged.append(f"{int(toks[i+1]):,}")
            i += 2
            continue 

        # '3' '만' '5' '천원' -> '35,000' 로 표준화
        if i + 3 < len(toks) and toks[i+1] == "만" and re.fullmatch(r"\d+", toks[i]) and re.fullmatch(r"\d+", toks[i+2]):
            if toks[i+3].endswith("원"):
                val = int(toks[i]) * 10000 + int(toks[i+2]) * 1000
                merged.append(f"{val:,}")
                i += 4
                continue

        merged.append(t)
        i += 1
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 모델 라벨 로딩
# ─────────────────────────────────────────────────────────────────────────────
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
        if isinstance(id2, dict):
            return [id2[str(i)] if str(i) in id2 else id2[i] for i in range(len(id2))]
        return [id2[i] for i in range(len(id2))]
    raise RuntimeError(f"labels not found under {model_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 토큰 → WordPiece 변환 & 역정렬
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# (rules off용) BIO → fields 간단 변환
# ─────────────────────────────────────────────────────────────────────────────
def _bio_to_spans(tokens, labels):
    spans = []
    cur_type, s = None, None
    for i, lab in enumerate(labels):
        if not lab or lab == "O":
            if cur_type is not None:
                spans.append((cur_type, s, i))
                cur_type, s = None, None
            continue
        if "-" in lab:
            tag, etype = lab.split("-", 1)
        else:
            tag, etype = lab, None
        if tag == "B":
            if cur_type is not None:
                spans.append((cur_type, s, i))
            cur_type, s = etype, i
        elif tag == "I":
            if cur_type != etype:
                if cur_type is not None:
                    spans.append((cur_type, s, i))
                cur_type, s = etype, i
        else:
            if cur_type is not None:
                spans.append((cur_type, s, i))
            cur_type, s = None, None
    if cur_type is not None:
        spans.append((cur_type, s, len(labels)))
    return spans


def _fields_from_bio(tokens, labels):
    fields = {
        "TITLE": [], "DATE": [], "TIME": [],
        "PRICE": [], "PRICE_ONSITE": [], "PRICE_TYPE": [],
        "LINEUP": [], "INSTAGRAM": [],
        "VENUE": [], "V_INSTA": [],
        "TICKET_OPEN_DATE": [], "TICKET_OPEN_TIME": []
    }
    spans = _bio_to_spans(tokens, labels)
    def join(a, b): return " ".join(tokens[a:b]).strip()
    for etype, s, e in spans:
        txt = join(s, e)
        if not txt: continue
        if   etype == 'PRICE':              fields['PRICE'].append(txt)
        elif etype == 'PRICE_ONSITE':       fields['PRICE_ONSITE'].append(txt)
        elif etype == 'PRICE_TYPE':         fields['PRICE_TYPE'].append(txt)
        elif etype == 'LINEUP':             fields['LINEUP'].append(txt)
        elif etype == 'INSTAGRAM':          fields['INSTAGRAM'].append(txt)
        elif etype == 'VENUE':              fields['VENUE'].append(txt)
        elif etype == 'V_INSTA':            fields['V_INSTA'].append(txt)
        elif etype == 'TICKET_OPEN_DATE':   fields['TICKET_OPEN_DATE'].append(txt)
        elif etype == 'TICKET_OPEN_TIME':   fields['TICKET_OPEN_TIME'].append(txt)
        elif etype == 'DATE':               fields['DATE'].append(txt)
        elif etype == 'TIME':               fields['TIME'].append(txt)
        elif etype == 'TITLE':              fields['TITLE'].append(txt)
    return fields


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text_file", required=True, help="분석할 텍스트 파일 경로")
    ap.add_argument("--thresholds", default=None, help="thresholds.json 경로(없으면 규칙만 적용)")
    ap.add_argument("--artists_csv", default=None)
    ap.add_argument("--venues_csv", default=None)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--save_jsonl", default=None, help="RAW 예측(jsonl) 저장 경로(옵션)")
    ap.add_argument("--rules", choices=["on", "off"], default="on",
                    help="정규표현식/사전 후처리 사용 여부 (default: on)")
    ap.add_argument("--out", default=None, help="최종 JSON 결과를 파일로 저장")
    ap.add_argument("--quiet", action="store_true",
                help="최종 JSON만 stdout에 출력(나머지 로그/프린트 숨김)")
    ap.add_argument("--no_tokens", action="store_true",
                help="TOKENS/MODEL PRED 섹션 출력 안 함")
    ap.add_argument("--no_text", action="store_true",
                    help="INPUT TEXT 미리보기 출력 안 함")
    ap.add_argument("--strip_internal", action="store_true",
                    help="최종 JSON에서 tokens/text 키 제거")

    args = ap.parse_args()
    if args.quiet:
        _silence_lib_logs()

    # 입력 텍스트 로드
    if not os.path.exists(args.text_file):
        raise FileNotFoundError(f"--text_file not found: {args.text_file}")
    with io.open(args.text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()
    if raw_text is None or not str(raw_text).strip():
        raise ValueError("입력 파일이 비어있습니다.")

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
    tokens = simple_tokenize(raw_text)
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
                "text": raw_text,
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # rules + thresholds 적용
    thresholds = {}
    if args.thresholds and os.path.exists(args.thresholds):
        with io.open(args.thresholds, "r", encoding="utf-8") as f:
            got = json.load(f)
        thresholds = got.get("thresholds", got)  # {"DATE":0.5,...} or plain dict

    if args.rules == "on":
        lex = load_lexicons(args.artists_csv, args.venues_csv)
        # 원본 텍스트(raw_text)를 규칙 엔진에 전달하여 멀티라인 정규식이 정확히 동작하도록 함
        merged = merge_model_and_rules(
            tokens, pred_labels,
            conf_token, thresholds,
            lexicons=lex,
            raw_text=raw_text
        )
        clean  = schema_guard(merged)
    else:
        # thresholds만 적용하고 BIO 그대로 출력
        def _apply_thresholds(labels_, confs_, thr_):
            if not confs_: return labels_
            out = []
            for lab, conf in zip(labels_, confs_):
                if not lab or lab == 'O':
                    out.append('O'); continue
                etype = lab.split('-', 1)[-1] if '-' in lab else lab
                t = float(thr_.get(etype, 0.0)) if isinstance(thr_, dict) else 0.0
                keep = (conf is None) or (float(conf) >= t)
                out.append(lab if keep else 'O')
            return out

        labels_thr = _apply_thresholds(pred_labels, conf_token, thresholds)
        fields = _fields_from_bio(tokens, labels_thr)
        fields["tokens"] = tokens
        fields["text"] = " ".join(tokens)
        clean = schema_guard(fields)

    # ---- 출력 ----
    # (최종 JSON 만들기 전에 내부 키 제거 옵션 적용)
    if args.strip_internal:
        clean.pop("tokens", None)
        clean.pop("text", None)

    result_str = json.dumps(clean, ensure_ascii=False, indent=2)

    if args.quiet:
        # 최종 JSON만 출력
        print(result_str)
    else:
        # (기존 디버그/미리보기 등 상세 출력)
        # print("\n=== INPUT FILE ===")
        # print(args.text_file)
        # print("\n=== INPUT TEXT (first 300 chars) ===")
        # preview = raw_text[:300].replace("\n", " ")
        # print(preview + ("..." if len(raw_text) > 300 else ""))
        # print("\n=== TOKENS ===")
        # print(tokens)
        # print("\n=== MODEL PRED (token-level) ===")
        # print(list(zip(tokens, pred_labels, [round(c,3) for c in conf_token])))
        # print("\n=== FINAL FIELDS (after thresholds {} rules) ===".format("+" if args.rules=="on" else "without"))
        print(result_str)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with io.open(args.out, "w", encoding="utf-8") as f:
            f.write(result_str)


if __name__ == "__main__":
    main()
