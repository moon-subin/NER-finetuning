# src/rules/regex_postrules.py
# 날짜/시간/가격/인스타 핸들/라인업/티켓오픈 보정 규칙 모음

from __future__ import annotations
import re
import os, csv
from typing import List, Dict, Any, Tuple, Optional

# =========================
# 기본 정규식/상수
# =========================
DATE_RE = re.compile(
    r"""(?ix)
    (?:\d{4}[-./년]\s*\d{1,2}[-./월]\s*\d{1,2}(?:\s*[일\)])?)   # 2025-07-08 / 2025.07.08 / 2025년 7월 8일
    |(?:\d{1,2}/\d{1,2}/\d{2,4})                               # 07/08/2025
    |(?:\d{1,2}월\s*\d{1,2}일)                                  # 7월 8일
    |(?:\d{2}\.\d{2}\.\d{2})                                   # 25.02.13
    """
)
TIME_RE = re.compile(
    r"""(?ix)
    (?:\d{1,2}[:시]\d{0,2}\s*(?:am|pm|AM|PM)?)
    |(?:오전\s*\d{1,2}시(?:\s*\d{1,2}분)?)
    |(?:오후\s*\d{1,2}시(?:\s*\d{1,2}분)?)
    |(?:낮\s*\d{1,2}시(?:\s*\d{1,2}분)?)
    |(?:밤\s*\d{1,2}시(?:\s*\d{1,2}분)?)
    |(?:저녁\s*\d{1,2}시(?:\s*\d{1,2}분)?)
    |(?:새벽\s*\d{1,2}시(?:\s*\d{1,2}분)?)
    """
)
PRICE_TOKEN_RE = re.compile(r"(?i)(?:\d{1,3}(?:[,\s]?\d{3})+|\d+)\s*(?:원|₩|won)?")
# 한글 금액 묶음(3만5천원 / 3만원 / 5천원 / 35000원 등)
KOR_PRICE_SEQ_RE = re.compile(r"^(?:(\d+)만)?(?:(\d+)천)?(?:(\d+))?(?:원)?$")

PRICE_TYPE_HINTS = {
    "예매": "B-PRICE_TYPE",
    "현매": "B-PRICE_TYPE",
    "현장": "B-PRICE_TYPE",
    "사전": "B-PRICE_TYPE",
    "온라인": "B-PRICE_TYPE",
    "Door": "B-PRICE_TYPE",
    "Only": "I-PRICE_TYPE",
}

WEEKDAYS = {"월", "화", "수", "목", "금", "토", "일"}

FIELD_MAP = {
    "TITLE": ["B-TITLE", "I-TITLE"],
    "DATE": ["B-DATE", "I-DATE"],
    "TIME": ["B-TIME", "I-TIME"],
    "PRICE": ["B-PRICE", "I-PRICE"],
    "PRICE_ONSITE": ["B-PRICE_ONSITE", "I-PRICE_ONSITE"],
    "PRICE_TYPE": ["B-PRICE_TYPE", "I-PRICE_TYPE"],
    "LINEUP": ["B-LINEUP", "I-LINEUP"],
    "INSTAGRAM": ["B-INSTAGRAM"],
    "VENUE": ["B-VENUE", "I-VENUE"],
    "V_INSTA": ["B-V_INSTA"],
    "TICKET_OPEN_DATE": ["B-TICKET_OPEN_DATE", "I-TICKET_OPEN_DATE"],
    "TICKET_OPEN_TIME": ["B-TICKET_OPEN_TIME", "I-TICKET_OPEN_TIME"],
}
LABEL2FIELD = {lab: field for field, labs in FIELD_MAP.items() for lab in labs}

# =========================
# 한국어 조사/구두점 보정
# =========================
_JOSA = ("은","는","이","가","을","를","과","와","도","으로","로","에서","의")
def base_ko(tok: str) -> str:
    t = tok.strip("()[]{}:;,+!?./\\&|'\"")
    for j in _JOSA:
        if t.endswith(j) and len(t) > len(j):
            t = t[: -len(j)]
            break
    return t

def _strip_punct_token(s: str) -> str:
    return s.strip("()[]{}:;,+!?./\\&|'\"")

# =========================
# Threshold 관련
# =========================
def _label_threshold(label: str, thresholds: Dict[str, float]) -> float:
    if label == "O":
        return 0.0
    ent = label.split("-", 1)[-1] if "-" in label else label
    return thresholds.get(label, thresholds.get(ent, thresholds.get("default", 0.5)))

def _apply_threshold(label: str, conf: float, thresholds: Dict[str, float]) -> str:
    if label == "O":
        return "O"
    th = _label_threshold(label, thresholds)
    return label if conf >= th else "O"

# =========================
# BIO/스팬 유틸
# =========================
def _is_handle(tok: str) -> bool:
    return tok.startswith("@") and len(tok) > 1 and re.match(r"^@[A-Za-z0-9_.]+$", tok) is not None

def _is_weekday_wrap(tok: str) -> bool:
    t = tok.strip()
    return len(t) >= 3 and t[0] == "(" and t[-1] == ")" and t[1:-1] in WEEKDAYS

def _bio_spans(tokens: List[str], labels: List[str]) -> List[Tuple[int, int, str]]:
    spans = []
    cur_start, cur_type = None, None
    for i, lab in enumerate(labels):
        if lab == "O":
            if cur_start is not None:
                spans.append((cur_start, i, cur_type))
                cur_start, cur_type = None, None
            continue
        prefix, t = lab.split("-", 1)
        if prefix == "B":
            if cur_start is not None:
                spans.append((cur_start, i, cur_type))
            cur_start, cur_type = i, t
        elif prefix == "I":
            if cur_type != t or cur_start is None:
                if cur_start is not None:
                    spans.append((cur_start, i, cur_type))
                cur_start, cur_type = i, t
        else:
            if cur_start is not None:
                spans.append((cur_start, i, cur_type))
                cur_start, cur_type = None, None
    if cur_start is not None:
        spans.append((cur_start, len(labels), cur_type))
    return spans

def _collect_fields(tokens: List[str], spans: List[Tuple[int, int, str]]) -> Dict[str, List[str]]:
    fields = {k: [] for k in FIELD_MAP.keys()}
    for s, e, t in spans:
        text = " ".join(tokens[s:e]).strip()
        if not text: 
            continue
        if t in fields:
            fields[t].append(text)
    return fields

# =========================
# Lexicon 로더
# =========================
def _normalize_cell(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s).strip(" '\"")
    return s

def _read_lexicon_csv(path: str) -> set:
    bag = set()
    if not path or not os.path.exists(path): 
        return bag
    with open(path, "r", encoding="utf-8") as f:
        try:
            f.seek(0)
            reader = csv.DictReader(f)
            rows = list(reader)
            if reader.fieldnames:
                for row in rows:
                    for _, v in row.items():
                        cell = _normalize_cell(v)
                        if not cell: continue
                        for token in re.split(r"[;,/|]", cell):
                            token = _normalize_cell(token)
                            if token: bag.add(token)
                return bag
        except Exception:
            pass
        f.seek(0)
        reader = csv.reader(f)
        for row in reader:
            for v in row:
                cell = _normalize_cell(v)
                if not cell: continue
                for token in re.split(r"[;,/|]", cell):
                    token = _normalize_cell(token)
                    if token: bag.add(token)
    return bag

def _make_keyword_regex(words: set):
    if not words: return None
    ws = sorted({w for w in words if w}, key=len, reverse=True)
    esc = [re.escape(w) for w in ws]
    boundary = r"[A-Za-z0-9_가-힣]"
    patt = rf"(?i)(?<!{boundary})(?:{'|'.join(esc)})(?!{boundary})"
    return re.compile(patt)

def load_lexicons(artists_csv: str | None = None, venues_csv: str | None = None) -> dict:
    artists = _read_lexicon_csv(artists_csv) if artists_csv else set()
    venues  = _read_lexicon_csv(venues_csv)  if venues_csv  else set()
    artists = {w for w in artists if len(w) >= 2}
    venues  = {w for w in venues  if len(w) >= 2}
    return {
        "artists": artists,
        "venues": venues,
        "artist_re": _make_keyword_regex(artists),
        "venue_re":  _make_keyword_regex(venues),
    }

# =========================
# VENUE 기본 규칙
# =========================
def _find_span_by_phrase(tokens, phrase):
    parts = [p for p in phrase.strip().split() if p]
    if not parts: return None
    n, m = len(tokens), len(parts)
    for i in range(n - m + 1):
        ok = True
        for j in range(m):
            if tokens[i + j] != parts[j]:
                ok = False; break
        if ok: return (i, i + m - 1)
    return None

def _tag_span(labels, start, end, ent):
    labels[start] = f"B-{ent}"
    for k in range(start + 1, end + 1):
        labels[k] = f"I-{ent}"

def rule_venue_basic(tokens, labels, confidences, venues_set=None):
    if venues_set is None: venues_set = set()
    idxs = [i for i, t in enumerate(tokens) if t in ("장소:", "장소", "공연장:", "공연장")]
    if not idxs: return labels, confidences

    STOP_TOKENS = {"•", "-", "—", "|", "/", "좌석형태:", "좌석:", "예매처:", "티켓오픈:", "예매:", "티켓:", "입장:", "공연일시:", "일시:", "시간:", "at"}

    for pos in idxs:
        start = pos + 1
        if start >= len(tokens): continue
        end = start
        while end < len(tokens):
            if tokens[end] in STOP_TOKENS: break
            if tokens[end] in (")", "]") or tokens[end].startswith("•"): break
            end += 1
        cand = tokens[start:end]
        if not cand: continue

        matched = False
        if venues_set:
            for v in sorted(venues_set, key=lambda s: len(s.split()), reverse=True):
                span = _find_span_by_phrase(tokens[start:end], v)
                if span:
                    s = start + span[0]; e = start + span[1]
                    _tag_span(labels, s, e, "VENUE"); matched = True; break
        if not matched:
            take = min(5, len(cand))
            if take > 0:
                s = start; e = start + take - 1
                _tag_span(labels, s, e, "VENUE")
    return labels, confidences

# =========================
# PRICE/PRICE_ONSITE 규칙
# =========================
def kor_price_value_from_seq(seq_tokens: List[str]) -> Optional[int]:
    """['3','만','5','천','원'] -> 35000, ['3','만원'] -> 30000, ['35000','원']->35000 등"""
    s = "".join(seq_tokens).replace(" ", "")
    # 이미 12,345원 형태
    m_fmt = re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)원", s)
    if m_fmt:
        num = int(s[:-1].replace(",", ""))
        return num if num > 0 else None
    m = KOR_PRICE_SEQ_RE.fullmatch(s)
    if not m: 
        return None
    man = int(m.group(1)) if m.group(1) else 0
    chun = int(m.group(2)) if m.group(2) else 0
    rest = int(m.group(3)) if m.group(3) else 0
    val = man*10000 + chun*1000 + rest
    return val if val > 0 else None

def normalize_price_string(s: str) -> str:
    s0 = s.strip()
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+원", s0): return s0
    if re.fullmatch(r"\d+원", s0): return f"{int(s0[:-1]):,}원"
    compact = s0.replace(" ", "")
    m = KOR_PRICE_SEQ_RE.fullmatch(compact)
    if m:
        man = int(m.group(1)) if m.group(1) else 0
        chun = int(m.group(2)) if m.group(2) else 0
        rest = int(m.group(3)) if m.group(3) else 0
        v = man*10000 + chun*1000 + rest
        if v>0: return f"{v:,}원"
    return s0

def rule_price_from_keywords(tokens, labels, window=6):
    """
    '예매' 주변 -> PRICE, '현장/현매' 주변 -> PRICE_ONSITE
    한국어 금액(3만5천원 등)과 숫자/콤마 금액 모두 처리.
    """
    n = len(tokens)
    for i, tok in enumerate(tokens):
        key = base_ko(tok)
        if key not in {"예매", "현장", "현매"}:
            continue
        # 오른쪽으로 금액 토큰 군 수집
        j = i + 1
        seq = []
        # 시작 숫자(혹은 이미 '12,000원') 체크
        if j < n and (re.fullmatch(r"\d+", tokens[j]) or re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)원", tokens[j])):
            seq.append(tokens[j])
            j += 1
        # 뒤에 '만','천','원','숫자' 연속 붙으면 계속 수집
        while j < n and j <= i+window and (tokens[j] in {"만","천","원"} or re.fullmatch(r"\d+", tokens[j]) or re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)원", tokens[j])):
            seq.append(tokens[j])
            j += 1
        if not seq:
            continue
        val = kor_price_value_from_seq(seq)
        if not val:
            continue
        # 스팬 시작/끝 계산: i 다음 토큰부터 seq 길이만큼
        s = i + 1
        e = s + (len(seq) - 1)
        if not (0 <= s < n and 0 <= e < n): 
            continue
        ent = "PRICE" if key == "예매" else "PRICE_ONSITE"
        _tag_span(labels, s, e, ent)
    return labels

def rule_price_type_link(tokens, labels):
    """
    모델이 PRICE로 잡아놨을 때 '현장/현매' 키워드 인접하면 PRICE_ONSITE로 바꿔줌.
    """
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if base_ko(tok) in {"현장","현매"}:
            for j in range(i+1, min(n, i+4)):
                if labels[j].startswith("B-PRICE"):
                    labels[j] = "B-PRICE_ONSITE"
                elif labels[j].startswith("I-PRICE"):
                    labels[j] = "I-PRICE_ONSITE"
    return labels

# =========================
# LINEUP: 힌트 + 사전
# =========================
LINEUP_HINTS = {"출연", "라인업", "with", "With", "WITH"}
CONJ = {"와", "과", "&", "/", ",", "그리고"}

def rule_lineup_hints(tokens, labels, lookahead=6):
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if base_ko(tok) in LINEUP_HINTS:
            j = i + 1
            taken = 0
            while j < n and taken < lookahead:
                if labels[j] == "O" and not _is_handle(tokens[j]) and not DATE_RE.fullmatch(tokens[j]) and not TIME_RE.fullmatch(tokens[j]):
                    if re.fullmatch(r"[(),/|&]", tokens[j]):
                        j += 1; continue
                    labels[j] = "B-LINEUP"
                    taken += 1
                j += 1
    return labels

def rule_lineup_from_lexicon(tokens, labels, lexicons=None):
    if not lexicons or "artists" not in lexicons:
        return labels
    art = set(lexicons["artists"]) | {a.lower() for a in lexicons["artists"]}
    n = len(tokens)
    i = 0
    while i < n:
        b = base_ko(tokens[i])
        if (b in art or b.lower() in art) and labels[i] == "O":
            labels[i] = "B-LINEUP"
            # 접속사로 이어진 다음 아티스트 붙이기
            j = i + 1
            if j < n and base_ko(tokens[j]) in CONJ:
                k = j + 1
                if k < n:
                    b2 = base_ko(tokens[k])
                    if (b2 in art or b2.lower() in art) and labels[k] == "O":
                        labels[k] = "I-LINEUP"
                        i = k + 1
                        continue
            i += 1
        else:
            i += 1
    return labels

# =========================
# TICKET OPEN: 양방향 키워드
# =========================
_OPEN_LEFT = {"티켓", "예매", "티켓오픈", "티켓 오픈", "예매오픈", "예매 오픈"}
_OPEN_RIGHT = {"오픈", "오픈:", "오픈합니다", "오픈해요", "open", "OPEN"}

def is_time_token(tok: str) -> bool:
    return bool(re.fullmatch(r"(?:(?:오전|오후|낮|밤|저녁|새벽)\s*\d{1,2}시(?:\s*\d{1,2}분)?|\d{1,2}:\d{2}|\d{1,2}시(?:\s*\d{1,2}분)?)", tok))

def is_date_token(tok: str) -> bool:
    return bool(
        re.fullmatch(r"\d{4}년", tok) or
        re.fullmatch(r"\d{1,2}월", tok) or
        re.fullmatch(r"\d{1,2}일", tok) or
        re.fullmatch(r"\([월화수목금토일]\)", tok) or
        DATE_RE.fullmatch(tok)
    )

def tag_seq(labels, s, e, typ):
    if s is None or e is None or e < s: return
    _tag_span(labels, s, e, typ)

def rule_ticket_open_plus(tokens, labels, window=14):
    n = len(tokens)

    # case A: '티켓오픈' 같은 키워드 직후 날짜/시간
    for i, tok in enumerate(tokens):
        if base_ko(tok) in {"티켓오픈"} or tok in {"티켓오픈", "티켓오픈:"}:
            j = i + 1
            s = e = None
            while j < n and j <= i+window and is_date_token(tokens[j]):
                s = j if s is None else s
                e = j; j += 1
            tag_seq(labels, s, e, "TICKET_OPEN_DATE")
            while j < n and j <= i+window and re.fullmatch(r"[,\-–~:/]|및|과|그리고", tokens[j]):
                j += 1
            if j < n and is_time_token(tokens[j]):
                labels[j] = "B-TICKET_OPEN_TIME"

    # case B: '티켓/예매 … 오픈' (조사/구두점 허용)
    for i, tok in enumerate(tokens):
        left = base_ko(tok)
        if left in _OPEN_LEFT:
            r = None
            for k in range(i+1, min(n, i+1+window)):
                if base_ko(tokens[k]) in _OPEN_RIGHT:
                    r = k; break
            if r is None: continue

            s = e = None
            for p in range(i+1, r):
                if is_date_token(tokens[p]):
                    s = p if s is None else s
                    e = p
            tag_seq(labels, s, e, "TICKET_OPEN_DATE")

            # 시간은 날짜 직후/중간 구두점 건너뛰고 찾기
            q = (e + 1) if e is not None else (i + 1)
            while q < r and re.fullmatch(r"[,\-–~:/]|및|과|그리고", tokens[q]):
                q += 1
            if q < r and is_time_token(tokens[q]):
                labels[q] = "B-TICKET_OPEN_TIME"
    return labels

# =========================
# 최종 가격 필드 정규화
# =========================
def normalize_price_fields(fields: Dict[str, list]):
    for k in ["PRICE","PRICE_ONSITE"]:
        if k in fields:
            fields[k] = [normalize_price_string(v) for v in fields[k]]
    return fields

# =========================
# 메인: 모델 + 규칙 병합
# =========================
def merge_model_and_rules(
    tokens: List[str],
    model_labels: List[str],
    model_confs: Optional[List[float]],
    thresholds: Dict[str, float],
    lexicons: Optional[Dict[str, set]] = None,
) -> Dict[str, Any]:
    if model_confs is None:
        model_confs = [1.0] * len(model_labels)

    n = min(len(tokens), len(model_labels), len(model_confs))
    tokens = tokens[:n]; model_labels = model_labels[:n]; model_confs = model_confs[:n]

    lexicons = lexicons or {}
    artist_lex = lexicons.get("artists", set())
    venue_lex = lexicons.get("venues", set())

    # 1) threshold + 기본 보강
    labels = []
    for tok, y, c in zip(tokens, model_labels, model_confs):
        y2 = _apply_threshold(y, c, thresholds)
        if y2 == "O":
            if _is_weekday_wrap(tok): y2 = "I-DATE"
            elif DATE_RE.fullmatch(tok): y2 = "B-DATE"
            elif TIME_RE.fullmatch(tok): y2 = "B-TIME"
            elif PRICE_TOKEN_RE.fullmatch(tok): y2 = "B-PRICE"
        if _is_handle(tok):
            prev = labels[-1] if labels else "O"
            y2 = "B-V_INSTA" if prev.startswith(("B-VENUE","I-VENUE")) else "B-INSTAGRAM"
        if y2 == "O":
            hint = PRICE_TYPE_HINTS.get(tok, None)
            if hint: y2 = hint
        low = tok.lower()
        if y2 == "O":
            if tok in artist_lex or low in artist_lex: y2 = "B-LINEUP"
            elif tok in venue_lex or low in venue_lex: y2 = "B-VENUE"
        labels.append(y2)

    # 2) BIO 정합성
    fixed = []
    prev_type = None
    for lab in labels:
        if lab == "O":
            fixed.append("O"); prev_type=None; continue
        try:
            pref, t = lab.split("-", 1)
        except ValueError:
            fixed.append("O"); prev_type=None; continue
        if pref == "I" and prev_type != t:
            fixed.append(f"B-{t}"); prev_type=t
        else:
            fixed.append(lab); prev_type=t
    labels = fixed

    # 2.5) 규칙 적용 (순서 중요)
    labels = rule_price_type_link(tokens, labels)           # 현장/현매 → PRICE_ONSITE
    labels = rule_lineup_hints(tokens, labels)              # '출연/with/라인업' 힌트
    labels = rule_lineup_from_lexicon(tokens, labels, lexicons=lexicons)  # 사전
    labels = rule_ticket_open_plus(tokens, labels)          # 티켓오픈(양방향)
    labels = rule_price_from_keywords(tokens, labels)       # 예매/현장 주변 금액 태깅

    # 3) VENUE 보강(키워드/사전)
    venues_set = set(lexicons.get("venues", [])) if lexicons else set()
    labels, model_confs = rule_venue_basic(tokens, labels, model_confs, venues_set=venues_set)

    # 4) 스팬/필드
    spans = _bio_spans(tokens, labels)
    entities = []
    for s, e, t in spans:
        text = " ".join(tokens[s:e]).strip()
        if text: entities.append({"type": t, "start": s, "end": e, "text": text})
    fields = _collect_fields(tokens, spans)

    # 5) 필드 정리 + 가격 정규화
    for k in fields:
        uniq=[]; seen=set()
        for v in fields[k]:
            v2 = re.sub(r"\s+", " ", v).strip()
            v2 = re.sub(r"\s*\(\s*$", "", v2)  # '... (' 꼬리 제거
            if v2 and v2 not in seen:
                seen.add(v2); uniq.append(v2)
        fields[k] = uniq
    fields = normalize_price_fields(fields)

    return {
        "tokens": tokens,
        "labels": labels,
        "confs": model_confs,
        "entities": entities,
        "fields": fields,
    }

# =========================
# 스키마 가드
# =========================
def schema_guard(doc: dict, max_per_field: int = 20) -> dict:
    doc = dict(doc or {})
    tokens = list(doc.get("tokens", []))
    labels = list(doc.get("labels", []))
    confs  = list(doc.get("confs",  []))
    ents   = list(doc.get("entities", []))
    fields = dict(doc.get("fields", {}))

    n = min(len(tokens), len(labels), len(confs)) if labels and confs else len(tokens)
    tokens = tokens[:n]; labels = labels[:n] if labels else ["O"]*n; confs = confs[:n] if confs else [1.0]*n

    valid_types = {t for t in FIELD_MAP.keys()}
    def _norm(lab: str) -> str:
        if lab == "O": return "O"
        if isinstance(lab, str) and "-" in lab:
            p, t = lab.split("-", 1)
            if p in {"B","I"} and t in valid_types: return f"{p}-{t}"
        return "O"
    labels = [_norm(l) for l in labels]

    clean_ents=[]
    for e in ents:
        try:
            t = e.get("type"); s = int(e.get("start")); eidx = int(e.get("end"))
            if t in valid_types and 0 <= s < eidx <= n:
                text = " ".join(tokens[s:eidx]).strip()
                if text: clean_ents.append({"type": t, "start": s, "end": eidx, "text": text})
        except Exception:
            continue
    ents = clean_ents

    for k in FIELD_MAP.keys():
        vals = [str(v).strip() for v in fields.get(k, []) if str(v).strip()]
        seen, uniq = set(), []
        for v in vals:
            v2 = re.sub(r"\s+", " ", v)
            v2 = re.sub(r"\s*\(\s*$", "", v2)
            if v2 not in seen:
                seen.add(v2); uniq.append(v2)
        fields[k] = uniq[:max_per_field]

    def _is_handle_text(s: str) -> bool:
        return s.startswith("@") and re.fullmatch(r"@[A-Za-z0-9_.]+", s) is not None

    mis = [v for v in fields.get("LINEUP", []) if _is_handle_text(v)]
    if mis:
        fields["LINEUP"] = [v for v in fields["LINEUP"] if not _is_handle_text(v)]
        fields["INSTAGRAM"] = (fields.get("INSTAGRAM", []) + mis)[:max_per_field]

    mis = [v for v in fields.get("VENUE", []) if _is_handle_text(v)]
    if mis:
        fields["VENUE"] = [v for v in fields["VENUE"] if not _is_handle_text(v)]
        fields["V_INSTA"] = (fields.get("V_INSTA", []) + mis)[:max_per_field]

    doc["tokens"] = tokens
    doc["labels"] = labels
    doc["confs"]  = confs
    doc["entities"] = ents
    doc["fields"] = fields
    return doc

# 내보내기
try:
    __all__.extend([
        "merge_model_and_rules","schema_guard","load_lexicons",
        "rule_ticket_open_plus","rule_lineup_from_lexicon","rule_price_from_keywords"
    ])
except NameError:
    __all__ = [
        "merge_model_and_rules","schema_guard","load_lexicons",
        "rule_ticket_open_plus","rule_lineup_from_lexicon","rule_price_from_keywords"
    ]
