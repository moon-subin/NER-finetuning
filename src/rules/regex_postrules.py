# src/rules/regex_postrules.py
# -*- coding: utf-8 -*-
"""
regex_postrules.py — 모델 예측 결과를 규칙으로 보정하는 후처리 모듈(완성본)

공개 API:
- merge_model_and_rules(tokens, model_labels, confidences=None, thresholds=None, lexicons=None) -> Dict
- schema_guard(obj) -> Dict
- load_lexicons(artists_csv=None, venues_csv=None) -> Dict[str, Set[str]]

내부(규칙 본체):
- apply_regex_postrules(text, tokens, fields, lexicons=None) -> Dict (필드 보정)
"""

from __future__ import annotations
import re
import csv
from typing import Dict, List, Tuple, Optional

# =========================
# 정규표현식 패턴
# =========================

RE_YEAR = r'(?P<year>\d{4})년'
RE_MD = r'(?P<month>\d{1,2})\s*월\s*(?P<day>\d{1,2})\s*일'
RE_DOW = r'(?:\s*\((?P<dow>[월화수목금토일]|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\))?'

RE_FULLDATE_OPT_YEAR = re.compile(rf'(?:{RE_YEAR}\s*)?{RE_MD}\s*{RE_DOW}')
RE_FULLDATE_MUST_YEAR = re.compile(rf'{RE_YEAR}\s*{RE_MD}\s*{RE_DOW}')

# 시간(패치): 맨숫자 금지. am/pm 있으면 '시' 없어도, 없으면 '시' 필수.
RE_TIME = re.compile(
    r'((?P<ampm>오전|오후|낮|밤|저녁)\s*(?P<hour1>\d{1,2})(?:\s*:\s*(?P<min1>\d{2}))?\s*시?)'
    r'|'
    r'((?P<hour2>\d{1,2})(?:\s*:\s*(?P<min2>\d{2}))\s*시?)'
    r'|'
    r'((?P<hour3>\d{1,2})\s*시)'
)

# 숫자+원
RE_PRICE = re.compile(r'(?P<amount>\d{1,3}(?:,\d{3})+|\d{4,6})\s*원?')

# 한국식 금액(패치)
RE_KR_PRICE = re.compile(
    r'(?:(?P<man>\d+)\s*만)?\s*(?:(?P<chon>\d+)\s*천)?\s*(?P<num>\d{1,3}(?:,\d{3})+|\d+)?\s*원?'
)

# '티켓 오픈' 또는 '티켓은 … 오픈'(패치)
RE_TICKET_OPEN = re.compile(r'티\s*켓\s*(?:은|:)?[^\n]{0,20}?오\s*픈', re.IGNORECASE)

# 공연장 키워드
VENUE_KEYWORDS = r'(홀|클럽|라이브|센터|스테이지|스튜디오|씨어터|극장|하우스|라운지|스퀘어|플랫폼|플레이스|페스티벌|레코드|창고|공간|바|펍|플라자|파크|웨이브|베뉴)'

# LINEUP 불용 토큰
LINEUP_STOPWORDS = {'티켓','오픈','출연','공연','콘서트','쇼케이스','단독','공지','예매','현매','현장','가격'}

TITLE_STOP = {'달','공지','안내','정보','발표'}

POSTFIX_JOSA = re.compile(r'(와|과|이|가|는|은|도|를|을|과의|와의)$')


# =========================
# 유틸
# =========================

def _strip_josa(name: str) -> str:
    return POSTFIX_JOSA.sub('', _strip_space(name))

def _find_dates_with_span(text: str):
    return [( _join_full_date(m), m.start(), m.end() ) for m in RE_FULLDATE_OPT_YEAR.finditer(text)]

def _find_times_with_span(text: str):
    out = []
    for m in RE_TIME.finditer(text):
        out.append( (_normalize_time(m.group(0)), m.start(), m.end()) )
    return out


def _tidy_title(fields: Dict) -> None:
    xs = []
    for t in fields.get('TITLE', []) or []:
        t = _strip_space(t)
        if not t: continue
        if t in TITLE_STOP: continue
        if len(t) <= 1: continue
        xs.append(t)
    fields['TITLE'] = _dedupe(xs)

def _price_to_int(p: str) -> int:
    try:
        return int(p.replace(',', '').replace('원','').strip())
    except:
        return -1

def _dedupe(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x is None: continue
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _strip_space(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def _join_full_date(m: re.Match) -> str:
    year = m.group('year')
    month = m.group('month')
    day = m.group('day')
    dow = m.group('dow') or ''
    core = f"{int(month)}월 {int(day)}일"
    if dow:
        core += f" ({dow})"
    return (f"{year}년 " if year else "") + core

def _normalize_time(s: str) -> str:
    m = RE_TIME.search(s)
    if not m:
        return _strip_space(s)
    ampm = (m.group('ampm') or '').strip()
    hour = m.group('hour1') or m.group('hour2') or m.group('hour3')
    minute = m.group('min1') or m.group('min2')
    t = f"{hour}:{minute}" if minute else f"{hour}시"
    return _strip_space(f"{ampm} {t}".strip())

def _normalize_price(s: str) -> str:
    m = RE_PRICE.search(s)
    if not m: return _strip_space(s)
    amount = m.group('amount')
    if ',' not in amount and len(amount) > 3:
        amount = f"{int(amount):,}"
    return f"{amount}원"

def _parse_kr_price(s: str) -> Optional[str]:
    raw = _strip_space(s)
    m = RE_KR_PRICE.fullmatch(raw)
    if not m:
        return None
    man = m.group('man')
    chon = m.group('chon')
    num = m.group('num')

    # '3' 단독 같은 경우 배제(단위/원 없고 4자리 미만)
    has_unit = ('만' in raw) or ('천' in raw) or ('원' in raw)
    long_num = bool(num and len(num.replace(',', '')) >= 4)
    if not (man or chon or has_unit or long_num):
        return None

    total = 0
    if man:  total += int(man) * 10_000
    if chon: total += int(chon) * 1_000
    if num:  total += int(num.replace(',', ''))
    if total <= 0:
        return None
    return f"{total:,}원"

def _window_after(text: str, start_idx: int, max_chars: int = 80) -> str:
    return text[start_idx:start_idx + max_chars]

def _find_all_full_dates(text: str) -> List[str]:
    out = []
    for m in RE_FULLDATE_OPT_YEAR.finditer(text):
        out.append(_join_full_date(m))
    return _dedupe(out)

def _find_all_times(text: str) -> List[str]:
    out = []
    for m in RE_TIME.finditer(text):
        out.append(_normalize_time(m.group(0)))
    return _dedupe(out)

def _merge_year_with_partial_dates(tokens: List[str], existing_dates: List[str]) -> List[str]:
    merged = list(existing_dates)
    for i, tk in enumerate(tokens):
        if re.fullmatch(r'\d{4}년', tk):
            window = ' '.join(tokens[i+1:i+5])
            m = RE_FULLDATE_OPT_YEAR.search(f"{tk} {window}")
            if m:
                merged.append(_join_full_date(m))
    return _dedupe([_strip_space(x) for x in merged])

def _is_probable_venue_word(word: str) -> bool:
    return bool(re.search(VENUE_KEYWORDS, word))

def _clean_venue_token(v: str) -> str:
    v = re.sub(r'(에서|에|으로|의)\s*$', '', v)
    v = re.sub(r'[,\.\!]+$', '', v)
    return _strip_space(v)

def _combine_area_and_venue(text: str, venues: List[str]) -> List[str]:
    """
    '홍대 롤링홀에서' 뿐 아니라
    '클럽 FF에서'처럼 '접두 키워드(클럽/홀/극장...) + 고유명'도 결합 (패치)
    """
    candidates = list(venues)
    for m in re.finditer(r'([가-힣A-Za-z·]+)\s+([가-힣A-Za-z·0-9]+?)\s*(?:에서|에)\b', text):
        area, place = m.group(1), m.group(2)
        if _is_probable_venue_word(place) or _is_probable_venue_word(area):
            candidates.append(_strip_space(f"{area} {place}"))
    return _dedupe([_clean_venue_token(v) for v in candidates])

# =========================
# 규칙 본체
# =========================

def _rescan_and_fix_prices(text: str, tokens: List[str], fields: Dict) -> None:
    price = list(fields.get('PRICE', []) or [])
    onsite = list(fields.get('PRICE_ONSITE', []) or [])

    def tok_span_to_str(i, j):
        return _strip_space(''.join(tokens[i:j]))  # "3 만 5 천 원" -> "3만5천원"

    def pick_best_amount(slices: List[str]) -> Optional[str]:
        best = None; best_val = -1
        for cand in slices:
            v = _parse_kr_price(cand)
            if not v and RE_PRICE.fullmatch(_strip_space(cand or '')):
                v = _normalize_price(cand)
            if v:
                val = _price_to_int(v)
                if val > best_val:
                    best, best_val = v, val
        return best

    # A) 토큰 기반: '예매' / '현매|현장' 뒤 5토큰 내에서 "가장 큰" 금액 선택
    for i, tk in enumerate(tokens):
        if tk == '예매':
            candidates = [tok_span_to_str(i+1, j+1) for j in range(i+1, min(i+6, len(tokens)))]
            best = pick_best_amount(candidates)
            if best: price.append(best)

        if tk in ('현매', '현장'):
            candidates = [tok_span_to_str(i+1, j+1) for j in range(i+1, min(i+6, len(tokens)))]
            best = pick_best_amount(candidates)
            if best: onsite.append(best)

    # B) 텍스트 앵커 보조
    for m in re.finditer(r'(예매)', text):
        win = _window_after(text, m.end(), 60)
        words = re.findall(r'[0-9가-힣,]+원?|[0-9가-힣,]+', win)
        best = pick_best_amount(words)
        if best: price.append(best)
    for m in re.finditer(r'(현매|현장)', text):
        win = _window_after(text, m.end(), 60)
        words = re.findall(r'[0-9가-힣,]+원?|[0-9가-힣,]+', win)
        best = pick_best_amount(words)
        if best: onsite.append(best)

    # C) 불량 값 정리: 숫자 단독/짧은 숫자 제거
    def clean_prices(xs):
        out = []
        for x in xs:
            x = _strip_space(x)
            if not x: continue
            if not x.endswith('원'):
                # 숫자만 들어온 경우 버림
                if re.fullmatch(r'\d+', x):
                    continue
                # 혹시 '3만' 같은 건 정규화 시도
                v = _parse_kr_price(x)
                if v: x = v
                else: continue
            out.append(x)
        return _dedupe(out)

    fields['PRICE'] = clean_prices(price)
    fields['PRICE_ONSITE'] = clean_prices(onsite)

def _ticket_open_mapping(text: str, fields: Dict) -> None:
    """
    '티켓은 … 오픈' 포함 다양한 변형에서
    '오픈' 앵커에 가장 가까운 날짜/시간을 선택.
    너무 멀리 떨어진(> 80자) 후보는 버림.
    """
    opens = list(RE_TICKET_OPEN.finditer(text))
    if not opens: 
        fields['TICKET_OPEN_DATE'] = _dedupe(fields.get('TICKET_OPEN_DATE', []))
        fields['TICKET_OPEN_TIME'] = _dedupe(fields.get('TICKET_OPEN_TIME', []))
        return

    all_dates = _find_dates_with_span(text)  # [(txt, s, e)]
    all_times = _find_times_with_span(text)

    to_dates, to_times = [], []
    for op in opens:
        anchor = (op.start() + op.end()) // 2

        # 날짜: 앵커와의 거리가 가장 가까운 것 1개
        best_d, best = 10**9, None
        for dtxt, ds, de in all_dates:
            dist = min(abs(anchor - ds), abs(anchor - de))
            if dist < best_d and dist <= 80:  # 80자 이내만 허용
                best_d, best = dist, dtxt
        if best: 
            to_dates.append(best)

        # 시간: 앵커와의 거리가 가장 가까운 것 1개
        best_d, best = 10**9, None
        for ttxt, ts, te in all_times:
            dist = min(abs(anchor - ts), abs(anchor - te))
            if dist < best_d and dist <= 80:
                best_d, best = dist, ttxt
        if best:
            to_times.append(best)

    fields['TICKET_OPEN_DATE'] = _dedupe(to_dates)
    fields['TICKET_OPEN_TIME'] = _dedupe(to_times)


def _filter_lineup(fields: Dict, lexicons: Optional[Dict]=None, text: Optional[str]=None) -> None:
    lineup = fields.get('LINEUP', []) or []
    out = []
    # 1) 모델이 준 후보 정리
    for x in lineup:
        s = _strip_josa(x)
        if not s or len(s) <= 1: continue
        if s in LINEUP_STOPWORDS: continue
        if s in ('!',',','/','(',')','·','-','—'): continue
        if RE_PRICE.search(s): continue
        if RE_FULLDATE_OPT_YEAR.search(s): continue
        if s.startswith('@') or re.fullmatch(r'[A-Za-z가-힣·\s]+', s):
            out.append(s)

    # 2) 아티스트 사전으로 텍스트에서 보강
    if lexicons and text:
        for name in lexicons.get('artists', []):
            # 공백/대소문자 무시 매칭
            pat = re.compile(re.escape(name), re.IGNORECASE)
            if pat.search(text):
                out.append(name)

    fields['LINEUP'] = _dedupe(out)


def _fix_instagram(fields: Dict) -> None:
    inst = list(fields.get('INSTAGRAM', []) or [])
    for x in fields.get('LINEUP', []):
        if x.startswith('@'): inst.append(x)
    fields['INSTAGRAM'] = _dedupe(inst)

def _prefer_longer_unique(items: List[str]) -> List[str]:
    items = _dedupe([_strip_space(x) for x in items if x])
    keep = []
    for i, a in enumerate(items):
        if any((a != b) and (a in b) for b in items):
            # a가 b에 포함되면(부분 문자열) 버림
            continue
        keep.append(a)
    return keep

def _fix_venues(text: str, fields: Dict, lexicons: Optional[Dict]=None) -> None:
    venues = [_clean_venue_token(v) for v in (fields.get('VENUE', []) or [])]
    venues = _combine_area_and_venue(text, venues)
    for m in re.finditer(r'([가-힣A-Za-z·0-9]+)(?:에서|에)\b', text):
        w = _clean_venue_token(m.group(1))
        if _is_probable_venue_word(w):
            venues.append(w)
    fields['VENUE'] = _dedupe([_strip_space(v) for v in venues if v])
    cand = [_strip_space(v) for v in venues if v]
    fields['VENUE'] = _prefer_longer_unique(cand)

def _normalize_dates(tokens: List[str], text: str, fields: Dict) -> None:
    dates = list(fields.get('DATE', []) or [])
    dates = _merge_year_with_partial_dates(tokens, dates)
    dates.extend(_find_all_full_dates(text))
    fields['DATE'] = _dedupe([_strip_space(d) for d in dates])

def _normalize_times(text: str, fields: Dict) -> None:
    times = list(fields.get('TIME', []) or [])
    times.extend(_find_all_times(text))
    fields['TIME'] = _dedupe([_normalize_time(t) for t in times])

def _final_tidy(fields: Dict) -> None:
    if 'PRICE_TYPE' in fields and fields['PRICE_TYPE']:
        uniq, seen = [], set()
        for x in fields['PRICE_TYPE']:
            s = '현매' if x in ('현매','현장') else x
            if s not in seen: seen.add(s); uniq.append(s)
        fields['PRICE_TYPE'] = uniq
    for k, v in list(fields.items()):
        if isinstance(v, list):
            fields[k] = [_strip_space(x) for x in v if _strip_space(str(x))]
        elif isinstance(v, str):
            fields[k] = _strip_space(v)

# 날짜 정합성 체크
def _is_valid_full_or_partial_date(s: str) -> bool:
    return bool(RE_FULLDATE_OPT_YEAR.fullmatch(s))

def _sanitize_ticket_open(fields: Dict) -> None:
    tod = []
    for d in fields.get('TICKET_OPEN_DATE', []) or []:
        d = _strip_space(d)
        if _is_valid_full_or_partial_date(d):
            tod.append(d)
    fields['TICKET_OPEN_DATE'] = _dedupe(tod)

def apply_regex_postrules(
    text: str,
    tokens: List[str],
    fields: Dict[str, List[str]],
    lexicons: Optional[Dict]=None
) -> Dict[str, List[str]]:
    """
    규칙 기반 후처리의 본체
    """
    _normalize_dates(tokens, text, fields)
    _normalize_times(text, fields)
    _ticket_open_mapping(text, fields)
    _rescan_and_fix_prices(text, tokens, fields)
    _filter_lineup(fields, lexicons=lexicons, text=text)
    _fix_instagram(fields)
    _fix_venues(text, fields, lexicons=lexicons)
    _sanitize_ticket_open(fields)
    _tidy_title(fields)
    _final_tidy(fields)
    return fields

# =========================
# BIO/스키마/렉시콘 유틸 (apply_rules.py에서 import)
# =========================

def _bio_to_spans(tokens: List[str], labels: List[str]) -> List[Tuple[str,int,int]]:
    spans = []
    cur_type, s = None, None
    for i, lab in enumerate(labels):
        if not lab or lab == 'O':
            if cur_type is not None:
                spans.append((cur_type, s, i))
                cur_type, s = None, None
            continue
        if '-' in lab:
            tag, etype = lab.split('-', 1)
        else:
            tag, etype = lab, None
        if tag == 'B':
            if cur_type is not None: spans.append((cur_type, s, i))
            cur_type, s = etype, i
        elif tag == 'I':
            if cur_type != etype:
                if cur_type is not None: spans.append((cur_type, s, i))
                cur_type, s = etype, i
        else:
            if cur_type is not None:
                spans.append((cur_type, s, i))
            cur_type, s = None, None
    if cur_type is not None:
        spans.append((cur_type, s, len(labels)))
    return spans

def _apply_thresholds(labels: List[str], confidences: Optional[List[float]], thresholds) -> List[str]:
    if not confidences: return labels
    out = []
    for lab, conf in zip(labels, confidences):
        if not lab or lab == 'O':
            out.append('O'); continue
        etype = lab.split('-', 1)[-1] if '-' in lab else lab
        thr = thresholds.get(etype, 0.0) if isinstance(thresholds, dict) else 0.0
        keep = (conf is None) or (float(conf) >= float(thr))
        out.append(lab if keep else 'O')
    return out

def _fields_from_bio(tokens: List[str], labels: List[str]) -> Dict[str, List[str]]:
    fields = {
        "TITLE": [], "DATE": [], "TIME": [],
        "PRICE": [], "PRICE_ONSITE": [], "PRICE_TYPE": [],
        "LINEUP": [], "INSTAGRAM": [],
        "VENUE": [], "V_INSTA": [],
        "TICKET_OPEN_DATE": [], "TICKET_OPEN_TIME": []
    }
    spans = _bio_to_spans(tokens, labels)
    def tok_join(a,b): return ' '.join(tokens[a:b]).strip()
    for etype, s, e in spans:
        text = tok_join(s,e)
        if not text: continue
        if   etype == 'PRICE':              fields['PRICE'].append(text)
        elif etype == 'PRICE_ONSITE':       fields['PRICE_ONSITE'].append(text)
        elif etype == 'PRICE_TYPE':         fields['PRICE_TYPE'].append(text)
        elif etype == 'LINEUP':             fields['LINEUP'].append(text)
        elif etype == 'INSTAGRAM':          fields['INSTAGRAM'].append(text)
        elif etype == 'VENUE':              fields['VENUE'].append(text)
        elif etype == 'V_INSTA':            fields['V_INSTA'].append(text)
        elif etype == 'TICKET_OPEN_DATE':   fields['TICKET_OPEN_DATE'].append(text)
        elif etype == 'TICKET_OPEN_TIME':   fields['TICKET_OPEN_TIME'].append(text)
        elif etype == 'DATE':               fields['DATE'].append(text)
        elif etype == 'TIME':               fields['TIME'].append(text)
        elif etype == 'TITLE':              fields['TITLE'].append(text)
    return fields

def load_lexicons(artists_csv: Optional[str]=None, venues_csv: Optional[str]=None) -> Dict[str, set]:
    """CSV에서 아티스트/공연장 사전 로드. 한 열에 하나씩 적혀 있다고 가정."""
    def load_csv(path: Optional[str]) -> set:
        bag = set()
        if not path: return bag
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for row in csv.reader(f):
                    for col in row:
                        col = (col or '').strip()
                        if col: bag.add(col)
        except FileNotFoundError:
            pass
        return bag
    return {"artists": load_csv(artists_csv), "venues": load_csv(venues_csv)}

def schema_guard(obj: Dict) -> Dict:
    """필수 키 보장 + 타입 정규화"""
    keys = ["TITLE","DATE","TIME","PRICE","PRICE_ONSITE","PRICE_TYPE",
            "LINEUP","INSTAGRAM","VENUE","V_INSTA",
            "TICKET_OPEN_DATE","TICKET_OPEN_TIME"]
    out = {"tokens": obj.get("tokens", []), "text": obj.get("text", "")}
    for k in keys:
        v = obj.get(k, [])
        if isinstance(v, list): out[k] = v
        elif v is None: out[k] = []
        else: out[k] = [str(v)]
    return out

def merge_model_and_rules(
    tokens: List[str],
    model_labels: List[str],
    confidences: Optional[List[float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    lexicons: Optional[Dict] = None
) -> Dict:
    """
    apply_rules.py에서 사용하는 진입점.
    1) 모델 BIO에 임계치 적용
    2) BIO → fields
    3) 텍스트 조립 후 regex 기반 후처리
    4) 스키마 정리
    """
    thresholds = thresholds or {}
    labels = _apply_thresholds(model_labels, confidences, thresholds)
    fields = _fields_from_bio(tokens, labels)
    text = ' '.join(tokens)
    final = apply_regex_postrules(text, tokens, fields, lexicons=lexicons)
    final["tokens"] = tokens
    final["text"] = text
    return schema_guard(final)


# =========================
# (선택) 셀프 테스트
# =========================
if __name__ == "__main__":
    text = "2025년 9월 21일(일) 오후 6시, 홍대 롤링홀에서 김오키, CADEJO(@cadejo___) 출연! 티켓은 2025년 8월 25일(월) 낮 12시 오픈, 예매 3만원 / 현장 3만5천원!"
    tokens = ['2025년','9월','21일','(일)','오후 6시',',','홍대','롤링홀에서','김오키',',','CADEJO','(','@cadejo___',')','출연','!','티켓은','2025년','8월','25일','(월)','낮 12시','오픈',',','예매','3','만원','/','현장','3','만','5','천원','!']
    labels = ['B-DATE','I-DATE','I-DATE','O','B-TIME','O','B-VENUE','O','B-LINEUP','O','B-LINEUP','O','B-INSTAGRAM','O','O','O','O','B-DATE','I-DATE','I-DATE','O','B-TIME','O','O','O','B-PRICE','O','B-PRICE_TYPE','B-PRICE','O','B-PRICE','O']
    merged = merge_model_and_rules(tokens, labels)
    from pprint import pprint
    pprint(merged)
