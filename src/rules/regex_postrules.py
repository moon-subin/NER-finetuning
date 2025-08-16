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
import unicodedata
from typing import Dict, List, Tuple, Optional

# =========================
# 정규표현식 패턴
# =========================

RE_YEAR = r'(?P<year>\d{4})년'
RE_MD = r'(?P<month>\d{1,2})\s*월\s*(?P<day>\d{1,2})\s*일'
RE_DOW = r'(?:\s*\((?P<dow>[월화수목금토일]|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\))?'

RE_FULLDATE_OPT_YEAR = re.compile(rf'(?:{RE_YEAR}\s*)?{RE_MD}\s*{RE_DOW}')
RE_FULLDATE_MUST_YEAR = re.compile(rf'{RE_YEAR}\s*{RE_MD}\s*{RE_DOW}')

# 시간(한글/숫자)
RE_TIME = re.compile(
    r'((?P<ampm>오전|오후|낮|밤|저녁)\s*(?P<hour1>\d{1,2})(?:\s*:\s*(?P<min1>\d{2}))?\s*시?)'
    r'|'
    r'((?P<hour2>\d{1,2})(?:\s*:\s*(?P<min2>\d{2}))\s*시?)'
    r'|'
    r'((?P<hour3>\d{1,2})\s*시)'
)

# 영어 AM/PM 시각/범위
RE_EN_TIME = re.compile(r'\b(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>AM|PM|am|pm)\b')
RE_EN_TIME_RANGE = re.compile(r'\b(?P<h1>\d{1,2})(?::(?P<m1>\d{2}))?-(?P<h2>\d{1,2})(?::(?P<m2>\d{2}))?\s*(?P<ap>AM|PM|am|pm)\b')

# 08.02 / 08/02 / 08-02 (+ optional year, DOW)
RE_MD_DOTTED = re.compile(
    r'\b(?P<m>\d{1,2})[.\-/](?P<d>\d{1,2})(?:[.\-/](?P<y>\d{2,4}))?(?:\s*\.\s*)?(?:\s*\((?P<dow>[A-Za-z]{3})\))?\b'
)

# 숫자+통화 (원/₩/￦/KRW, 통화 기호가 앞뒤 어디에 있어도 허용)
RE_PRICE = re.compile(
    r'(?:(?:₩|￦)\s*)?(?P<amount>\d{1,3}(?:,\d{3})+|\d{4,6}|\d+)\s*(?:원|₩|￦|KRW)?',
    re.IGNORECASE
)

# 한국식 금액(3만5천 등) + 통화 단위 확장
RE_KR_PRICE = re.compile(
    r'(?:(?P<man>\d+)\s*만)?\s*(?:(?P<chon>\d+)\s*천)?\s*(?P<num>\d{1,3}(?:,\d{3})+|\d+)?\s*(?:원|₩|￦|KRW)?',
    re.IGNORECASE
)

# '티켓 오픈' 또는 '티켓은 … 오픈'
RE_TICKET_OPEN = re.compile(r'티\s*켓\s*(?:은|:)?[^\n]{0,20}?오\s*픈', re.IGNORECASE)

# 공연장 키워드(+살롱)
VENUE_KEYWORDS = r'(홀|클럽|라이브|센터|스테이지|스튜디오|씨어터|극장|하우스|라운지|스퀘어|플랫폼|플레이스|페스티벌|레코드|창고|공간|바|펍|플라자|파크|웨이브|베뉴|살롱)'

# VENUE 오탐 방지 블랙리스트
VENUE_BLACKLIST_WORDS = {'도어', '티케팅', '티켓팅', '입장', '현매', '예약', '예매'}

# LINEUP 불용 토큰
LINEUP_STOPWORDS = {'티켓','오픈','출연','공연','콘서트','쇼케이스','단독','공지','예매','현매','현장','가격'}
TITLE_STOP = {'달','공지','안내','정보','발표'}
POSTFIX_JOSA = re.compile(r'(와|과|이|가|는|은|도|를|을|과의|와의)$')

# 라인업/섹션 탐지
LINEUP_SEC_HEAD = re.compile(
    r'^\s*(?:\[?\s*(?:LINE\s*[- ]?\s*UP|라인업|ARTIST|Artist(?:\s*information)?|Artist\s*Info)\s*\]?|\-+\s*LINE\s*UP\s*\-+)\s*$',
    re.IGNORECASE | re.MULTILINE
)
BULLET_LINE = re.compile(r'^\s*[✅▪️•●◦\-–—▶]\s*(.+)$', re.MULTILINE)
NAME_CAND = re.compile(
    r'^\s*(?P<name>(?:DJ\s+)?[A-Za-z][A-Za-z0-9 .\'\-\&\?\/!·:_]{1,60}|[가-힣0-9·]+(?:\s[가-힣0-9·]+){0,6})'
    r'(?:\s*\([^)]+\))?\s*(?:@[A-Za-z0-9._]+)?\s*$'
)
SECTION_BREAK = re.compile(
    r'^\s*(?:\[.*?\]|공연\s*정보|일시|장소|티켓|입장|문의|PRICE|TIME|DATE)\s*[:|]|^[-=]{3,}\s*$',
    re.MULTILINE
)
LINEUP_BLACKLIST_SUBSTR = {
    '오픈', '티켓', '문의', '입장', '가격', '현매', '예매', '공지', '안내',
    '라이브홀', '클럽', '홀', '극장', '페스티벌', '월드뮤직', '록 페스티벌',
    '노들섬', '스트레인지', '프룻', '장소', '일시'
}

# 굵은 유니코드 헤더(쇼케이스/음감회/DJ)
SECTION_SHOWCASE_HEAD = re.compile(
    r'^\s*\[\s*(?:Showcase|쇼케이스)\s*(?:\|\s*(?:Showcase|쇼케이스))?\s*\]\s*$',
    re.IGNORECASE | re.MULTILINE
)
SECTION_LISTEN_HEAD = re.compile(
    r'^\s*\[\s*(?:Listening\s*Session|음감회)\s*(?:\|\s*(?:Listening\s*Session|음감회))?\s*\]\s*$',
    re.IGNORECASE | re.MULTILINE
)
SECTION_DJ_HEAD = re.compile(
    r'^\s*\[\s*(?:DJ|디제이)\s*(?:\|\s*(?:DJ|디제이))?\s*\]\s*$',
    re.IGNORECASE | re.MULTILINE
)

# =========================
# 유틸
# =========================

def _extract_title_from_head(text: str) -> Optional[str]:
    """맨 앞쪽 짧은 헤더 라인을 타이틀로 채택"""
    for ln in text.splitlines():
        s = _strip_space(ln)
        if not s:
            continue
        # 너무 길면 배제 (포스터 설명문 방지), 토막 단어만도 배제
        if 3 <= len(s) <= 80 and re.search(r'(Vol\.?\s*\d+|LIVE|Vol|공연|Concert|Show|Night|Festival)', s, re.IGNORECASE):
            return s
        # 위 키워드 없어도 대문자/Title Case가 강하면 허용
        if 3 <= len(s) <= 60 and re.search(r'[A-Za-z].*[A-Za-z]', s) and not re.search(r'[.:]\s*$', s):
            return s
    return None

def _to_ascii_compat(s: str) -> str:
    """굵은 유니코드, 전각/장식 기호를 ASCII 근사치로 정규화"""
    if not s:
        return s
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = (s.replace('◉', ' ')
           .replace('✹', '-')
           .replace('•', '-')
           .replace('▪️', '-')
           .replace('●', '-')
           .replace('◦', '-')
           .replace('\u3000', ' ')
           .replace('–', '-')
           .replace('—', '-'))
    return s

def _strip_josa(name: str) -> str:
    return POSTFIX_JOSA.sub('', _strip_space(name))

def _find_dates_with_span(text: str):
    return [(_join_full_date(m), m.start(), m.end()) for m in RE_FULLDATE_OPT_YEAR.finditer(text)]

def _find_times_with_span(text: str):
    out = []
    for m in RE_TIME.finditer(text):
        out.append((_normalize_time(m.group(0)), m.start(), m.end()))
    # 영어 단건
    for m in RE_EN_TIME.finditer(text):
        h = int(m.group('h'))
        out.append((f"{h}시", m.start(), m.end()))
    # 영어 범위: 시작만
    for m in RE_EN_TIME_RANGE.finditer(text):
        h1 = int(m.group('h1'))
        out.append((f"{h1}시", m.start(), m.end()))
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
        digits = re.sub(r'[^\d]', '', p)  # 콤마/통화문자 등 제거
        return int(digits) if digits else -1
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
    raw = _strip_space(s)
    # 통화 표기 통일
    raw = re.sub(r'\bKRW\b', '원', raw, flags=re.IGNORECASE)
    raw = raw.replace('₩', '원').replace('￦', '원')
    m = RE_PRICE.search(raw)
    if not m:
        return _strip_space(raw)
    amount = m.group('amount')
    if ',' not in amount and len(amount) > 3:
        amount = f"{int(amount):,}"
    return f"{amount}원"

def _parse_kr_price(s: str) -> Optional[str]:
    raw = _strip_space(s)
    # 통화 표기 통일
    raw = re.sub(r'\bKRW\b', '원', raw, flags=re.IGNORECASE)
    raw = raw.replace('₩', '원').replace('￦', '원')
    m = RE_KR_PRICE.fullmatch(raw)
    if not m:
        return None
    man = m.group('man')
    chon = m.group('chon')
    num = m.group('num')

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
    # 한글형
    for m in RE_FULLDATE_OPT_YEAR.finditer(text):
        out.append(_join_full_date(m))
    # 08.02 / 08/02 / 08-02
    for m in RE_MD_DOTTED.finditer(text):
        y = m.group('y'); mth = m.group('m'); day = m.group('d'); dow = m.group('dow')
        core = f"{int(mth)}월 {int(day)}일"
        if dow:
            core += f" ({dow})"
        if y and len(y) == 4:
            out.append(f"{y}년 {core}")
        else:
            out.append(core)
    return _dedupe(out)

def _find_all_times(text: str) -> List[str]:
    out = []
    # 한글/숫자
    for m in RE_TIME.finditer(text):
        out.append(_normalize_time(m.group(0)))
    # 영어 단건 → 'X시'
    for m in RE_EN_TIME.finditer(text):
        h = int(m.group('h'))
        out.append(f"{h}시")
    # 영어 범위 → 시작 시각만 'X시'
    for m in RE_EN_TIME_RANGE.finditer(text):
        h1 = int(m.group('h1'))
        out.append(f"{h1}시")
    return _dedupe(out)

def _merge_date_with_weekday(dates: list[str]) -> list[str]:
    """DATE 리스트에서 '날짜' + '(요일)' 형태를 하나로 합침."""
    merged = []
    skip_next = False
    for i, val in enumerate(dates):
        if skip_next:
            skip_next = False
            continue
        # 날짜 + 다음 항목이 (요일)인 경우
        if i + 1 < len(dates) and re.match(r'^\(?[월화수목금토일]\)?$', dates[i+1]) or re.match(r'^\([월화수목금토일]\)$', dates[i+1]):
            weekday = dates[i+1].strip()
            weekday = weekday if weekday.startswith("(") else f"({weekday})"
            merged.append(f"{val} {weekday}")
            skip_next = True
        else:
            merged.append(val)
    return merged

def _pick_start_time_only(text: str, times: List[str]) -> List[str]:
    """
    'OPEN 18:00' / 'START 18:30' 같이 있으면 START에 붙은 시각만 남김
    """
    # START 라인에서 숫자시각(또는 hh:mm) 우선 추출
    m = re.search(r'\bSTART\b[^\n]{0,20}?(\d{1,2}(?::\d{2})?)', text, flags=re.IGNORECASE)
    if m:
        val = m.group(1)
        # '18:30' → '18:30', '18' → '18시'
        if ':' in val:
            keep = val
        else:
            keep = f"{int(val)}시"
        return [keep]
    return times

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
    candidates = list(venues)
    for m in re.finditer(r'([가-힣A-Za-z·]+)\s+([가-힣A-Za-z·0-9]+?)\s*(?:에서|에)\b', text):
        area, place = m.group(1), m.group(2)
        if _is_probable_venue_word(place) or _is_probable_venue_word(area):
            candidates.append(_strip_space(f"{area} {place}"))
    return _dedupe([_clean_venue_token(v) for v in candidates])

# =========================
# LINEUP 강화 유틸
# =========================

def _clean_artist_name(s: str) -> str:
    s = _strip_space(s)
    s = re.sub(r'^[\-\–—·\•\▶\✅\[\(]+', '', s)
    s = re.sub(r'[@\(].*$', '', s)   # 괄호/핸들 이후 삭제
    s = re.sub(r'[\*\_`]+', '', s)   # 서식문자 제거
    s = POSTFIX_JOSA.sub('', s)      # 조사 제거
    s = _strip_space(s)
    return s

def _looks_like_artist(s: str) -> bool:
    if not s or len(s) <= 1: return False
    if any(bad in s for bad in LINEUP_BLACKLIST_SUBSTR): return False
    if RE_PRICE.search(s) or RE_FULLDATE_OPT_YEAR.search(s): return False
    if s.count(' ') >= 8 and not s.lower().startswith('dj '):
        return False
    return True

def _harvest_names_from_lines(lines: list) -> list:
    out = []
    for ln in lines:
        ln = _strip_space(ln)
        if not ln: continue
        m = NAME_CAND.match(ln)
        if not m: continue
        name = _clean_artist_name(m.group('name'))
        if _looks_like_artist(name):
            out.append(name)
    return out

def _collect_lines_in_lineup_sections(text: str) -> list:
    lines = text.splitlines()
    take = False
    buf, collected = [], []
    for ln in lines:
        if LINEUP_SEC_HEAD.search(ln):
            if buf: collected.extend(buf); buf=[]
            take = True
            continue
        if take:
            if SECTION_BREAK.search(ln):
                take = False
                if buf: collected.extend(buf); buf=[]
                continue
            buf.append(ln)
    if buf: collected.extend(buf)
    return collected

def _collect_bullet_lines(text: str) -> list:
    return [m.group(1) for m in BULLET_LINE.finditer(text)]

def _collect_lines_in_named_section(text: str, head_re: re.Pattern) -> list:
    lines = text.splitlines()
    take = False
    buf, out = [], []
    for ln in lines:
        if head_re.search(ln):
            if buf: out.extend(buf); buf = []
            take = True
            continue
        if take:
            if (SECTION_BREAK.search(ln) or
                LINEUP_SEC_HEAD.search(ln) or
                SECTION_SHOWCASE_HEAD.search(ln) or
                SECTION_LISTEN_HEAD.search(ln) or
                SECTION_DJ_HEAD.search(ln)):
                take = False
                if buf: out.extend(buf); buf = []
                continue
            buf.append(ln)
    if buf: out.extend(buf)
    return out

def _extract_lineup_and_handles_linewise_from(text_blocks: List[str]) -> List[tuple]:
    """주어진 블록들에서 '이름 @handle' 라인을 (name, @handle)로 추출"""
    pairs = []
    pat = re.compile(r'^(?P<name>[^@#\|\[\]\(\)]+)\s+(@[A-Za-z0-9._]+)\s*$', re.UNICODE)
    for block in text_blocks:
        for ln in block.splitlines():
            ln = _strip_space(ln)
            if not ln:
                continue
            m = pat.match(ln)
            if not m:
                continue
            name = _clean_artist_name(m.group('name'))
            handle = m.group(2)
            if _looks_like_artist(name):
                pairs.append((name, handle))
    return pairs

def _pair_name_then_handle(text: str, blocks: List[str]) -> List[tuple]:
    """
    '이름' 단독 라인 다음 줄이 '@handle' 인 2줄 패턴을 (name, handle)로 매칭.
    blocks: 쇼케이스/음감회/일반 라인업 섹션(=DJ 제외) 텍스트 블록 리스트
    """
    pairs = []
    name_pat = re.compile(r'^(?P<name>[^@#\|\[\]\(\)]+?)\s*$', re.UNICODE)
    handle_pat = re.compile(r'^(@[A-Za-z0-9._]+)\s*$')
    for block in blocks:
        lines = [ _strip_space(x) for x in block.splitlines() ]
        for i, ln in enumerate(lines[:-1]):
            m1 = name_pat.match(ln)
            m2 = handle_pat.match(lines[i+1])
            if not (m1 and m2): 
                continue
            name = _clean_artist_name(m1.group('name'))
            handle = m2.group(1)
            if _looks_like_artist(name):
                pairs.append((name, handle))
    return pairs

def _filter_instagram_handles(text: str, fields: Dict, pairs: List[tuple]) -> None:
    """라인업 옆 핸들 우선. 없으면 첫 2줄(헤더)의 핸들은 제거"""
    if pairs:
        keep = {h for _, h in pairs}
        fields['INSTAGRAM'] = [h for h in fields.get('INSTAGRAM', []) if h in keep]
        return
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header = "\n".join(lines[:2])
    header_handles = set(re.findall(r'@[A-Za-z0-9._]+', header))
    fields['INSTAGRAM'] = [h for h in fields.get('INSTAGRAM', []) if h not in header_handles]

def _dj_block_names(text_norm: str) -> set:
    """정규화 텍스트에서 DJ 섹션 이름 후보 수확(라인업에서 강제 제외용)"""
    dj_lines = _collect_lines_in_named_section(text_norm, SECTION_DJ_HEAD)
    names = set(_harvest_names_from_lines(dj_lines))
    bullet_dj = _collect_bullet_lines("\n".join(dj_lines))
    names |= set(_harvest_names_from_lines(bullet_dj))
    return names

def _prefer_longer_unique(items: List[str]) -> List[str]:
    items = _dedupe([_strip_space(x) for x in items if x])
    keep = []
    for i, a in enumerate(items):
        if any((a != b) and (a in b) for b in items):
            continue
        keep.append(a)
    return keep

# =========================
# 규칙 본체
# =========================

def _rescan_and_fix_prices(text: str, tokens: List[str], fields: Dict) -> None:
    # 통화 표기 선정규화
    text = re.sub(r'\bKRW\b', '원', text, flags=re.IGNORECASE).replace('₩','원').replace('￦','원')

    price = list(fields.get('PRICE', []) or [])
    onsite = list(fields.get('PRICE_ONSITE', []) or [])

    def tok_span_to_str(i, j):
        return _strip_space(''.join(tokens[i:j]))

    def pick_best_amount(slices: List[str]) -> Optional[str]:
        best = None; best_val = -1
        for cand in slices:
            s = _strip_space(cand)
            s = re.sub(r'\bKRW\b', '원', s, flags=re.IGNORECASE).replace('₩','원').replace('￦','원')
            v = _parse_kr_price(s)
            if not v and RE_PRICE.fullmatch(_strip_space(s or '')):
                v = _normalize_price(s)
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
        words = re.findall(r'[0-9가-힣,]+(?:\s*(?:KRW|원))?|[0-9가-힣,]+', win, flags=re.IGNORECASE)
        best = pick_best_amount(words)
        if best: price.append(best)
    for m in re.finditer(r'(현매|현장)', text):
        win = _window_after(text, m.end(), 60)
        words = re.findall(r'[0-9가-힣,]+(?:\s*(?:KRW|원))?|[0-9가-힣,]+', win, flags=re.IGNORECASE)
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
    opens = list(RE_TICKET_OPEN.finditer(text))
    if not opens:
        fields['TICKET_OPEN_DATE'] = _dedupe(fields.get('TICKET_OPEN_DATE', []))
        fields['TICKET_OPEN_TIME'] = _dedupe(fields.get('TICKET_OPEN_TIME', []))
        return

    all_dates = _find_dates_with_span(text)
    all_times = _find_times_with_span(text)

    to_dates, to_times = [], []
    for op in opens:
        anchor = (op.start() + op.end()) // 2

        # 날짜: 앵커와의 거리가 가장 가까운 것 1개
        best_d, best = 10**9, None
        for dtxt, ds, de in all_dates:
            dist = min(abs(anchor - ds), abs(anchor - de))
            if dist < best_d and dist <= 80:
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
    """모델 후보 + (섹션/불릿/서술형/사전) 기반 라인업 보강. DJ 섹션 제외."""
    lineup = fields.get('LINEUP', []) or []
    out = []

    for x in lineup:
        s = _clean_artist_name(x)
        if _looks_like_artist(s):
            out.append(s)

    if text:
        sc_lines = _collect_lines_in_named_section(text, SECTION_SHOWCASE_HEAD)
        ls_lines = _collect_lines_in_named_section(text, SECTION_LISTEN_HEAD)
        dj_lines = _collect_lines_in_named_section(text, SECTION_DJ_HEAD)

        out += _harvest_names_from_lines(sc_lines)
        out += _harvest_names_from_lines(ls_lines)

        sec_lines = _collect_lines_in_lineup_sections(text)
        sec_names = _harvest_names_from_lines(sec_lines)
        dj_names = set(_harvest_names_from_lines(dj_lines))
        sec_names = [n for n in sec_names if n not in dj_names]
        out += sec_names

        bullet_all = _collect_bullet_lines(text)
        bullet_dj  = _collect_bullet_lines("\n".join(dj_lines))
        bullet_lines = [b for b in bullet_all if b not in bullet_dj]
        out += _harvest_names_from_lines(bullet_lines)

        for m in re.finditer(r'(?:한국팀|국내팀|게스트)\s*으로는\s*([^\n]+?)가\s*(?:같이|함께)\s*합니다', text):
            cand = _clean_artist_name(m.group(1))
            if _looks_like_artist(cand):
                out.append(cand)

        sc_ls_block = '\n'.join(sc_lines + ls_lines + sec_lines)
        for m in re.finditer(r'^\s*(.+?)\s*\([^)]+\)\s*$', sc_ls_block, flags=re.MULTILINE):
            cand = _clean_artist_name(m.group(1))
            if _looks_like_artist(cand):
                out.append(cand)

        if lexicons and lexicons.get('artists'):
            safe_blocks = sc_ls_block
            for name in lexicons['artists']:
                if re.search(re.escape(name), safe_blocks, flags=re.IGNORECASE):
                    out.append(name)

    out = [s for s in (_strip_space(x) for x in out) if _looks_like_artist(s)]
    fields['LINEUP'] = _dedupe(out)

def _fix_instagram(fields: Dict) -> None:
    inst = list(fields.get('INSTAGRAM', []) or [])
    for x in fields.get('LINEUP', []):
        if x.startswith('@'): inst.append(x)
    fields['INSTAGRAM'] = _dedupe(inst)

def _fix_venues(text: str, fields: Dict, lexicons: Optional[Dict]=None) -> None:
    def _is_blacklisted(v: str) -> bool:
        return any(w in v for w in VENUE_BLACKLIST_WORDS)

    venues = [_clean_venue_token(v) for v in (fields.get('VENUE', []) or [])]
    venues = _combine_area_and_venue(text, venues)

    for m in re.finditer(r'([가-힣A-Za-z·0-9]+)(?:에서|에)\b', text):
        w = _clean_venue_token(m.group(1))
        if _is_blacklisted(w):
            continue
        if _is_probable_venue_word(w):
            venues.append(w)

    venues = [_strip_space(v) for v in venues if v and not _is_blacklisted(v)]
    fields['VENUE'] = _prefer_longer_unique(_dedupe(venues))

def _normalize_dates(tokens: List[str], text: str, fields: Dict) -> None:
    dates = list(fields.get('DATE', []) or [])
    dates = _merge_year_with_partial_dates(tokens, dates)
    dates.extend(_find_all_full_dates(text))
    fields['DATE'] = _dedupe([_strip_space(d) for d in dates])

def _normalize_times(text: str, fields: Dict) -> None:
    times = list(fields.get('TIME', []) or [])
    times.extend(_find_all_times(text))
    fields['TIME'] = _dedupe([_strip_space(t) for t in times])

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

def _is_valid_full_or_partial_date(s: str) -> bool:
    return bool(RE_FULLDATE_OPT_YEAR.fullmatch(s))

def _sanitize_ticket_open(fields: Dict) -> None:
    tod = []
    for d in fields.get('TICKET_OPEN_DATE', []) or []:
        d = _strip_space(d)
        if _is_valid_full_or_partial_date(d):
            tod.append(d)
    fields['TICKET_OPEN_DATE'] = _dedupe(tod)

# =========================
# 본체
# =========================

def apply_regex_postrules(
    text: str,
    tokens: List[str],
    fields: Dict[str, List[str]],
    lexicons: Optional[Dict]=None
) -> Dict[str, List[str]]:
    """규칙 기반 후처리의 본체"""
    # 섹션/불릿 인식을 위해 정규화 텍스트 사용
    t = _to_ascii_compat(text)

    # 섹션 수집(정규화 텍스트 기준)
    sc_lines = _collect_lines_in_named_section(t, SECTION_SHOWCASE_HEAD)
    ls_lines = _collect_lines_in_named_section(t, SECTION_LISTEN_HEAD)
    dj_lines = _collect_lines_in_named_section(t, SECTION_DJ_HEAD)
    sec_lines = _collect_lines_in_lineup_sections(t)

    # 먼저 '이름 @handle' (DJ 제외 블록만)
    allowed_blocks = ['\n'.join(sc_lines), '\n'.join(ls_lines), '\n'.join(sec_lines)]
    pairs_same_line = _extract_lineup_and_handles_linewise_from(allowed_blocks)  # '이름 @핸들' 한 줄
    pairs_two_lines = _pair_name_then_handle(t, allowed_blocks)                  # '이름' ↵ '@핸들'
    pairs = _dedupe(pairs_same_line + pairs_two_lines)

    if pairs:
        fields.setdefault('LINEUP', [])
        fields.setdefault('INSTAGRAM', [])
        for name, handle in pairs:
            fields['LINEUP'].append(name)
            fields['INSTAGRAM'].append(handle)
        fields['LINEUP'] = _dedupe(fields['LINEUP'])
        fields['INSTAGRAM'] = _dedupe(fields['INSTAGRAM'])

    # 인스타 핸들 정리(헤더 핸들 제거/라인업 옆 핸들 유지) — 원문 기준
    _filter_instagram_handles(text, fields, pairs)

    # 날짜/시간/가격/오픈 (정규화 텍스트 t 기준)
    _normalize_dates(tokens, t, fields)
    _normalize_times(t, fields)
    # START 시간만 남기기 (있다면)
    fields['TIME'] = _pick_start_time_only(t, fields.get('TIME', []))
    _ticket_open_mapping(t, fields)
    _rescan_and_fix_prices(t, tokens, fields)

    # 라인업 보강 (DJ 제외)
    def _harvest_excluding_dj():
        out = []
        out += _harvest_names_from_lines(sc_lines)
        out += _harvest_names_from_lines(ls_lines)
        sec_names = _harvest_names_from_lines(sec_lines)
        out += sec_names
        bullet_all = _collect_bullet_lines(t)
        bullet_dj  = _collect_bullet_lines('\n'.join(dj_lines))
        bullet_lines = [b for b in bullet_all if b not in bullet_dj]
        out += _harvest_names_from_lines(bullet_lines)
        return out

    lineup = fields.get('LINEUP', []) or []
    lineup += _harvest_excluding_dj()

    # 사전 매칭 — DJ 제외 블록에만 적용
    if lexicons and lexicons.get('artists'):
        safe_blocks = '\n'.join(allowed_blocks)
        for name in lexicons['artists']:
            if re.search(re.escape(name), safe_blocks, flags=re.IGNORECASE):
                lineup.append(name)

    # DJ 블록 이름 강제 제거
    dj_names = _dj_block_names(t)
    lineup = [n for n in map(_strip_space, lineup) if _looks_like_artist(n) and n not in dj_names]
    fields['LINEUP'] = _dedupe(lineup)

    # TITLE 비었으면 헤더에서 보강
    if not fields.get('TITLE'):
        t_title = _extract_title_from_head(text)
        if t_title:
            fields['TITLE'] = [t_title]

    # VENUE/마무리
    _fix_instagram(fields)
    _fix_venues(t, fields, lexicons=lexicons)
    _sanitize_ticket_open(fields)
    _tidy_title(fields)
    _final_tidy(fields)
    # 날짜 + 요일 합치기
    if fields.get("DATE"):
        fields["DATE"] = _merge_date_with_weekday(fields["DATE"])
    return fields

# =========================
# BIO/스키마/렉시콘 유틸
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
    """CSV에서 아티스트/공연장 사전 로드. 한 열에 하나씩."""
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
    text = """
Night Of Victim Vol.63

The Redemptions
@band_the_redemptions
Nimbus
@nimbus_grunge
Guny N Guns
@guny_n_guns
Varnish
@varnish_band

2025. 8. 17.(일)
Club VICTIM
OPEN 18:00
START 18:30
25,000 KRW

i 현매만 가능(계좌이체, 카드가능)
ii 외부 음식 반입가능, 외부주류 반입금지

문의
DM or clubxvictim@gmail.com

Special Thanx to @coszise
"""
    toks = text.split()
    labels = ['O'] * len(toks)
    merged = merge_model_and_rules(toks, labels)
    from pprint import pprint
    pprint(merged)
