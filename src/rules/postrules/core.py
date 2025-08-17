# /rules/postrules/core.py
# -*- coding: utf-8 -*-
from typing import Dict, List, Optional
import re
from .textutils import _to_ascii_compat, _strip_space, _dedupe
from .lineup import (
    extract_pairs_anywhere, harvest_lineup_names, collect_all_handles,
    _tidy_title, _fix_instagram,
)
from .datetimex import _normalize_dates, _normalize_times, _merge_date_with_weekday, _pick_start_time_only
from .tickets import _ticket_open_mapping, _sanitize_ticket_open
from .price import _rescan_and_fix_prices
from .venue import _fix_venues

VENUE_HEADER = re.compile(r'(?:^|[^\S\r\n]).{0,10}?(?:Venue|VENUE|장소)\s*[:：]\s*(.+)$', re.IGNORECASE | re.MULTILINE)

# ── NEW: 해시태그 배제 & 점수 기반TITLE 선택기 ───────────────────────────
_TITLE_NOTICE = re.compile(r'(NOTICE|공지|안내|현장\s*안내)', re.IGNORECASE)
_TITLE_BAD    = re.compile(r'(공연\s*시간|공연시간)', re.IGNORECASE)
_TITLE_HINT   = re.compile(r'(단독\s*콘서트|콘서트|공연|쇼케이스|LIVE)', re.IGNORECASE)
# “날짜만” 같은 라인 배제(짧은 길이의 날짜/요일/숫자/구분자 조합)
_DATEISH = re.compile(
    r'^[\s0-9./()\-\u00B7]*(?:년|월|일|Mon|Tue|Wed|Thu|Fri|Sat|Sun|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[\s0-9./()\-\u00B7,]*$',
    re.IGNORECASE
)

def _pick_best_title_from_text(text: str) -> str | None:
    """
    규칙:
      - 해시태그(#…) 라인: 절대 제외
      - '공지/안내' 라인: 완전 제외 X, 다만 후순위(점수↓)
      - 날짜만 같은 라인(짧고 날짜표현 위주): 제외
      - 괄호형(<…>, 【…】 등): 가산점
      - '공연/콘서트/쇼케이스/LIVE' 키워드 포함: 가산점
      - ':' 포함: 가산점
      - 길이 6–60자: 가산점
      - 최고 점수(동점이면 더 긴 것) 채택
    """
    best, best_score = None, -10
    for raw in (ln.strip() for ln in (text or '').splitlines()):
        if not raw:
            continue
        # 꾸밈 글자/기호 정규화 (NFKC + 불릿 치환)
        raw = _to_ascii_compat(raw)
        if raw.startswith('#'):
            continue  # 해시태그는 절대 TITLE 후보 아님
        # 좌측 장식/이모지/불릿 제거
        ln = re.sub(r'^[\s📢🗓️🕰️⏱️🎪🎫💵👨🏻‍⚖️👨🏻‍💻\-\–—•●◦▪️_:\|·]+', '', raw)
        ln = re.sub(r'^\-\s*', '', ln)  # "- title" 형태도 정리
        if not ln:
            continue
        # 짧은 날짜 전용 라인은 배제(태그성/날짜성 제목 방지)
        if len(ln) <= 30 and _DATEISH.fullmatch(ln or ""):
            continue
        score = 0
        # 괄호/브라켓형 제목(공지어도 후보로 남김, 단 점수↓)
        m = re.match(r'^[\[\(<【]\s*(.+?)\s*[】>\)\]]$', ln)
        if m:
            ln = _strip_space(m.group(1))
            score += 2
        # 힌트 키워드
        if _TITLE_HINT.search(ln):
            score += 3
        # 콜론 포함(부제/형식 제목)
        if ':' in ln or '：' in ln:
            score += 1
        # 적당 길이
        if 6 <= len(ln) <= 60:
            score += 1
        # 공지/안내는 후순위(감점)
        if _TITLE_NOTICE.search(ln):
            score -= 2
        # '공연시간' 같은 안내성 라인은 강한 감점
        if _TITLE_BAD.search(ln):
            score -= 4
        if score > best_score or (score == best_score and best and len(ln) > len(best)):
            best, best_score = ln, score
    return best if best_score > -10 else None

def _fallback_title(text: str) -> Optional[str]:
    # 새 점수 기반 선택기로 대체
    return _pick_best_title_from_text(text)

def _extract_venue_header(text: str) -> Optional[str]:
    m = VENUE_HEADER.search(text)
    if not m: return None
    v = m.group(1)
    v = re.sub(r'\(.*?\)$', '', v)                 # (주소/좌석수) 꼬리 괄호 제거
    v = re.sub(r'@[A-Za-z0-9._]+', '', v)          # 핸들 제거
    v = re.sub(r'(프로필|링크|참조).*$', '', v)    # 꼬리 문구 제거
    v = _strip_space(v)
    return v if v else None

def apply_regex_postrules(text: str, tokens: List[str], fields: Dict[str, List[str]], lexicons: Optional[Dict]=None) -> Dict[str, List[str]]:
    t = _to_ascii_compat(text)

    # 1) LINEUP / INSTAGRAM — with 블록/같은 줄/타임테이블 모두 수확
    pairs = extract_pairs_anywhere(t)
    lineup = [n for n, h in pairs if n]
    handles_from_pairs = [h for n, h in pairs if h]

    # 인스타: 쌍에서 나온 핸들 우선, 없으면 전체 수집
    handles_all = handles_from_pairs if handles_from_pairs else collect_all_handles(t)

    fields['LINEUP'] = _dedupe([_strip_space(n) for n in lineup if _strip_space(n)])

    # 아티스트 핸들 vs 공연장/주최 핸들 분리
    from .lineup import _filter_instagram_handles
    artist_handles, venue_handles = _filter_instagram_handles(text=t, handles=handles_all)
    fields['INSTAGRAM'] = _dedupe(artist_handles)
    if venue_handles:
        cur_v_insta = list(fields.get('V_INSTA', []) or [])
        fields['V_INSTA'] = _dedupe(cur_v_insta + venue_handles)

    # 2) 날짜/시간/티켓오픈/가격
    _normalize_dates(tokens, t, fields)
    _normalize_times(t, fields)
    fields['TIME'] = _pick_start_time_only(t, fields.get('TIME', []))
    # '입장/door open' 주변에서 나온 시간들은 버림
    if fields.get('TIME'):
        kept=[]
        for tm in fields['TIME']:
            if not tm: continue
            # 시간 문자열 주변 0~15자 범위에 '입장|door open'이 있으면 제외
            pat = re.compile(rf'(입장|door\s*open)[^\n]{{0,15}}{re.escape(tm)}|{re.escape(tm)}[^\n]{{0,15}}(입장|door\s*open)', re.IGNORECASE)
            if pat.search(t): 
                continue
            kept.append(tm)
        fields['TIME'] = kept or fields['TIME']
    _ticket_open_mapping(t, fields)
    _rescan_and_fix_prices(t, tokens, fields)

    # 2-1) DATE 잡음 제거: '25.08' 같은 부분 매치는 버림
    fields['DATE'] = [d for d in fields.get('DATE', []) if not re.fullmatch(r'\d{1,2}\.\d{2}', d)]

    # 3) 라인업 이름 보강
    extra = harvest_lineup_names(t)
    fields['LINEUP'] = _dedupe(fields['LINEUP'] + extra)

    # 4) VENUE — 헤더 라인으로 긴 형태를 우선 확보
    long_v = _extract_venue_header(t) or _extract_venue_header(text)
    if long_v:
        cur = fields.get('VENUE', []) or []
        if (not cur) or (len(long_v) > max(len(x) for x in cur)):
            fields['VENUE'] = [long_v]

    # 5) TITLE 보정: 폴백 후보가 있으면 기존 공지/해시태그/짧은 타이틀을 교체
    tt = _fallback_title(text)
    if tt:
        cur = [ _strip_space(x) for x in (fields.get('TITLE', []) or []) ]
        def _looks_bad_title(x: str) -> bool:
            if not x or len(x) <= 2: return True
            if x.startswith('#'): return True
            if _TITLE_NOTICE.search(x): return True
            return False
        if (not cur) or any(_looks_bad_title(x) for x in cur) or (len(tt) > max(len(x) for x in cur)):
            fields['TITLE'] = [tt]
    _tidy_title(fields)

    # 6) 마무리
    _fix_instagram(fields)
    _fix_venues(t, fields, lexicons=lexicons)
    _sanitize_ticket_open(fields)

    if fields.get("DATE"):
        fields["DATE"] = _merge_date_with_weekday(fields["DATE"])
    return fields
