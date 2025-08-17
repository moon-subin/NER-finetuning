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

def _fallback_title(text: str) -> Optional[str]:
    for ln in (x.strip() for x in text.splitlines()):
        if not ln or ln.startswith('['): continue
        # < … 안내 > 같은 표기 정리
        if re.fullmatch(r'<\s*.+?\s*>', ln):
            inner = re.sub(r'^[<]\s*|\s*[>]$', '', ln)
            inner = re.sub(r'\s*안내\s*$', '', inner)
            inner = _strip_space(inner)
            if inner:
                return inner
        # 일정/장소/예매 안내 라인은 제목 후보에서 제외
        if re.match(r'^(일시|장소|티켓|예매\s*오픈)\s*[:：]', ln):
            continue
        ln = ln.lstrip('_-•●◦❏').strip()
        if ':' in ln and re.search(r'[A-Za-z가-힣]', ln):
            left, right = ln.split(':', 1)
            if _strip_space(left) and _strip_space(right):
                return _strip_space(left + ' : ' + right)
    return None

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

    # 5) TITLE 보정: ':'로 시작하거나 2자 이하인 기존 타이틀은 폴백으로 교체
    if (not fields.get('TITLE')) or any(_strip_space(x).startswith(':') or len(_strip_space(x)) <= 2 for x in fields['TITLE']):
        tt = _fallback_title(text)
        if tt: fields['TITLE'] = [tt]
    _tidy_title(fields)

    # 6) 마무리
    _fix_instagram(fields)
    _fix_venues(t, fields, lexicons=lexicons)
    _sanitize_ticket_open(fields)

    if fields.get("DATE"):
        fields["DATE"] = _merge_date_with_weekday(fields["DATE"])
    return fields
