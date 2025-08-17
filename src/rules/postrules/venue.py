# /rules/postrules/venue.py
# 공연장 후보 수집/정리
# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Optional
from .patterns import VENUE_KEYWORDS, VENUE_BLACKLIST_WORDS
from .textutils import _strip_space, _dedupe, _prefer_longer_unique

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

def _fix_venues(text: str, fields: Dict, lexicons: Optional[Dict]=None) -> None:
    def _is_blacklisted(v: str) -> bool:
        return any(w in v for w in VENUE_BLACKLIST_WORDS)

    venues = [_clean_venue_token(v) for v in (fields.get('VENUE', []) or [])]
    venues = _combine_area_and_venue(text, venues)

    for m in re.finditer(r'([가-힣A-Za-z·0-9]+)(?:에서|에)\b', text):
        w = _clean_venue_token(m.group(1))
        if _is_blacklisted(w): continue
        if _is_probable_venue_word(w): venues.append(w)

    venues = [_strip_space(v) for v in venues if v and not _is_blacklisted(v)]
    # NEW(narrowed): "@ Alleyway Taphouse, Suwon" 같은 라인만 수확
    #  - "@ " (공백)으로 시작해야 함 (순수 핸들 "@username" 배제)
    #  - 장소 키워드가 있거나, 쉼표가 포함된 지명형식일 때만 수용
    for ln in text.splitlines():
        raw = ln.strip()
        if not raw.startswith('@ '):
            continue
        seg = raw[2:].strip()  # "@ " 제거
        # 장식/부가정보 제거
        seg = seg.split('~')[0]
        seg = seg.split('(')[0]
        seg = seg.rstrip(',').strip()
        # 안전장치: 키워드 또는 쉼표(지역 구분) 필요
        if not seg:
            continue
        if not (_is_probable_venue_word(seg) or (',' in seg)):
            continue
        if not _is_blacklisted(seg):
            venues.append(seg)
    m = re.search(
        r'^\s*(?:[•●◦▪️\-\–\—✦❏]\s*)*(?:Venue|VENUE|장소)\s*[:：]\s*([^\n(]+)',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )   
    if m:
        v = _clean_venue_token(m.group(1))
        if v and not _is_blacklisted(v):
            venues.append(v)
    fields['VENUE'] = _prefer_longer_unique(_dedupe(venues))
