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
    fields['VENUE'] = _prefer_longer_unique(_dedupe(venues))
