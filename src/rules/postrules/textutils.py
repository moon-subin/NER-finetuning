# /rules/postrules/textutils.py
# 공통 유틸(정규화/strip/dedupe/가격 숫자화 등)
# -*- coding: utf-8 -*-
import re, unicodedata
from typing import List, Optional
from .patterns import RE_PRICE, RE_KR_PRICE

def _strip_space(s: str) -> str:
    return re.sub(r'\s+', ' ', str(s)).strip()

def _dedupe(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x is None: continue
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _to_ascii_compat(s: str) -> str:
    if not s: return s
    # NFKD + 결합문자 제거 → 한글 자모 분해 문제 발생
    # ⇒ 폭/호환만 정규화(NFKC)하고, 불릿/기호만 치환
    s = unicodedata.normalize('NFKC', s)
    s = (s.replace('◉', ' ').replace('✹', '-').replace('•', '-').replace('▪️', '-')
           .replace('●', '-').replace('◦', '-').replace('\u3000', ' ')
           .replace('–', '-').replace('—', '-'))
    return s

def _price_to_int(p: str) -> int:
    try:
        digits = re.sub(r'[^\d]', '', p or '')
        return int(digits) if digits else -1
    except:
        return -1

def _normalize_price(raw: str) -> str:
    raw = _strip_space(raw).replace('₩','원').replace('￦','원')
    raw = re.sub(r'\bKRW\b', '원', raw, flags=re.IGNORECASE)
    m = RE_PRICE.search(raw)
    if not m:
        return _strip_space(raw)
    amount = m.group('amount')
    if ',' not in amount and len(amount) > 3:
        amount = f"{int(amount):,}"
    return f"{amount}"

def _parse_kr_price(raw: str) -> Optional[str]:
    s = _strip_space(raw).replace('₩','원').replace('￦','원')
    s = re.sub(r'\bKRW\b', '원', s, flags=re.IGNORECASE)
    m = RE_KR_PRICE.fullmatch(s)
    if not m: return None
    man, chon, num = m.group('man'), m.group('chon'), m.group('num')
    has_unit = ('만' in s) or ('천' in s) or ('원' in s)
    long_num = bool(num and len(num.replace(',', '')) >= 4)
    if not (man or chon or has_unit or long_num): return None
    total = 0
    if man:  total += int(man) * 10_000
    if chon: total += int(chon) * 1_000
    if num:  total += int(num.replace(',', ''))
    if total <= 0: return None
    return f"{total:,}"

def _prefer_longer_unique(items: List[str]) -> List[str]:
    items = _dedupe([_strip_space(x) for x in items if x])
    keep = []
    for a in items:
        if any((a != b) and (a in b) for b in items):
            continue
        keep.append(a)
    return keep
