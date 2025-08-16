# 날짜/시간 관련 유틸과 통합(normalize, merge 등)
# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Tuple
from .patterns import (
    RE_FULLDATE_OPT_YEAR, RE_MD_DOTTED, RE_EN_TIME, RE_EN_TIME_RANGE, RE_TIME
)
from .textutils import _strip_space, _dedupe

def _join_full_date(m: re.Match) -> str:
    y = m.group('year'); month = m.group('month'); day = m.group('day'); dow = m.group('dow') or ''
    core = f"{int(month)}월 {int(day)}일"
    if dow: core += f" ({dow})"
    return (f"{y}년 " if y else "") + core

def _normalize_time(s: str) -> str:
    m = RE_TIME.search(s)
    if not m: return _strip_space(s)
    ampm = (m.group('ampm') or '').strip()
    hour = m.group('hour1') or m.group('hour2') or m.group('hour3')
    minute = m.group('min1') or m.group('min2')
    t = f"{hour}:{minute}" if minute else f"{hour}시"
    return _strip_space(f"{ampm} {t}".strip())

def _find_dates_with_span(text: str):
    return [(_join_full_date(m), m.start(), m.end()) for m in RE_FULLDATE_OPT_YEAR.finditer(text)]

def _find_times_with_span(text: str):
    out = []
    for m in RE_TIME.finditer(text):
        out.append((_normalize_time(m.group(0)), m.start(), m.end()))
    for m in RE_EN_TIME.finditer(text):
        out.append((f"{int(m.group('h'))}시", m.start(), m.end()))
    for m in RE_EN_TIME_RANGE.finditer(text):
        out.append((f"{int(m.group('h1'))}시", m.start(), m.end()))
    return out

def _find_all_full_dates(text: str) -> List[str]:
    out = []
    for m in RE_FULLDATE_OPT_YEAR.finditer(text):
        out.append(_join_full_date(m))
    for m in RE_MD_DOTTED.finditer(text):
        y = m.group('y'); mth = m.group('m'); day = m.group('d'); dow = m.group('dow')
        core = f"{int(mth)}월 {int(day)}일"
        if dow: core += f" ({dow})"
        if y and len(y) == 4: out.append(f"{y}년 {core}")
        else: out.append(core)
    return _dedupe(out)

def _find_all_times(text: str) -> List[str]:
    out = []
    for m in RE_TIME.finditer(text): out.append(_normalize_time(m.group(0)))
    for m in RE_EN_TIME.finditer(text): out.append(f"{int(m.group('h'))}시")
    for m in RE_EN_TIME_RANGE.finditer(text): out.append(f"{int(m.group('h1'))}시")
    return _dedupe(out)

def _merge_year_with_partial_dates(tokens: List[str], existing_dates: List[str]) -> List[str]:
    merged = list(existing_dates)
    for i, tk in enumerate(tokens):
        if re.fullmatch(r'\d{4}년', tk):
            window = ' '.join(tokens[i+1:i+5])
            m = RE_FULLDATE_OPT_YEAR.search(f"{tk} {window}")
            if m: merged.append(_join_full_date(m))
    return _dedupe([_strip_space(x) for x in merged])

def _merge_date_with_weekday(dates: list[str]) -> list[str]:
    merged, skip = [], False
    for i, val in enumerate(dates):
        if skip: skip = False; continue
        nxt = dates[i+1] if i+1 < len(dates) else None
        if nxt and (re.fullmatch(r'^\(?[월화수목금토일]\)?$', nxt) or re.fullmatch(r'^\([월화수목금토일]\)$', nxt)):
            weekday = nxt if nxt.startswith("(") else f"({nxt})"
            merged.append(f"{val} {weekday}"); skip = True
        else:
            merged.append(val)
    return merged

def _pick_start_time_only(text: str, times: List[str]) -> List[str]:
    m = re.search(r'\bSTART\b[^\n]{0,20}?(\d{1,2}(?::\d{2})?)', text, flags=re.IGNORECASE)
    if m:
        val = m.group(1)
        return [val if ':' in val else f"{int(val)}시"]
    return times

def _normalize_dates(tokens: List[str], text: str, fields: Dict) -> None:
    """
    모델이 뽑은 DATE 후보에:
      1) 'YYYY년' + 부분 날짜 토큰을 결합
      2) 본문 전체에서 추가로 발견되는 (연도 포함/미포함) 날짜들을 합침
      3) strip + dedupe
    """
    dates = list(fields.get('DATE', []) or [])
    dates = _merge_year_with_partial_dates(tokens, dates)
    dates.extend(_find_all_full_dates(text))
    fields['DATE'] = _dedupe([_strip_space(d) for d in dates])

def _normalize_times(text: str, fields: Dict) -> None:
    """
    모델이 뽑은 TIME 후보에 본문 전체에서 찾은 시각들을 합쳐 정규화
    """
    times = list(fields.get('TIME', []) or [])
    times.extend(_find_all_times(text))
    fields['TIME'] = _dedupe([_strip_space(t) for t in times])