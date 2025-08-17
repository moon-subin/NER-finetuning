# /rules/postrules/datetimex.py
# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Tuple
from .patterns import (
    RE_FULLDATE_OPT_YEAR, RE_MD_DOTTED, RE_YMD_DOTTED,
    RE_EN_TIME, RE_EN_TIME_RANGE, RE_TIME, RE_EN_DATE
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
    out=[]
    for m in RE_FULLDATE_OPT_YEAR.finditer(text):
        out.append((_join_full_date(m), m.start(), m.end()))
    # NEW: English dates → "YYYY년 M월 D일" (연도 없으면 "M월 D일")
    mon_map = {
        'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
        'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
        'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12
    }
    for m in RE_EN_DATE.finditer(text):
        mm = mon_map.get(m.group('mon').lower())
        dd = int(m.group('day'))
        yy = m.group('year')
        core = f"{mm}월 {dd}일"
        out.append(f"{yy}년 {core}" if yy else core)
    for m in RE_YMD_DOTTED.finditer(text):
        y, mth, d, dow = m.group('y'), m.group('m'), m.group('d'), m.group('dow')
        core = f"{int(mth)}월 {int(d)}일"
        if dow: core += f" ({dow})"
        if y and len(y) == 4: txt = f"{y}년 {core}"
        else: txt = core
        out.append((txt, m.start(), m.end()))
    return out

def _find_times_with_span(text: str):
    out=[]
    for m in RE_TIME.finditer(text):
        out.append((_normalize_time(m.group(0)), m.start(), m.end()))
    for m in RE_EN_TIME.finditer(text):
        out.append((f"{int(m.group('h'))}시", m.start(), m.end()))
    for m in RE_EN_TIME_RANGE.finditer(text):
        out.append((f"{int(m.group('h1'))}시", m.start(), m.end()))
    return out

def _find_all_full_dates(text: str) -> List[str]:
    out=[]
    ymd_spans=[]
    for m in RE_YMD_DOTTED.finditer(text):
        y, mth, d, dow = m.group('y'), m.group('m'), m.group('d'), m.group('dow')
        core = f"{int(mth)}월 {int(d)}일"
        if dow: core += f" ({dow})"
        if y and len(y) == 4: out.append(f"{y}년 {core}")
        else: out.append(core)
        ymd_spans.append((m.start(), m.end()))

    def _overlaps_ymd(a,b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    for m in RE_FULLDATE_OPT_YEAR.finditer(text):
        out.append(_join_full_date(m))

    for m in RE_MD_DOTTED.finditer(text):
        s,e = m.start(), m.end()
        if any(_overlaps_ymd((s,e), span) for span in ymd_spans):
            continue  # 25.08.29 안에서 25.08 같은 부분 매치 제거
        mth = int(m.group('m')); day = int(m.group('d'))
        if not (1 <= mth <= 12 and 1 <= day <= 31):
            continue
        y = m.group('y'); dow = m.group('dow')
        core = f"{mth}월 {day}일"
        if dow: core += f" ({dow})"
        if y and len(y) == 4: out.append(f"{y}년 {core}")
        else: out.append(core)

    return _dedupe(out)

def _merge_year_with_partial_dates(tokens: List[str], existing_dates: List[str]) -> List[str]:
    # 현재 샘플에서 역효과 가능 → 그대로 두되 기본 동작만 유지
    return _dedupe([_strip_space(x) for x in existing_dates])

def _merge_date_with_weekday(dates: list[str]) -> list[str]:
    merged, skip = [], False
    for i, val in enumerate(dates):
        if skip: skip=False; continue
        nxt = dates[i+1] if i+1 < len(dates) else None
        if nxt and re.fullmatch(r'^\(?[월화수목금토일]\)?$', nxt):
            weekday = nxt if nxt.startswith("(") else f"({nxt})"
            merged.append(f"{val} {weekday}"); skip=True
        else:
            merged.append(val)
    return merged

def _pick_start_time_only(text: str, times: List[str]) -> List[str]:
    # START / begin / 공연시작 등에서 첫 시작시각을 우선
    m = re.search(r'(START|begin|공연\s*시작)[^\n]{0,30}?(\d{1,2}(?::\d{2})?)', text, flags=re.IGNORECASE)
    if m:
        val = m.group(2)
        return [val if ':' in val else f"{int(val)}시"]
    return times

def _normalize_dates(tokens: List[str], text: str, fields: Dict) -> None:
    dates = list(fields.get('DATE', []) or [])
    dates.extend(_find_all_full_dates(text))
    fields['DATE'] = _dedupe([_strip_space(d) for d in dates])

def _normalize_times(text: str, fields: Dict) -> None:
    times = list(fields.get('TIME', []) or [])
    for m in RE_TIME.finditer(text):
        times.append(_normalize_time(m.group(0)))
    for m in RE_EN_TIME.finditer(text): times.append(f"{int(m.group('h'))}시")
    for m in RE_EN_TIME_RANGE.finditer(text): times.append(f"{int(m.group('h1'))}시")
    fields['TIME'] = _dedupe([_strip_space(t) for t in times])
