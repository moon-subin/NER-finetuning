# /rules/postrules/tickets.py
# -*- coding: utf-8 -*-
import re
from typing import Dict
from .textutils import _strip_space, _dedupe

_ANCHOR = re.compile(
    r'^\s*(?:[•●◦\-\–\—✦❏]\s*)*'
    r'(?:(?:티\s*켓|예매)\s*(?:은|:)?[^\n]{0,20}?오\s*픈|Ticket\s*Open)'
    r'\s*[:：]?\s*(?P<rest>.*)$',
    re.IGNORECASE | re.MULTILINE
)
_DATE = re.compile(
    r'(?P<date>(?:\d{2,4}\s*\.\s*)?\d{1,2}\s*\.\s*\d{1,2}\.?\s*(?:\([^)]+\))?|\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일(?:\s*\([^)]+\))?)'
)
_TIME = re.compile(r'(?P<time>(?:오전|오후)\s*\d{1,2}\s*시|\b\d{1,2}:\d{2}\b)')

def _ticket_open_mapping(text: str, fields: Dict) -> None:
    dates, times = [], []
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        m = _ANCHOR.match(raw)
        if not m: continue
        tail = m.group('rest') or ""
        dm = _DATE.search(tail)
        tm = _TIME.search(tail)
        j = idx + 1
        while (dm is None or tm is None) and j < len(lines):
            nxt = lines[j].strip()
            if nxt == "": j += 1; continue
            if dm is None: dm = _DATE.search(nxt)
            if tm is None: tm = _TIME.search(nxt)
            break
        if dm: dates.append(_strip_space(dm.group('date')))
        if tm: times.append(_strip_space(tm.group('time')))
    fields['TICKET_OPEN_DATE'] = _dedupe(dates)
    fields['TICKET_OPEN_TIME'] = _dedupe(times)

def _sanitize_ticket_open(fields: Dict) -> None:
    def _normalize_dotted_ymd(s: str) -> str:
        """
        '25.08.19(화)' → '8월 19일 (화)' 형식으로 정규화(연도는 생략).
        """
        if not s: return s
        s0 = _strip_space(s)
        m = re.search(r'(?P<y>\d{2,4})\s*\.\s*(?P<m>\d{1,2})\s*\.\s*(?P<d>\d{1,2})\.?\s*(?:\((?P<dow>[^)]+)\))?', s0)
        if not m:
            # '2025년 8월 19일 (화)' 같은 케이스는 그대로 둠
            return s0
        mth = int(m.group('m')); day = int(m.group('d'))
        dow = m.group('dow')
        core = f"{mth}월 {day}일"
        return f"{core} ({dow})" if dow else core

    dates = [_normalize_dotted_ymd(x) for x in (fields.get('TICKET_OPEN_DATE', []) or [])]
    fields['TICKET_OPEN_DATE'] = _dedupe([_strip_space(x) for x in dates if _strip_space(x)])
