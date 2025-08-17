# /rules/postrules/tickets.py
# -*- coding: utf-8 -*-
import re
from typing import Dict
from .textutils import _strip_space, _dedupe

# 머리부(불릿/이모지) 허용 폭 확대 + 앵커 키워드 확장
_ANCHOR = re.compile(
    r'^\s*(?:[•●◦▪️\-\–\—✦❏✅✔▶🎫🗓️🕰️⏱️]\s*)*'
    r'(?:'
    r'티\s*켓\s*오\s*픈\s*[:：]?'                            # 티켓오픈 / 티켓 오픈 / 티켓오픈:
    r'|티\s*켓\s*(?:은|:)?[^\n]{0,20}?오\s*픈'              
    r'|Ticket\s*Open\s*[:：]?'
    r'|예\s*매\s*오\s*픈\s*[:：]?'
    r'|티\s*켓\s*판\s*매\s*(?:일\s*시|시\s*작)\s*[:：]?'
    r'|티\s*켓\s*오\s*픈\s*(?:일|일\s*시)?\s*[:：]?'
    r'|티켓판매일시\s*[:：]?'
    r'|tickets?\s*(?:on\s*sale|sale\s*start|sales?\s*open)\s*[:：]?'
    r')'
    r'\s*(?P<rest>.*)$',
    re.IGNORECASE | re.MULTILINE
)

# 날짜/시간 추출 패턴
_DATE = re.compile(
    r'(?P<date>('
    # 1) Y.M.D / Y-M-D / Y/M/D (요일 괄호 optional)
    r'\d{2,4}\s*[.\-/]\s*\d{1,2}\s*[.\-/]\s*\d{1,2}\.?\s*(?:\([^)]+\))?'
    r'|'
    # 2) M.D / M-D / M/D (연도 없음, 요일 괄호 optional)  ← 여기서 8/18(월) 잡힘
    r'\d{1,2}\s*[.\-/]\s*\d{1,2}\.?\s*(?:\([^)]+\))?'
    r'|'
    # 3) 2025년 8월 18일 (요일 optional)
    r'\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일(?:\s*\([^)]+\))?'
    r'))'
)

# ‘17시’ 같은 순시 표기, ‘7PM’도 인식
_TIME = re.compile(
    r'(?P<time>('
    r'(?:오전|오후)\s*\d{1,2}\s*시'     # 오전 8시 / 오후 8시
    r'|\b\d{1,2}:\d{2}\b'              # 20:00
    r'|\b\d{1,2}\s*시\b'               # 17시
    r'|\b\d{1,2}\s*(?:AM|PM|am|pm)\b'  # 7PM / 8am
    r'))'
)

def _ticket_open_mapping(text: str, fields: Dict) -> None:
    dates, times = [], []
    lines = text.splitlines()
    for idx, raw in enumerate(lines):
        m = _ANCHOR.match(raw)
        if not m:
            continue
        tail = m.group('rest') or ""

        # 동일 라인에서 먼저 찾기
        dm = _DATE.search(tail)
        tm = _TIME.search(tail)

        # 없다면 다음 ‘의미 있는’ 한 줄까지 살펴보기
        j = idx + 1
        while (dm is None or tm is None) and j < len(lines):
            nxt = lines[j].strip()
            if nxt == "":
                j += 1
                continue
            if dm is None:
                dm = _DATE.search(nxt)
            if tm is None:
                tm = _TIME.search(nxt)
            break

        if dm:
            dates.append(_strip_space(dm.group('date')))
        if tm:
            times.append(_strip_space(tm.group('time')))

    fields['TICKET_OPEN_DATE'] = _dedupe(dates)
    fields['TICKET_OPEN_TIME'] = _dedupe(times)

def _sanitize_ticket_open(fields: Dict) -> None:
    fields['TICKET_OPEN_DATE'] = _dedupe(
        [_strip_space(x) for x in fields.get('TICKET_OPEN_DATE', []) or []]
    )
