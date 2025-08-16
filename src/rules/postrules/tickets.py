# 티켓오픈 매핑/정화(_ticket_open_mapping 등)
# -*- coding: utf-8 -*-
import re
from typing import Dict
from .patterns import RE_TICKET_OPEN
from .datetimex import _find_dates_with_span, _find_times_with_span
from .textutils import _dedupe, _strip_space
from .patterns import RE_FULLDATE_OPT_YEAR

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

        best_d, best = 10**9, None
        for dtxt, ds, de in all_dates:
            dist = min(abs(anchor - ds), abs(anchor - de))
            if dist < best_d and dist <= 80:
                best_d, best = dist, dtxt
        if best: to_dates.append(best)

        best_d, best = 10**9, None
        for ttxt, ts, te in all_times:
            dist = min(abs(anchor - ts), abs(anchor - te))
            if dist < best_d and dist <= 80:
                best_d, best = dist, ttxt
        if best: to_times.append(best)

    fields['TICKET_OPEN_DATE'] = _dedupe(to_dates)
    fields['TICKET_OPEN_TIME'] = _dedupe(to_times)

def _is_valid_full_or_partial_date(s: str) -> bool:
    return bool(RE_FULLDATE_OPT_YEAR.fullmatch(s))

def _sanitize_ticket_open(fields: Dict) -> None:
    tod = []
    for d in fields.get('TICKET_OPEN_DATE', []) or []:
        d = _strip_space(d)
        if _is_valid_full_or_partial_date(d):
            tod.append(d)
    fields['TICKET_OPEN_DATE'] = _dedupe(tod)
