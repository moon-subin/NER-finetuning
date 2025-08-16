# 공통 유틸(토큰/정규화/컨텍스트/섹션감지)

# apply_regex_postrules 본체(오케스트레이션)
# -*- coding: utf-8 -*-
from typing import Dict, List, Optional
from .textutils import _to_ascii_compat
from .lineup import (
    _extract_lineup_and_handles_linewise_from, _pair_name_then_handle, _pairs_from_anywhere,
    _collect_lines_in_named_section, _collect_lines_in_lineup_sections,
    _collect_bullet_lines, _dj_block_names, _tidy_title, _fix_instagram, _harvest_names_from_lines,
    _extract_name_handle_runs
)
from .patterns import SECTION_SHOWCASE_HEAD, SECTION_LISTEN_HEAD, SECTION_DJ_HEAD
from .datetimex import _normalize_dates, _normalize_times, _merge_date_with_weekday, _pick_start_time_only
from .tickets import _ticket_open_mapping, _sanitize_ticket_open
from .price import _rescan_and_fix_prices
from .venue import _fix_venues
from .textutils import _strip_space, _dedupe

def apply_regex_postrules(
    text: str,
    tokens: List[str],
    fields: Dict[str, List[str]],
    lexicons: Optional[Dict]=None
) -> Dict[str, List[str]]:
    t = _to_ascii_compat(text)

    sc_lines = _collect_lines_in_named_section(t, SECTION_SHOWCASE_HEAD)
    ls_lines = _collect_lines_in_named_section(t, SECTION_LISTEN_HEAD)
    dj_lines = _collect_lines_in_named_section(t, SECTION_DJ_HEAD)
    sec_lines = _collect_lines_in_lineup_sections(t)

    allowed_blocks = ['\n'.join(sc_lines), '\n'.join(ls_lines), '\n'.join(sec_lines)]
    pairs_same_line = _extract_lineup_and_handles_linewise_from(allowed_blocks)
    pairs_two_lines = _pair_name_then_handle(t, allowed_blocks)
    pairs = _dedupe(pairs_same_line + pairs_two_lines)
    extra_pairs = _extract_name_handle_runs(text)  # 원문(text) 기준

    if not pairs:
        # 섹션이 없을 때도 '이름 ↵ @handle'을 잡아내기 위해 전역 탐색
        pairs = _pairs_from_anywhere(t)

    if pairs:
        fields.setdefault('LINEUP', []); fields.setdefault('INSTAGRAM', [])
        for name, handle in pairs:
            fields['LINEUP'].append(name); fields['INSTAGRAM'].append(handle)
        fields['LINEUP'] = _dedupe(fields['LINEUP'])
        fields['INSTAGRAM'] = _dedupe(fields['INSTAGRAM'])

    if extra_pairs:
        # 섹션 기반으로 잡힌 pairs와 합집합
        pairs = _dedupe((pairs or []) + extra_pairs)
    
    # 헤더 핸들 제거, 라인업 옆 핸들 유지
    from .lineup import _filter_instagram_handles
    _filter_instagram_handles(text, fields, pairs)

    # 날짜/시간/티켓오픈/가격
    _normalize_dates(tokens, t, fields)
    _normalize_times(t, fields)
    fields['TIME'] = _pick_start_time_only(t, fields.get('TIME', []))
    _ticket_open_mapping(t, fields)
    _rescan_and_fix_prices(t, tokens, fields)

    # 라인업 보강 (DJ 제외)
    def _harvest_excluding_dj():
        out=[]
        out += _harvest_names_from_lines(sc_lines)
        out += _harvest_names_from_lines(ls_lines)
        sec_names = _harvest_names_from_lines(sec_lines)
        out += sec_names
        from .lineup import _collect_bullet_lines
        bullet_all = _collect_bullet_lines(t)
        bullet_dj  = _collect_bullet_lines('\n'.join(dj_lines))
        bullet_lines = [b for b in bullet_all if b not in bullet_dj]
        out += _harvest_names_from_lines(bullet_lines)
        return out

    lineup = fields.get('LINEUP', []) or []
    lineup += _harvest_excluding_dj()

    # 사전 매칭 — DJ 제외 블록에만
    if lexicons and lexicons.get('artists'):
        safe_blocks = '\n'.join(allowed_blocks)
        import re
        for name in lexicons['artists']:
            if re.search(re.escape(name), safe_blocks, flags=re.IGNORECASE):
                lineup.append(name)

    dj_names = _dj_block_names(t)
    lineup = [ _strip_space(n) for n in lineup if _strip_space(n) and n not in dj_names ]
    fields['LINEUP'] = _dedupe(lineup)

    # TITLE 보강
    if not fields.get('TITLE'):
        from .lineup import _extract_title_from_head
        t_title = _extract_title_from_head(text)
        if t_title: fields['TITLE'] = [t_title]

    # VENUE/마무리
    _fix_instagram(fields)
    _fix_venues(t, fields, lexicons=lexicons)
    _sanitize_ticket_open(fields)
    _tidy_title(fields)

    if fields.get("DATE"):
        fields["DATE"] = _merge_date_with_weekday(fields["DATE"])
    return fields
