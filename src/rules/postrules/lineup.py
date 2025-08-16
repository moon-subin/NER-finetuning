# 라인업/인스타 핸들 수확/정리
# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Optional, Set, Tuple
from .patterns import (
    LINEUP_SEC_HEAD, BULLET_LINE, NAME_CAND, SECTION_BREAK,
    SECTION_SHOWCASE_HEAD, SECTION_LISTEN_HEAD, SECTION_DJ_HEAD,
    LINEUP_BLACKLIST_SUBSTR, POSTFIX_JOSA, TITLE_STOP
)
from .textutils import _strip_space, _dedupe
from .textutils import _prefer_longer_unique  # (혹시 타이틀 정리에서 사용)
# === 이름/핸들 수확 ===

def _clean_artist_name(s: str) -> str:
    s = _strip_space(s)
    s = re.sub(r'^[\-\–—·\•\▶\✅\[\(]+', '', s)
    s = re.sub(r'[@\(].*$', '', s)
    s = re.sub(r'[\*\_`]+', '', s)
    s = POSTFIX_JOSA.sub('', s)
    return _strip_space(s)

def _looks_like_artist(s: str) -> bool:
    if not s or len(s) <= 1: return False
    if any(bad in s for bad in LINEUP_BLACKLIST_SUBSTR): return False
    if re.search(r'\d{1,2}\s*월|\d{1,2}\s*:\s*\d{2}|원\b', s): return False
    if s.count(' ') >= 8 and not s.lower().startswith('dj '): return False
    return True

def _harvest_names_from_lines(lines: list) -> list:
    out = []
    for ln in lines:
        ln = _strip_space(ln)
        if not ln: continue
        m = NAME_CAND.match(ln)
        if not m: continue
        name = _clean_artist_name(m.group('name'))
        if _looks_like_artist(name): out.append(name)
    return out

def _collect_lines_in_lineup_sections(text: str) -> list:
    lines = text.splitlines(); take = False
    buf, collected = [], []
    for ln in lines:
        if LINEUP_SEC_HEAD.search(ln):
            if buf: collected.extend(buf); buf=[]
            take = True; continue
        if take:
            if SECTION_BREAK.search(ln):
                take = False
                if buf: collected.extend(buf); buf=[]
                continue
            buf.append(ln)
    if buf: collected.extend(buf)
    return collected

def _collect_bullet_lines(text: str) -> list:
    return [m.group(1) for m in BULLET_LINE.finditer(text)]

def _collect_lines_in_named_section(text: str, head_re: re.Pattern) -> list:
    lines = text.splitlines(); take = False
    buf, out = [], []
    for ln in lines:
        if head_re.search(ln):
            if buf: out.extend(buf); buf=[]
            take = True; continue
        if take:
            if (SECTION_BREAK.search(ln) or
                LINEUP_SEC_HEAD.search(ln) or
                SECTION_SHOWCASE_HEAD.search(ln) or
                SECTION_LISTEN_HEAD.search(ln) or
                SECTION_DJ_HEAD.search(ln)):
                take = False
                if buf: out.extend(buf); buf=[]
                continue
            buf.append(ln)
    if buf: out.extend(buf)
    return out

def _extract_lineup_and_handles_linewise_from(blocks: List[str]) -> List[tuple]:
    pairs=[]; pat=re.compile(r'^(?P<name>[^@#\|\[\]\(\)]+)\s+(@[A-Za-z0-9._]+)\s*$')
    for block in blocks:
        for ln in block.splitlines():
            ln = _strip_space(ln)
            if not ln: continue
            m = pat.match(ln)
            if not m: continue
            name = _clean_artist_name(m.group('name'))
            handle = m.group(2)
            if _looks_like_artist(name): pairs.append((name, handle))
    return pairs

def _pair_name_then_handle(text: str, blocks: List[str]) -> List[tuple]:
    pairs=[]; name_pat=re.compile(r'^(?P<name>[^@#\|\[\]\(\)]+?)\s*$'); handle_pat=re.compile(r'^(@[A-Za-z0-9._]+)\s*$')
    for block in blocks:
        lines=[_strip_space(x) for x in block.splitlines()]
        for i, ln in enumerate(lines[:-1]):
            m1=name_pat.match(ln); m2=handle_pat.match(lines[i+1])
            if not (m1 and m2): continue
            name=_clean_artist_name(m1.group('name')); handle=m2.group(1)
            if _looks_like_artist(name): pairs.append((name, handle))
    return pairs

def _filter_instagram_handles(text: str, fields: Dict, pairs: List[tuple]) -> None:
    if pairs:
        keep = {h for _, h in pairs}
        fields['INSTAGRAM'] = [h for h in fields.get('INSTAGRAM', []) if h in keep]
        return
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header = "\n".join(lines[:2])
    header_handles = set(re.findall(r'@[A-Za-z0-9._]+', header))
    fields['INSTAGRAM'] = [h for h in fields.get('INSTAGRAM', []) if h not in header_handles]

def _dj_block_names(text_norm: str) -> set:
    dj_lines = _collect_lines_in_named_section(text_norm, SECTION_DJ_HEAD)
    names = set(_harvest_names_from_lines(dj_lines))
    bullet_dj = _collect_bullet_lines("\n".join(dj_lines))
    names |= set(_harvest_names_from_lines(bullet_dj))
    return names

def _tidy_title(fields: Dict) -> None:
    xs=[]
    for t in fields.get('TITLE', []) or []:
        t=_strip_space(t)
        if not t: continue
        if t in TITLE_STOP: continue
        if len(t) <= 1: continue
        xs.append(t)
    fields['TITLE'] = _dedupe(xs)

def _fix_instagram(fields: Dict) -> None:
    inst=list(fields.get('INSTAGRAM', []) or [])
    for x in fields.get('LINEUP', []):
        if x.startswith('@'): inst.append(x)
    fields['INSTAGRAM'] = _dedupe(inst)

def _cut_named_sections(text: str, head_res: list[re.Pattern]) -> str:
    """특정 섹션(예: DJ) 본문을 text에서 제거해 안전 블록을 만든다."""
    lines = text.splitlines()
    out, take, buf = [], True, []
    i = 0
    while i < len(lines):
        ln = lines[i]
        # 섹션 헤더 시작?
        if any(hr.search(ln) for hr in head_res):
            # 해당 섹션 끝(SECTION_BREAK 또는 다음 섹션 헤더류)까지 스킵
            i += 1
            while i < len(lines):
                ln2 = lines[i]
                if (SECTION_BREAK.search(ln2) or
                    LINEUP_SEC_HEAD.search(ln2) or
                    SECTION_SHOWCASE_HEAD.search(ln2) or
                    SECTION_LISTEN_HEAD.search(ln2) or
                    SECTION_DJ_HEAD.search(ln2)):
                    break
                i += 1
            # 현재 i는 섹션 경계(또는 종료) 지점
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)

def _pairs_from_anywhere(text: str) -> list[tuple]:
    """
    DJ 섹션을 제외한 전체 텍스트에서
    1) '이름 @handle' 한 줄 패턴
    2) '이름' ↵ '@handle' 두 줄 패턴
    을 모두 수확.
    """
    safe = _cut_named_sections(text, [SECTION_DJ_HEAD])
    blocks = [safe]
    pairs1 = _extract_lineup_and_handles_linewise_from(blocks)
    pairs2 = _pair_name_then_handle(safe, blocks)
    return _dedupe(pairs1 + pairs2)

def _extract_name_handle_runs(text: str) -> list[tuple]:
    """
    전역 텍스트에서 '이름' 다음 줄이 '@handle'인 패턴을 쭉 긁어옵니다.
    - 공백 줄은 건너뜀
    - DJ 섹션은 무시 (오탐 방지)
    """
    # DJ 섹션 제거(대충이라도 잘라내면 오탐 크게 줄어듭니다)
    lines = text.splitlines()
    cleaned = []
    skip = False
    for ln in lines:
        if SECTION_DJ_HEAD.search(ln):
            skip = True
            continue
        if skip:
            # 다음 섹션 헤더/구분선 나오기 전까지 스킵
            if re.match(r'^\s*(\[.*?\]|[-=]{3,}|공연\s*정보|일시|장소|티켓|입장|문의|PRICE|TIME|DATE)\s*[:|]?', ln):
                skip = False
            continue
        cleaned.append(ln)

    lines = [l for l in (x.strip() for x in cleaned) if l != ""]
    pairs = []
    handle_pat = re.compile(r'^@[A-Za-z0-9._]+$')
    for i in range(len(lines) - 1):
        name = lines[i]
        nxt  = lines[i+1]
        if name.startswith('@'):
            continue
        if handle_pat.match(nxt):
            # 간단 정리: 괄호/핸들 제거, 양끝 공백 정리
            name_clean = re.sub(r'[@\(].*$', '', name).strip()
            # 너무 짧으면 제외
            if len(name_clean) <= 1: 
                continue
            pairs.append((name_clean, nxt))
    return _dedupe(pairs)