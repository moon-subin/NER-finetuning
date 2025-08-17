# /rules/postrules/lineup.py
# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Tuple
from .textutils import _strip_space, _dedupe

HANDLE_RE = re.compile(r'^@[A-Za-z0-9._]+$')
HANDLE_ANYWHERE_RE = re.compile(r'@[A-Za-z0-9._]+')
TIME_RE   = r'\d{1,2}\s*:\s*\d{2}'
TIMETABLE_LINE = re.compile(
    rf'^\s*(?:[•●◦▪️\-\–\—✦❏]\s*)?(?:{TIME_RE})(?:\s*-\s*{TIME_RE})?\s+(.+)$',
    re.MULTILINE
)
NAME_RE = re.compile(
    r'^\s*(?P<name>(?:DJ\s+)?[A-Za-z][A-Za-z0-9 .\'’\-\&\*/:+_]{1,60}|[가-힣0-9·]+(?:\s[가-힣0-9·]+){0,6})\s*$'
)
HASH_NAME_LINE = re.compile(r'^\s*#\s*(?P<tag>[A-Za-z0-9._\-]+|[가-힣0-9·]+)\s*$', re.UNICODE)
BI_NAME_LINE = re.compile(  # "English / 한글" 혹은 "English/한글"
    r'^\s*(?P<en>[A-Za-z0-9 .\'’\-&:_]{2,60})\s*/\s*[가-힣0-9 .\'’\-&:_]{1,60}\s*$'
)
BI_NAME_RE = re.compile(  # NEW: "English / 한글" or "English/한글"
    r'^\s*(?P<en>[A-Za-z0-9 .\'’\-&:_]{2,60})\s*/\s*[가-힣0-9 .\'’\-&:_]{1,60}\s*$'
)
LINEUP_BLACKLIST_SUBSTR = {
    '오픈','티켓','문의','입장','가격','현매','예매','공지','안내',
    '주최','주관','기획',
    '라이브홀','클럽','홀','극장','페스티벌','장소','일시','시간표','타임테이블'
}
BAD_GENERIC_HASHTAGS = {
    '공연','자선공연','정기','안내','정보','라이브','채널','구독','라이브앤라우드',
    '이벤트','예약','예매','현매','입장','무료','티켓'
}

# NEW: "출연: ..." / "라인업: ..." / "게스트: ..." 전역 패턴
#  - 라인 앞에 불릿(✅▪️•●◦-–—▶· 등)이 와도 허용
CAST_LINE = re.compile(
    r'^\s*(?:[✅▪️•●◦\-–—▶·]\s*)*(?:출연|라인업|게스트)\s*[:：]\s*(.+)$',
    re.IGNORECASE | re.MULTILINE
)

def _clean_artist_name(s: str) -> str:
    s = _strip_space(s)
    s = re.sub(r'^\d{1,2}\s*:\s*\d{2}(?:\s*-\s*\d{1,2}\s*:\s*\d{2})?\s*', '', s)
    s = re.sub(r'^(with|With)\b', '', s).strip()
    s = re.sub(r'[@\(].*$', '', s)
    s = re.sub(r'\s*\*\s*', '*', s)          # FRK * LOOP -> FRK*LOOP
    s = re.sub(r'\s*’\s*', '’', s)           # Hangman ’ s -> Hangman’s
    s = re.sub(r'\s*\'\s*', "'", s)
    s = re.sub(r'^[\-\–—•●◦▪️_:\|·]+', '', s) # 불릿/밑줄/콜론/중점 제거
    s = re.sub(r'[`_]+', '', s)
    return _strip_space(s)

def _looks_like_artist(s: str) -> bool:
    if not s or len(s) <= 1: return False
    if any(bad in s for bad in LINEUP_BLACKLIST_SUBSTR): return False
    if re.search(r'\d{1,2}\s*월|\d{1,2}\s*:\s*\d{2}|원\b', s): return False
    # 전부 대문자(3단어 이상)는 제목/슬로건일 확률↑ → DJ 접두만 예외
    if not s.lower().startswith('dj '):
        words = [w for w in re.split(r'\s+', s) if w]
        if len(words) >= 3 and all(w.isupper() for w in words if re.search(r'[A-Za-z]', w)):
            return False
    if s.count(' ') >= 8 and not s.lower().startswith('dj '): return False
    return True

def _names_from_hashtag_runs(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: List[str] = []
    run: List[str] = []

    def _flush(start_idx: int):
        nonlocal out, run
        if len(run) >= 3:
            # 1) 문서 끝 20% 안에 있는 해시태그 블록이면 무시
            if start_idx > int(len(lines) * 0.8):
                run = []
                return
            # 2) BAD_GENERIC_HASHTAGS가 대부분이면 무시
            bad_count = sum(1 for t in run if t in BAD_GENERIC_HASHTAGS)
            if bad_count >= len(run) * 0.6:  # 60% 이상이 generic 태그
                run = []
                return
            # 3) 정상 라인업 블록 처리
            for tag in run:
                tag = tag.replace('_', ' ')
                tag = _strip_space(tag)
                if not tag or tag in BAD_GENERIC_HASHTAGS:
                    continue
                if _looks_like_artist(tag):
                    out.append(tag)
        run = []

    for i, ln in enumerate(lines):
        m = HASH_NAME_LINE.match(ln)
        if m:
            run.append(m.group('tag'))
            run_start = i if len(run) == 1 else run_start
        else:
            if run:
                _flush(run_start)
    if run:
        _flush(run_start)
    return _dedupe(out)


def _extract_timeblock_pairs(text: str) -> List[Tuple[str, str]]:
    lines = [l for l in (x.rstrip() for x in text.splitlines())]
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        m = TIMETABLE_LINE.match(lines[i].strip())
        if not m:
            i += 1; continue
        name = _clean_artist_name(m.group(1))
        handle = None
        j = i + 1; hops = 0
        while j < len(lines) and hops < 12:
            ln = lines[j].strip()
            if TIMETABLE_LINE.match(ln): break
            if HANDLE_RE.fullmatch(ln): handle = ln; break
            j += 1; hops += 1
        if _looks_like_artist(name):
            pairs.append((name, handle if handle else None))
        i += 1
    # dedupe
    seen, out = set(), []
    for n, h in pairs:
        key = (n.lower(), (h or '').lower())
        if key in seen: continue
        seen.add(key); out.append((n, h))
    return out

def _pairs_from_same_line(text: str) -> List[Tuple[str, str]]:
    """한 줄에 이름과 핸들이 같이 있을 때: 핸들 앞 왼쪽 텍스트를 이름 후보로"""
    pairs: List[Tuple[str, str]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln: continue
        for m in HANDLE_ANYWHERE_RE.finditer(ln):
            left = _clean_artist_name(ln[:m.start()])
            left = re.sub(r'[-–—·•:|]\s*$', '', left).strip()
            if _looks_like_artist(left):
                pairs.append((left, m.group(0)))
    # dedupe
    seen, out = set(), []
    for n, h in pairs:
        key = (n.lower(), (h or '').lower())
        if key in seen: continue
        seen.add(key); out.append((n, h))
    return out

# NEW: with 블록 전용 파서
WITH_HEAD = re.compile(r'^\s*with\s*$', re.IGNORECASE)
NEXT_SECTION = re.compile(r'^\s*\[.*?\]\s*$')
def _pairs_from_with_block(text: str) -> List[Tuple[str, str]]:
    lines = [l.rstrip() for l in text.splitlines()]
    pairs: List[Tuple[str, str]] = []
    for i, ln in enumerate(lines):
        if not WITH_HEAD.match(ln): continue
        j = i + 1
        while j < len(lines):
            r = lines[j].strip()
            if r == "":
                j += 1; continue
            if NEXT_SECTION.match(r) or re.search(r'(공연\s*정보|일시|장소|티켓)', r):
                break
            m = HANDLE_ANYWHERE_RE.search(r)
            if m:
                name = _clean_artist_name(r[:m.start()])
                handle = m.group(0)
                if _looks_like_artist(name):
                    pairs.append((name, handle))
            j += 1
    # dedupe
    seen, out = set(), []
    for n, h in pairs:
        key = (n.lower(), h.lower())
        if key in seen: continue
        seen.add(key); out.append((n, h))
    return out

def extract_pairs_anywhere(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pairs += _pairs_from_with_block(text)
    pairs += _pairs_from_same_line(text)

    # NEW: 해시태그 블록(#이름 … #이름) → (이름, None) 쌍으로 수확
    for nm in _names_from_hashtag_runs(text):
        pairs.append((nm, None))

    # NEW: "출연: A with B, C ..." 형태 전역 수확 (핸들 없어도 이름만 라인업으로)
    for m in CAST_LINE.finditer(text):
        seg = m.group(1)
        # with/AND/&/그리고/및 → 구분자로 변환
        seg = re.sub(r'\bwith\b', ',', seg, flags=re.IGNORECASE)
        seg = re.sub(r'\b(and|&)\b', ',', seg, flags=re.IGNORECASE)
        seg = re.sub(r'\b(그리고|및)\b', ',', seg)
        # 콤마/슬래시/중점(·/U+00B7)/한글중점(·) 구분
        parts = re.split(r'[,\u00B7/·]+', seg)
        for p in parts:
            name = _clean_artist_name(p)
            if _looks_like_artist(name):
                pairs.append((name, None))

    # NEW: 바이링궐 라인을 전역에서 수확 (핸들 없어도 이름만 라인업으로)
    for ln in (x.strip() for x in text.splitlines()):
        if not ln or ln.startswith('#') or ln.startswith('@'): 
            continue
        # 문장성 안내문 배제
        if re.search(r'(Doors\s+open|FREE\s*entrance|admission|공연\s*시작)', ln, flags=re.IGNORECASE):
            continue
        m = BI_NAME_LINE.match(ln)
        if m:
            name = _clean_artist_name(m.group('en'))
            if _looks_like_artist(name):
                pairs.append((name, None))

    # 다음 줄 패턴: NAME ↵ @handle
    lines = [l for l in (x.strip() for x in text.splitlines()) if l != ""]
    for i in range(len(lines) - 1):
        if lines[i].startswith('@'): continue
        if HANDLE_RE.fullmatch(lines[i+1]):
            name = _clean_artist_name(lines[i])
            if _looks_like_artist(name):
                pairs.append((name, lines[i+1]))

    # 타임테이블
    pairs += _extract_timeblock_pairs(text)

    # dedupe
    seen, out = set(), []
    for n, h in pairs:
        key = (n.lower(), (h or '').lower())
        if key in seen: continue
        seen.add(key); out.append((n, h if h else None))
    return out

def harvest_lineup_names(text: str) -> List[str]:
    """
    보수적으로: with ~ [공연 정보] 구간 + 타임테이블에서만 라인업 수확
    "영문 / 한글" 이중 표기 라인도 영문을 우선 수확
    """
    names: List[str] = []
    # 타임테이블 먼저
    for m in TIMETABLE_LINE.finditer(text):
        nm = _clean_artist_name(m.group(1))
        if _looks_like_artist(nm): names.append(nm)
    # NEW: 해시태그 블록에서도 수확
    for nm in _names_from_hashtag_runs(text):
        if _looks_like_artist(nm):
            names.append(nm)
    # with 블록 추출
    lines = [ln.rstrip() for ln in text.splitlines()]
    collecting = False
    for ln in lines:
        raw = ln.strip()
        if re.fullmatch(r'with', raw, flags=re.IGNORECASE):
            collecting = True
            continue
        if collecting:
            if not raw:
                continue
            if re.fullmatch(r'\[.*?\]', raw) or re.search(r'(공연\s*정보|일시|장소|티켓)', raw):
                break
            if HANDLE_ANYWHERE_RE.search(raw):
                raw = re.sub(HANDLE_ANYWHERE_RE, '', raw)
            # NEW: bilingual "EN / KO" → EN 우선
            bm = BI_NAME_RE.match(raw)
            if bm:
                cand = _clean_artist_name(bm.group('en'))
                if _looks_like_artist(cand):
                    names.append(cand)
                continue
            cand = _clean_artist_name(raw)
            if _looks_like_artist(cand):
                names.append(cand)
    return _dedupe(names)

def collect_all_handles(text: str) -> List[str]:
    return _dedupe(HANDLE_ANYWHERE_RE.findall(text))

def tidy_title(fields: Dict) -> None:
    xs = []
    for t in fields.get('TITLE', []) or []:
        t = _strip_space(t)
        if not t: continue
        if t in {'달','공지','안내','정보','발표','NOTICE'}: continue
        if len(t) <= 1: continue
        xs.append(t)
    fields['TITLE'] = _dedupe(xs)
    # NEW: "… :" 로 끝난 항목과 바로 뒤 항목을 결합 (잘린 제목 복구)
    if len(xs) >= 2:
        merged = []
        i = 0
        while i < len(xs):
            cur = xs[i]
            nxt = xs[i+1] if i+1 < len(xs) else None
            if nxt and re.search(r'[:：]\s*$', cur):
                merged.append(_strip_space(f"{cur} {nxt}"))
                i += 2
            else:
                merged.append(cur)
                i += 1
        xs = merged
    fields['TITLE'] = _dedupe(xs)

def fix_instagram(fields: Dict) -> None:
    inst = list(fields.get('INSTAGRAM', []) or [])
    for x in fields.get('LINEUP', []) or []:
        if x and x.startswith('@') and x not in inst:
            inst.append(x)
    fields['INSTAGRAM'] = _dedupe(inst)

_tidy_title = tidy_title
_fix_instagram = fix_instagram
# 새로 추가: 공연장/주최 측 핸들을 분리(장소/프로필 링크 주변에 나타난 핸들은 V_INSTA로 분류)
def _filter_instagram_handles(text: str, handles: list[str]) -> tuple[list[str], list[str]]:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    venue_zone: set[str] = set()

    # (1) "장소:" / "Venue:" 라인 및 다음 2줄까지 스캔
    for i, ln in enumerate(lines):
        if re.search(r'(?:^|\s)(Venue|VENUE|장소)\s*[:：]', ln, flags=re.IGNORECASE):
            venue_zone.update(HANDLE_ANYWHERE_RE.findall(ln))
            for j in (i+1, i+2):
                if 0 <= j < len(lines):
                    venue_zone.update(HANDLE_ANYWHERE_RE.findall(lines[j]))

    # (2) "프로필 링크 참조" 같은 안내 라인에 나온 핸들도 장소 측으로 간주
    for ln in lines:
        if re.search(r'(프로필\s*링크|링크\s*참조)', ln):
            venue_zone.update(HANDLE_ANYWHERE_RE.findall(ln))

    venue_zone = {h for h in venue_zone if h in handles}
    # (3) '주최/주관/기획' 라인의 핸들은 통째로 버림
    drop: set[str] = set()
    for ln in lines:
        if re.search(r'(주최|주관|기획)\s*[:：]', ln):
            drop.update(HANDLE_ANYWHERE_RE.findall(ln))

    venue_zone = {h for h in venue_zone if h in handles and h not in drop}
    artist_handles = [h for h in handles if h not in venue_zone and h not in drop]
    venue_handles  = list(venue_zone)
    return artist_handles, venue_handles
