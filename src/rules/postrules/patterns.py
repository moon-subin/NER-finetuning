# 정규식/상수 (RE_*, 키워드/블랙리스트 등)
# -*- coding: utf-8 -*-
import re

RE_YEAR = r'(?P<year>\d{4})년'
RE_MD   = r'(?P<month>\d{1,2})\s*월\s*(?P<day>\d{1,2})\s*일'
RE_DOW  = r'(?:\s*\((?P<dow>[월화수목금토일]|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\))?'

RE_FULLDATE_OPT_YEAR  = re.compile(rf'(?:{RE_YEAR}\s*)?{RE_MD}\s*{RE_DOW}')
RE_FULLDATE_MUST_YEAR = re.compile(rf'{RE_YEAR}\s*{RE_MD}\s*{RE_DOW}')

RE_TIME = re.compile(
    r'((?P<ampm>오전|오후|낮|밤|저녁)\s*(?P<hour1>\d{1,2})(?:\s*:\s*(?P<min1>\d{2}))?\s*시?)'
    r'|((?P<hour2>\d{1,2})(?:\s*:\s*(?P<min2>\d{2}))\s*시?)'
    r'|((?P<hour3>\d{1,2})\s*시)'
)
RE_EN_TIME        = re.compile(r'\b(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>AM|PM|am|pm)\b')
RE_EN_TIME_RANGE  = re.compile(r'\b(?P<h1>\d{1,2})(?::(?P<m1>\d{2}))?-(?P<h2>\d{1,2})(?::(?P<m2>\d{2}))?\s*(?P<ap>AM|PM|am|pm)\b')

RE_MD_DOTTED = re.compile(
    r'\b(?P<m>\d{1,2})[.\-/](?P<d>\d{1,2})(?:[.\-/](?P<y>\d{2,4}))?(?:\s*\.\s*)?(?:\s*\((?P<dow>[A-Za-z]{3})\))?\b'
)

RE_PRICE   = re.compile(r'(?:(?:₩|￦)\s*)?(?P<amount>\d{1,3}(?:,\d{3})+|\d{4,6}|\d+)\s*(?:원|₩|￦|KRW)?', re.IGNORECASE)
RE_KR_PRICE= re.compile(r'(?:(?P<man>\d+)\s*만)?\s*(?:(?P<chon>\d+)\s*천)?\s*(?P<num>\d{1,3}(?:,\d{3})+|\d+)?\s*(?:원|₩|￦|KRW)?', re.IGNORECASE)

RE_TICKET_OPEN = re.compile(r'티\s*켓\s*(?:은|:)?[^\n]{0,20}?오\s*픈', re.IGNORECASE)

VENUE_KEYWORDS = r'(홀|클럽|라이브|센터|스테이지|스튜디오|씨어터|극장|하우스|라운지|스퀘어|플랫폼|플레이스|페스티벌|레코드|창고|공간|바|펍|플라자|파크|웨이브|베뉴|살롱)'
VENUE_BLACKLIST_WORDS = {'도어', '티케팅', '티켓팅', '입장', '현매', '예약', '예매'}

LINEUP_STOPWORDS = {'티켓','오픈','출연','공연','콘서트','쇼케이스','단독','공지','예매','현매','현장','가격'}
TITLE_STOP = {'달','공지','안내','정보','발표'}

POSTFIX_JOSA = re.compile(r'(와|과|이|가|는|은|도|를|을|과의|와의)$')

LINEUP_SEC_HEAD = re.compile(
    r'^\s*(?:\[?\s*(?:LINE\s*[- ]?\s*UP|라인업|ARTIST|Artist(?:\s*information)?|Artist\s*Info)\s*\]?|\-+\s*LINE\s*UP\s*\-+)\s*$',
    re.IGNORECASE | re.MULTILINE
)
BULLET_LINE = re.compile(r'^\s*[✅▪️•●◦\-–—▶]\s*(.+)$', re.MULTILINE)
NAME_CAND   = re.compile(
    r'^\s*(?P<name>(?:DJ\s+)?[A-Za-z][A-Za-z0-9 .\'\-\&\?\/!·:_]{1,60}|[가-힣0-9·]+(?:\s[가-힣0-9·]+){0,6})'
    r'(?:\s*\([^)]+\))?\s*(?:@[A-Za-z0-9._]+)?\s*$'
)
SECTION_BREAK = re.compile(
    r'^\s*(?:\[.*?\]|공연\s*정보|일시|장소|티켓|입장|문의|PRICE|TIME|DATE)\s*[:|]|^[-=]{3,}\s*$',
    re.MULTILINE
)
LINEUP_BLACKLIST_SUBSTR = {
    '오픈', '티켓', '문의', '입장', '가격', '현매', '예매', '공지', '안내',
    '라이브홀', '클럽', '홀', '극장', '페스티벌', '월드뮤직', '록 페스티벌',
    '노들섬', '스트레인지', '프룻', '장소', '일시'
}

SECTION_SHOWCASE_HEAD = re.compile(r'^\s*\[\s*(?:Showcase|쇼케이스)\s*(?:\|\s*(?:Showcase|쇼케이스))?\s*\]\s*$', re.IGNORECASE | re.MULTILINE)
SECTION_LISTEN_HEAD   = re.compile(r'^\s*\[\s*(?:Listening\s*Session|음감회)\s*(?:\|\s*(?:Listening\s*Session|음감회))?\s*\]\s*$', re.IGNORECASE | re.MULTILINE)
SECTION_DJ_HEAD       = re.compile(r'^\s*\[\s*(?:DJ|디제이)\s*(?:\|\s*(?:DJ|디제이))?\s*\]\s*$', re.IGNORECASE | re.MULTILINE)
