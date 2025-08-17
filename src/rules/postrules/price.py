# /rules/postrules/price.py
# 가격 스캔/정규화(_rescan_and_fix_prices)
# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Optional
from .textutils import _strip_space, _dedupe, _parse_kr_price, _normalize_price, _price_to_int

def _rescan_and_fix_prices(text: str, tokens: List[str], fields: Dict) -> None:
    # 통일
    text = (text or '').replace('₩','원').replace('￦','원')
    price = list(fields.get('PRICE', []) or [])
    onsite = list(fields.get('PRICE_ONSITE', []) or [])
    ptype = list(fields.get('PRICE_TYPE', []) or [])

    # 1) 토큰 기반: '예매' 바로 뒤 첫 금액 / '현매|현장' 뒤 첫 금액
    def tok_span_to_str(i, j): return _strip_space(''.join(tokens[i:j]))

    for i, tk in enumerate(tokens):
        if tk == '예매':
            # "예매 22,000원" 근처 첫 금액
            for j in range(i+1, min(i+8, len(tokens))):
                s = tok_span_to_str(i+1, j+1)
                m = re.search(r'(\d{1,3}(?:,\d{3})+|\d+)\s*(?:원|KRW)?', s, re.IGNORECASE)
                if m:
                    price.append(_normalize_price(m.group(0)))
                    break
        if tk in ('현매','현장'):
            for j in range(i+1, min(i+8, len(tokens))):
                s = tok_span_to_str(i+1, j+1)
                m = re.search(r'(\d{1,3}(?:,\d{3})+|\d+)\s*(?:원|KRW)?', s, re.IGNORECASE)
                if m:
                    onsite.append(_normalize_price(m.group(0)))
                    break

    # 2) 텍스트 라인 기반: "예매 … 22,000" / "(현장|현매)( 예매)? … 25,000"
    for m in re.finditer(r'예매[^\n]{0,30}?(\d{1,3}(?:,\d{3})+|\d+)\s*(?:원|KRW)?', text):
        price.append(_normalize_price(m.group(1)))
    for m in re.finditer(r'(현매|현장)(?:\s*예매)?[^\n]{0,30}?(\d{1,3}(?:,\d{3})+|\d+)\s*(?:원|KRW)?', text):
        onsite.append(_normalize_price(m.group(2)))

    # 2-1) 무료 입장 시 바로 시드
    if re.search(r'(무료\s*입장|무료입장|무료|free\s*(entrance|admission)?)', text, flags=re.IGNORECASE):
        price.append("무료")

    # 3) PRICE_TYPE 보정: '현장 예매' 합성, '현매' → '현장 예매'
    if re.search(r'현장\s*예매', text):
        ptype.append('현장 예매')
    ptype = [('현장 예매' if x in ('현매','현장예매','현장예매') else x) for x in ptype]
    # '현장' 단독은 '현장 예매'가 있으면 제거
    if '현장 예매' in ptype:
        ptype = [x for x in ptype if x != '현장']
    fields['PRICE_TYPE'] = _dedupe([_strip_space(x) for x in ptype if _strip_space(x)])

    # 4) 정리: 숫자만 보존 + 중복 제거
    def clean_prices(xs):
        out = []
        for x in xs:
            x = _strip_space(x)
            if not x: continue
            # 무료 케이스 바로 매핑
            if re.search(r'(무료\s*입장|무료입장|무료|free\s*(entrance|admission)?)', x, flags=re.IGNORECASE):
                out.append("무료");  # 한글 표준화
                continue
            # 숫자만 남기기
            m = re.search(r'(\d{1,3}(?:,\d{3})+|\d+)', x)
            if not m: continue
            num = m.group(1)
            # (a) '번호/순번/석/명' 같은 맥락의 숫자는 무시
            if re.search(r'(번호|순번?|석|명)(?=[^\d]|$)', x):
                continue
            # (b) 4자리 미만(쉼표 없음) 단독 숫자는 가격으로 보지 않음 (ex: '25')
            if ',' not in num and len(num) < 4:
                continue
            out.append(num)        
        return _dedupe(out)

    price  = clean_prices(price)
    onsite = clean_prices(onsite)

    # 5) 예매/현장에 중복 숫자 있으면, 현장쪽 숫자는 PRICE에서 제거
    if onsite:
        price = [p for p in price if p not in onsite]

    fields['PRICE'] = price
    fields['PRICE_ONSITE'] = onsite
