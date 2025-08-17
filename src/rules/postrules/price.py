# /rules/postrules/price.py
# 가격 스캔/정규화(_rescan_and_fix_prices)
# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Optional
from .textutils import _strip_space, _dedupe, _parse_kr_price, _normalize_price, _price_to_int

def _rescan_and_fix_prices(text: str, tokens: List[str], fields: Dict) -> None:
    """
    텍스트/토큰을 다시 스캔하여 가격 정보를 보강/정규화합니다.
    - 예매/현매(현장) 구분
    - '만/천' 단위 및 소수점(예: 2.5만) 처리
    - '판매가격', '가격' 라인 처리
    - 'Only door(현매로만)'일 때 PRICE_ONSITE만 남김
    """

    # 통일: 화폐기호를 '원'으로 정규화(내부 처리 일관성)
    text = (text or '').replace('₩', '원').replace('￦', '원')

    price  = list(fields.get('PRICE', []) or [])
    onsite = list(fields.get('PRICE_ONSITE', []) or [])
    ptype  = list(fields.get('PRICE_TYPE', []) or [])

    # -------------------------------
    # 0) Only door(현매로만/현장판매만 등) 감지
    # -------------------------------
    ONLY_DOOR = re.search(
        r'(only\s*door|현매\s*로\s*만|현장\s*판매\s*만|현장\s*구매\s*만)',
        text,
        flags=re.IGNORECASE
    ) is not None

    # -------------------------------
    # 1) 토큰 기반: '예매' 뒤 첫 금액 / '현매|현장' 뒤 첫 금액
    # -------------------------------
    def tok_span_to_str(i, j): 
        return _strip_space(''.join(tokens[i:j]))

    for i, tk in enumerate(tokens):
        if tk == '예매':
            # "예매 22,000원" 근처 첫 금액
            for j in range(i+1, min(i+8, len(tokens))):
                s = tok_span_to_str(i+1, j+1)
                m = re.search(r'(\d{1,3}(?:,\d{3})+|\d+)\s*(?:원|KRW)?', s, re.IGNORECASE)
                if m:
                    price.append(_normalize_price(m.group(1)))
                    break
        if tk in ('현매','현장'):
            for j in range(i+1, min(i+8, len(tokens))):
                s = tok_span_to_str(i+1, j+1)
                m = re.search(r'(\d{1,3}(?:,\d{3})+|\d+)\s*(?:원|KRW)?', s, re.IGNORECASE)
                if m:
                    onsite.append(_normalize_price(m.group(1)))
                    break

    # -------------------------------
    # 2) 텍스트 라인 기반
    #    2-1) 숫자형: "예매 … 22,000" / "(현장|현매)( 예매)? … 25,000"
    # -------------------------------
    for m in re.finditer(r'예매[^\n]{0,30}?(\d{1,3}(?:,\d{3})+|\d+)\s*(?:원|KRW)?', text):
        price.append(_normalize_price(m.group(1)))
    for m in re.finditer(r'(현매|현장)(?:\s*예매)?[^\n]{0,30}?(\d{1,3}(?:,\d{3})+|\d+)\s*(?:원|KRW)?', text):
        onsite.append(_normalize_price(m.group(2)))

    # -------------------------------
    # 2-2) 단위형(만/천): "예매 2.5만 / 현매 3만" / "예매 5천"
    # -------------------------------
    # 예매 → PRICE
    for m in re.finditer(r'예매[^\n]{0,20}?(\d+(?:\.\d+)?)\s*만', text):
        val = float(m.group(1)) * 10000
        price.append(_normalize_price(str(int(val))))
    for m in re.finditer(r'예매[^\n]{0,20}?(\d+(?:\.\d+)?)\s*천', text):
        val = float(m.group(1)) * 1000
        price.append(_normalize_price(str(int(val))))

    # 현매/현장 → PRICE_ONSITE
    for m in re.finditer(r'(현매|현장)[^\n]{0,20}?(\d+(?:\.\d+)?)\s*만', text):
        val = float(m.group(2)) * 10000
        onsite.append(_normalize_price(str(int(val))))
    for m in re.finditer(r'(현매|현장)[^\n]{0,20}?(\d+(?:\.\d+)?)\s*천', text):
        val = float(m.group(2)) * 1000
        onsite.append(_normalize_price(str(int(val))))

    # -------------------------------
    # 2-3) 판매가격/가격: "판매가격 : 11,000원" / "가격: 11,000"
    # -------------------------------
    for m in re.finditer(r'(?:판매\s*가격|판매가격|가격)\s*[:：]?\s*[^\n]{0,10}?(\d{1,3}(?:,\d{3})+|\d+)\s*(?:원|KRW)?', text):
        price.append(_normalize_price(m.group(1)))

    # -------------------------------
    # 3) PRICE_TYPE 보정
    #    - '현장 예매' 합성
    #    - '현매' 변형을 '현장 예매'로 통일
    # -------------------------------
    if re.search(r'현장\s*예매', text):
        ptype.append('현장 예매')
    ptype = [('현장 예매' if x in ('현매','현장예매','현장예매') else x) for x in ptype]
    # '현장' 단독은 '현장 예매'가 있으면 제거
    if '현장 예매' in ptype:
        ptype = [x for x in ptype if x != '현장']
    # Only door면 TYPE에도 힌트 남김(선택적)
    if ONLY_DOOR:
        ptype.append('현매')

    # -------------------------------
    # 4) 정리: 숫자만 보존 + 중복 제거
    # -------------------------------
    def clean_prices(xs):
        out = []
        for x in xs:
            x = _strip_space(x)
            if not x:
                continue
            # 숫자만 남기기
            m = re.search(r'(\d{1,3}(?:,\d{3})+|\d+)', x)
            if not m:
                continue
            out.append(m.group(1))
        return _dedupe(out)

    price  = clean_prices(price)
    onsite = clean_prices(onsite)

    # 5) 예매/현장에 중복 숫자 있으면, 현장쪽 숫자는 PRICE에서 제거
    if onsite:
        price = [p for p in price if p not in onsite]

    # 6) Only door 강제 규칙: PRICE에 남은 값이 있어도 전부 온사이트로 이동
    if ONLY_DOOR:
        for p in price:
            if p not in onsite:
                onsite.append(p)
        price = []

    fields['PRICE']        = price
    fields['PRICE_ONSITE'] = onsite
    fields['PRICE_TYPE']   = _dedupe([_strip_space(x) for x in ptype if _strip_space(x)])
