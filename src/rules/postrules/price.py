# 가격 스캔/정규화(_rescan_and_fix_prices)
# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Optional
from .textutils import _strip_space, _dedupe, _parse_kr_price, _normalize_price, _price_to_int
from .patterns import RE_PRICE

def _rescan_and_fix_prices(text: str, tokens: List[str], fields: Dict) -> None:
    text = re.sub(r'\bKRW\b', '원', text, flags=re.IGNORECASE).replace('₩','원').replace('￦','원')
    price = list(fields.get('PRICE', []) or [])
    onsite = list(fields.get('PRICE_ONSITE', []) or [])

    def tok_span_to_str(i, j): return _strip_space(''.join(tokens[i:j]))

    def pick_best_amount(slices: List[str]) -> Optional[str]:
        best, best_val = None, -1
        for cand in slices:
            s = _strip_space(cand)
            s = re.sub(r'\bKRW\b', '원', s, flags=re.IGNORECASE).replace('₩','원').replace('￦','원')
            v = _parse_kr_price(s)
            if not v and RE_PRICE.fullmatch(_strip_space(s or '')): v = _normalize_price(s)
            if v:
                val = _price_to_int(v)
                if val > best_val: best, best_val = v, val
        return best

    for i, tk in enumerate(tokens):
        if tk == '예매':
            candidates = [tok_span_to_str(i+1, j+1) for j in range(i+1, min(i+6, len(tokens)))]
            best = pick_best_amount(candidates)
            if best: price.append(best)
        if tk in ('현매', '현장'):
            candidates = [tok_span_to_str(i+1, j+1) for j in range(i+1, min(i+6, len(tokens)))]
            best = pick_best_amount(candidates)
            if best: onsite.append(best)

    for m in re.finditer(r'(예매)', text):
        win = text[m.end(): m.end()+60]
        words = re.findall(r'[0-9가-힣,]+(?:\s*(?:KRW|원))?|[0-9가-힣,]+', win, flags=re.IGNORECASE)
        best = pick_best_amount(words)
        if best: price.append(best)
    for m in re.finditer(r'(현매|현장)', text):
        win = text[m.end(): m.end()+60]
        words = re.findall(r'[0-9가-힣,]+(?:\s*(?:KRW|원))?|[0-9가-힣,]+', win, flags=re.IGNORECASE)
        best = pick_best_amount(words)
        if best: onsite.append(best)

    def clean_prices(xs):
        out = []
        for x in xs:
            x = _strip_space(x)
            if not x: continue
            if not x.endswith('원'):
                if re.fullmatch(r'\d+', x):  # 숫자만 → 버림
                    continue
                v = _parse_kr_price(x)
                if v: x = v
                else: continue
            out.append(x)
        return _dedupe(out)

    # ---- Fallback: '현매만'/'현장만' 텍스트가 있고 가격이 하나도 없을 때 ----
    text_norm = text  # 이미 KRW→원 통일됨
    no_price_info = (not price) and (not onsite)
    if no_price_info:
        if re.search(r'(현매만|현장만)\s*가능|현매만', text_norm):
            # 본문 전체에서 가장 큰 금액 하나를 찾아 현매가로 지정
            all_amounts = []
            for m in RE_PRICE.finditer(text_norm):
                amt = _normalize_price(m.group(0))
                if amt: all_amounts.append(amt)
            if all_amounts:
                all_amounts = sorted(all_amounts, key=_price_to_int, reverse=True)
                onsite.append(all_amounts[0])
        elif re.search(r'예매만', text_norm):
            all_amounts = []
            for m in RE_PRICE.finditer(text_norm):
                amt = _normalize_price(m.group(0))
                if amt: all_amounts.append(amt)
            if all_amounts:
                all_amounts = sorted(all_amounts, key=_price_to_int, reverse=True)
                price.append(all_amounts[0])

    fields['PRICE'] = clean_prices(price)
    fields['PRICE_ONSITE'] = clean_prices(onsite)
