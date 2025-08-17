# /rules/postrules/schema.py
# BIO→fields, thresholds, schema_guard, load_lexicons
# -*- coding: utf-8 -*-
import csv
from typing import Dict, List, Optional, Tuple

def _bio_to_spans(tokens: List[str], labels: List[str]) -> List[Tuple[str,int,int]]:
    spans=[]; cur_type, s = None, None
    for i, lab in enumerate(labels):
        if not lab or lab == 'O':
            if cur_type is not None: spans.append((cur_type, s, i)); cur_type, s = None, None
            continue
        if '-' in lab: tag, etype = lab.split('-', 1)
        else: tag, etype = lab, None
        if tag == 'B':
            if cur_type is not None: spans.append((cur_type, s, i))
            cur_type, s = etype, i
        elif tag == 'I':
            if cur_type != etype:
                if cur_type is not None: spans.append((cur_type, s, i))
                cur_type, s = etype, i
        else:
            if cur_type is not None: spans.append((cur_type, s, i))
            cur_type, s = None, None
    if cur_type is not None: spans.append((cur_type, s, len(labels)))
    return spans

def _apply_thresholds(labels: List[str], confidences: Optional[List[float]], thresholds) -> List[str]:
    if not confidences: return labels
    out=[]
    for lab, conf in zip(labels, confidences):
        if not lab or lab == 'O': out.append('O'); continue
        etype = lab.split('-', 1)[-1] if '-' in lab else lab
        thr = thresholds.get(etype, 0.0) if isinstance(thresholds, dict) else 0.0
        keep = (conf is None) or (float(conf) >= float(thr))
        out.append(lab if keep else 'O')
    return out

def _fields_from_bio(tokens: List[str], labels: List[str]) -> Dict[str, List[str]]:
    fields = {
        "TITLE": [], "DATE": [], "TIME": [],
        "PRICE": [], "PRICE_ONSITE": [], "PRICE_TYPE": [],
        "LINEUP": [], "INSTAGRAM": [],
        "VENUE": [], "V_INSTA": [],
        "TICKET_OPEN_DATE": [], "TICKET_OPEN_TIME": []
    }
    spans = _bio_to_spans(tokens, labels)
    def tok_join(a,b): return ' '.join(tokens[a:b]).strip()
    for etype, s, e in spans:
        text = tok_join(s,e)
        if not text: continue
        if   etype == 'PRICE':              fields['PRICE'].append(text)
        elif etype == 'PRICE_ONSITE':       fields['PRICE_ONSITE'].append(text)
        elif etype == 'PRICE_TYPE':         fields['PRICE_TYPE'].append(text)
        elif etype == 'LINEUP':             fields['LINEUP'].append(text)
        elif etype == 'INSTAGRAM':          fields['INSTAGRAM'].append(text)
        elif etype == 'VENUE':              fields['VENUE'].append(text)
        elif etype == 'V_INSTA':            fields['V_INSTA'].append(text)
        elif etype == 'TICKET_OPEN_DATE':   fields['TICKET_OPEN_DATE'].append(text)
        elif etype == 'TICKET_OPEN_TIME':   fields['TICKET_OPEN_TIME'].append(text)
        elif etype == 'DATE':               fields['DATE'].append(text)
        elif etype == 'TIME':               fields['TIME'].append(text)
        elif etype == 'TITLE':              fields['TITLE'].append(text)
    return fields

def load_lexicons(artists_csv: Optional[str]=None, venues_csv: Optional[str]=None) -> Dict[str, set]:
    def load_csv(path: Optional[str]) -> set:
        bag=set()
        if not path: return bag
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for row in csv.reader(f):
                    for col in row:
                        col=(col or '').strip()
                        if col: bag.add(col)
        except FileNotFoundError:
            pass
        return bag
    return {"artists": load_csv(artists_csv), "venues": load_csv(venues_csv)}

def schema_guard(obj: Dict) -> Dict:
    keys = ["TITLE","DATE","TIME","PRICE","PRICE_ONSITE","PRICE_TYPE",
            "LINEUP","INSTAGRAM","VENUE","V_INSTA",
            "TICKET_OPEN_DATE","TICKET_OPEN_TIME"]
    out = {"tokens": obj.get("tokens", []), "text": obj.get("text", "")}
    for k in keys:
        v = obj.get(k, [])
        if isinstance(v, list): out[k] = v
        elif v is None: out[k] = []
        else: out[k] = [str(v)]
    return out
