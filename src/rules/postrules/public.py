# -*- coding: utf-8 -*-
from typing import List, Optional, Dict
from .schema import _apply_thresholds, _fields_from_bio, schema_guard
from .core import apply_regex_postrules

def merge_model_and_rules(
    tokens: List[str],
    model_labels: List[str],
    confidences: Optional[List[float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    lexicons: Optional[Dict] = None
) -> Dict:
    """
    1) 모델 BIO에 임계치 적용
    2) BIO → fields
    3) 텍스트 조립 후 regex 기반 후처리
    4) 스키마 정리
    """
    thresholds = thresholds or {}
    labels = _apply_thresholds(model_labels, confidences, thresholds)
    fields = _fields_from_bio(tokens, labels)
    text = " ".join(tokens)
    final = apply_regex_postrules(text, tokens, fields, lexicons=lexicons)
    final["tokens"] = tokens
    final["text"] = text
    return schema_guard(final)
