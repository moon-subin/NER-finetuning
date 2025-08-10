# src/rules/normalizer.py
# 이모지/특수문자/공백 정리
import re

def normalize_text(s: str) -> str:
    s = s.replace("\u200b", "")  # zero-width
    s = re.sub(r"\s+", " ", s)
    return s.strip()
