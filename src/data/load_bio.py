# src/data/load_bio.py
# BIO txt 로더
import re
from typing import List, Tuple

def read_bio_txt(path: str) -> List[Tuple[List[str], List[str]]]:
    """Reads a token+label per line txt; blank line separates posts."""
    sents = []
    cur_toks, cur_labels = [], []
    pat = re.compile(r"^(.*)\s([BIO][-\w_]*)$")
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if cur_toks:
                    sents.append((cur_toks, cur_labels))
                    cur_toks, cur_labels = [], []
                continue
            m = pat.match(line.strip())
            if m:
                tok, lab = m.group(1), m.group(2)
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tok, lab = " ".join(parts[:-1]), parts[-1]
                else:
                    tok, lab = line.strip(), "O"
            cur_toks.append(tok)
            cur_labels.append(lab)
    if cur_toks:
        sents.append((cur_toks, cur_labels))
    return sents

def read_conll(path: str):
    """Reads CoNLL (token [space] label; blank line sep). Returns (sents, labels)."""
    sents, tags = [], []
    cur_x, cur_y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur_x:
                    sents.append(cur_x); tags.append(cur_y)
                    cur_x, cur_y = [], []
                continue
            parts = line.split()
            if len(parts) == 1:
                tok, lab = parts[0], "O"
            else:
                tok, lab = parts[0], parts[1]
            cur_x.append(tok); cur_y.append(lab)
    if cur_x:
        sents.append(cur_x); tags.append(cur_y)
    return sents, tags
