# src/inference/assemble_entities.py
# BIO → 엔티티 병합(JSON)
import json, io, argparse, re
from typing import List, Dict, Any

def bio_to_spans(tokens: List[str], labels: List[str]):
    spans = []
    cur_s, cur_t = None, None
    for i, lab in enumerate(labels):
        if lab == "O":
            if cur_s is not None:
                spans.append((cur_s, i, cur_t))
                cur_s, cur_t = None, None
            continue
        p, t = lab.split("-", 1)
        if p == "B":
            if cur_s is not None:
                spans.append((cur_s, i, cur_t))
            cur_s, cur_t = i, t
        elif p == "I":
            if cur_t != t or cur_s is None:
                if cur_s is not None:
                    spans.append((cur_s, i, cur_t))
                cur_s, cur_t = i, t
    if cur_s is not None:
        spans.append((cur_s, len(labels), cur_t))
    return spans

def assemble_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    rows = []
    with io.open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            tokens = obj["tokens"]; labels = obj["labels"]
            spans = bio_to_spans(tokens, labels)
            ents = [{"type": t, "start": s, "end": e, "text": " ".join(tokens[s:e]).strip()}
                    for (s,e,t) in spans]
            obj["entities"] = ents
            rows.append(obj)

    with io.open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] wrote {args.out_jsonl}  (n={len(rows)})")

if __name__ == "__main__":
    assemble_main()
