# scripts/apply_rules.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, io, json, argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from src.rules.regex_postrules import merge_model_and_rules, schema_guard, load_lexicons
sys.path.append(os.path.abspath(".")) 

def load_thresholds(path):
    with io.open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("thresholds", obj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="predict 출력(jsonl): {tokens, labels or model_labels, confidences, gold_labels?}")
    ap.add_argument("--thresholds", required=True, help="outputs/thresholds.json")
    ap.add_argument("--artists_csv", default=None)
    ap.add_argument("--venues_csv", default=None)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    ths = load_thresholds(args.thresholds)
    lex = load_lexicons(args.artists_csv, args.venues_csv)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    n=0
    with io.open(args.in_jsonl, "r", encoding="utf-8") as f, io.open(args.out_jsonl, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            tokens = ex["tokens"]
            yhat   = ex.get("model_labels") or ex.get("labels")  # dev/test 예측 호환
            conf   = ex.get("confidences")
            merged = merge_model_and_rules(tokens, yhat, conf, ths, lexicons=lex)
            clean  = schema_guard(merged)
            # gold 있으면 보존
            if "gold_labels" in ex:
                clean["gold_labels"] = ex["gold_labels"][:len(clean["tokens"])]
            g.write(json.dumps(clean, ensure_ascii=False) + "\n")
            n+=1
    print(f"[DONE] wrote {args.out_jsonl} (n={n})")

if __name__ == "__main__":
    main()
