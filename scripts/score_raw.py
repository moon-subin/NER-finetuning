# scripts/score_raw.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, json, argparse

def token_f1(pred, gold):
    tp=fp=fn=0
    for p,g in zip(pred,gold):
        if g!="O" and p==g: tp+=1
        elif p!="O" and p!=g: fp+=1
        elif g!="O" and p!=g: fn+=1
    prec = tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    return prec, rec, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    args = ap.parse_args()
    n=P=R=F=0
    with io.open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            if "gold_labels" not in ex: continue
            pred = ex["labels"]; gold = ex["gold_labels"]
            m = min(len(pred), len(gold))
            p,r,f = token_f1(pred[:m], gold[:m])
            P+=p; R+=r; F+=f; n+=1
    print({"n":n, "prec":P/n if n else 0, "rec":R/n if n else 0, "f1":F/n if n else 0})

if __name__ == "__main__":
    main()
