# scripts/score_after_rules.py
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
    ap.add_argument("--in_jsonl", required=True, help="apply_rules 결과(jsonl) - gold_labels 포함 필요")
    args = ap.parse_args()

    n=0; P=R=F=0.0
    with io.open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            if "gold_labels" not in ex: continue
            pred = ex["labels"]
            gold = ex["gold_labels"]
            # 길이 안전화
            m = min(len(pred), len(gold))
            p,r,f = token_f1(pred[:m], gold[:m])
            P+=p; R+=r; F+=f; n+=1
    if n==0:
        print("{\"error\": \"no gold_labels found\"}")
        return
    print(json.dumps({
        "n_examples": n,
        "token_precision": P/n,
        "token_recall": R/n,
        "token_f1": F/n
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
