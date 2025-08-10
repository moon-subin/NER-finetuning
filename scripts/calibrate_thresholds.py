# scripts/calibrate_thresholds.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, argparse, os
from collections import defaultdict
from src.rules.regex_postrules import merge_model_and_rules, schema_guard, load_lexicons

GRID = [round(x,2) for x in [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95]]
TARGET_ENTS = ["TITLE","DATE","TIME","PRICE","PRICE_TYPE","PRICE_ONSITE","LINEUP","INSTAGRAM","VENUE","V_INSTA","TICKET_OPEN_DATE","TICKET_OPEN_TIME"]

def token_f1(pred, gold):
    tp=fp=fn=0
    for p,g in zip(pred,gold):
        if g!="O" and p==g: tp+=1
        elif p!="O" and p!=g: fp+=1
        elif g!="O" and p!=g: fn+=1
    prec = tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    return f1

def load_dev_jsonl(path):
    data=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data.append(json.loads(line))
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_pred_jsonl", required=True, help="each line: {tokens, model_labels, confidences, gold_labels}")
    ap.add_argument("--out_path", default="outputs/thresholds.json")
    args = ap.parse_args()

    dev = load_dev_jsonl(args.dev_pred_jsonl)

    base = {e:0.65 for e in TARGET_ENTS}
    cur = base.copy()

    def evaluate(ths):
        f1s=[]; schema=[]
        for ex in dev:
            tokens = ex["tokens"]
            yhat = ex.get("model_labels") or ex.get("labels")
            conf = ex.get("confidences") or [1.0]*len(yhat)
            gold = ex["gold_labels"]
            merged = merge_model_and_rules(tokens, yhat, conf, ths)
            f1s.append(token_f1(merged["labels"], gold))  # BUGFIX: use ["labels"]
            schema.append(1 if schema_guard(merged) else 0)
        return (sum(f1s)/len(f1s), sum(schema)/len(schema))

    # 1) per-entity grid search
    for ent in TARGET_ENTS:
        local_best = cur.copy(); local_score=-1
        for v in GRID:
            trial = cur.copy(); trial[ent]=v
            f1, sch = evaluate(trial)
            score = f1 + 0.1*sch
            if score > local_score:
                local_score=score; local_best=trial
        cur = local_best

    # 2) small global tweaks
    for _ in range(2):
        improved=False
        for ent in TARGET_ENTS:
            base_v = cur[ent]
            for dv in [-0.10, -0.05, 0.05, 0.10]:
                nv = round(min(0.95, max(0.5, base_v+dv)),2)
                trial = cur.copy(); trial[ent]=nv
                f1, sch = evaluate(trial)
                score = f1 + 0.1*sch
                f1b, schb = evaluate(cur)
                if score > (f1b + 0.1*schb):
                    cur = trial; improved=True
        if not improved: break

    f1, sch = evaluate(cur)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump({"thresholds": cur, "dev_token_f1": f1, "dev_schema": sch}, f, ensure_ascii=False, indent=2)
    print("[save]", args.out_path)

if __name__ == "__main__":
    main()
