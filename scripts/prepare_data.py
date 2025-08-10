# scripts/prepare_data.py
# raw → processed/*.conll 생성
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, json
from src.data.load_bio import read_bio_txt
from src.data.split import split_dataset

def write_conll(path, sents):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for tokens, labels in sents:
            for t, l in zip(tokens, labels):
                f.write(f"{t} {l}\n")
            f.write("\n")

def collect_labels(sents):
    labs=set()
    for _, y in sents: labs.update(y)
    labs = ["O"] + sorted([l for l in labs if l!="O"])
    return labs

def calc_stats(sents):
    from collections import Counter
    stats = {
        "num_posts": len(sents),
        "num_tokens": sum(len(x[0]) for x in sents),
        "avg_tokens_per_post": round(sum(len(x[0]) for x in sents)/max(1,len(sents)),2),
        "label_counts": {}
    }
    c = Counter()
    for _, labs in sents: c.update(labs)
    stats["label_counts"] = dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratio", nargs=3, type=float, default=[0.8,0.1,0.1])
    args = ap.parse_args()

    sents = read_bio_txt(args.input)
    train, dev, test = split_dataset(sents, args.ratio, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    write_conll(os.path.join(args.out_dir, "train.conll"), train)
    write_conll(os.path.join(args.out_dir, "dev.conll"), dev)
    write_conll(os.path.join(args.out_dir, "test.conll"), test)

    labels = collect_labels(sents)
    with open(os.path.join(args.out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))

    stats = {
        "ratios": {"train": args.ratio[0], "dev": args.ratio[1], "test": args.ratio[2]},
        "seed": args.seed,
        "all": calc_stats(sents),
        "train": calc_stats(train),
        "dev": calc_stats(dev),
        "test": calc_stats(test),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("[done] saved CoNLL & stats under", args.out_dir)

if __name__ == "__main__":
    main()
