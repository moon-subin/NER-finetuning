# src/inference/export.py
# 모델/토크나이저 내보내기
import os
import shutil
import argparse

def export_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in ["pytorch_model.bin", "config.json", "labels.txt", "label_map.json"]:
        src = os.path.join(args.model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out_dir, name))
    # tokenizer folder/files
    for item in os.listdir(args.model_dir):
        if item.startswith("tokenizer") or item in ["special_tokens_map.json", "tokenizer.json", "vocab.txt", "spiece.model"]:
            src = os.path.join(args.model_dir, item)
            dst = os.path.join(args.out_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
    print(f"[DONE] exported to {args.out_dir}")

if __name__ == "__main__":
    export_main()
