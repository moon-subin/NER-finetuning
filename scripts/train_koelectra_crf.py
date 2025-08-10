# scripts/train_koelectra_crf.py
# 학습 엔트리
import sys, os
sys.path.append(os.path.abspath("."))  # ensure src/* visible

from src.training.train import train_main

if __name__ == "__main__":
    train_main()
