# src/data/split.py
# train/dev/test 분할(시드 고정)
import random
from typing import List, Tuple

def split_dataset(sents: List[Tuple[list, list]],
                  ratios=(0.8, 0.1, 0.1),
                  seed=42):
    r_train, r_dev, r_test = ratios
    n = len(sents)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)

    n_train = int(round(n * r_train))
    n_dev = int(round(n * r_dev))
    n_test = n - n_train - n_dev

    train_idx = idx[:n_train]
    dev_idx   = idx[n_train:n_train+n_dev]
    test_idx  = idx[n_train+n_dev:]

    take = lambda ids: [sents[i] for i in ids]
    return take(train_idx), take(dev_idx), take(test_idx)

