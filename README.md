# KoELECTRA + CRF NER (Concert IG posts)

## Quick Start
```bash
# 0) create venv & install
pip install -r requirements.txt

# 1) split BIO -> CoNLL
python scripts/prepare_data.py `
  --input data/raw/최종_학습데이터셋_clean.txt `
  --out_dir data/processed

# 2) train
python scripts/train_koelectra_crf.py `
  --model_name monologg/koelectra-small-discriminator `
  --train data/processed/train.conll `
  --dev data/processed/dev.conll `
  --labels data/processed/labels.txt `
  --out_dir outputs/models/koelectra_crf

# 3) dev predict with confidences (for threshold tuning)
python -m src.inference.predict `
  --model_dir outputs/models/koelectra_crf `
  --input_conll data/processed/dev.conll `
  --out_jsonl outputs/predictions/dev_pred_with_conf.jsonl `
  --with_conf `
  --include_gold

# 4) calibrate thresholds
python scripts/calibrate_thresholds.py `
  --dev_pred_jsonl outputs/predictions/dev_pred_with_conf.jsonl `
  --out_path outputs/thresholds.json

# 규칙+thresholds 적용
python scripts\apply_rules.py `
  --in_jsonl outputs/predictions/test_pred_with_conf.jsonl `
  --thresholds outputs/thresholds.json `
  --artists_csv data/lexicon/artists.csv `
  --venues_csv data/lexicon/venues.csv `
  --out_jsonl outputs/predictions/test_final.jsonl
