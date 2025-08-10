# src/training/metrics.py
# seqeval F1, 라벨별 점수
from seqeval.metrics import f1_score, classification_report

def seq_f1(y_true, y_pred, average="micro"):
    return f1_score(y_true, y_pred, average=average)

def seq_report(y_true, y_pred):
    return classification_report(y_true, y_pred, digits=4)
