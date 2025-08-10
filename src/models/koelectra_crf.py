# src/models/koelectra_crf.py
# KoELECTRA 백본 + CRF 모듈
import torch
from torch import nn
from transformers import AutoModel

# Robust CRF import: try multiple known module names
try:
    from torchcrf import CRF
except ImportError:
    try:
        from TorchCRF import CRF  # 일부 배포에서 대문자 모듈명
    except ImportError:
        from pytorch_crf import CRF  # kmkurn/pytorch-crf

class ElectraCRF(nn.Module):
    def __init__(self, model_name_or_dir: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name_or_dir)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(seq_out)
        if labels is not None:
            mask = (labels != -1).bool()
            loss = -self.crf(logits, labels, mask=mask, reduction="mean")
            return loss
        else:
            mask = attention_mask.bool()
            pred = self.crf.decode(logits, mask=mask)
            return pred
