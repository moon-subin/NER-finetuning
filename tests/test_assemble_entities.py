# tests/test_assemble_entities.py
from src.inference.assemble_entities import bio_to_spans

def test_bio_to_spans():
    tokens = ["A","B","C"]
    labels = ["B-TITLE","I-TITLE","O"]
    spans = bio_to_spans(tokens, labels)
    assert spans and spans[0][2] == "TITLE"
