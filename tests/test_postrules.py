# tests/test_postrules.py
from src.rules.postrules import merge_model_and_rules, schema_guard, load_lexicons

def test_threshold_and_schema():
    tokens = ["2025.08.10","(일)","20:00","예매","25000","원","장소:","합정","클럽","온에어"]
    yhat = ["O","O","B-TIME","O","B-PRICE","O","O","O","O","O"]
    conf = [0.1]*len(tokens)
    ths = {"TIME":0.5, "DATE":0.5, "PRICE":0.5}
    doc = merge_model_and_rules(tokens, yhat, conf, ths, lexicons={"venues":{"합정 클럽 온에어"}})
    clean = schema_guard(doc)
    assert "TIME" in clean["fields"]
    assert clean["fields"]["VENUE"]
