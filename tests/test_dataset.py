# tests/test_dataset.py
from src.data.load_bio import read_bio_txt

def test_read_bio_txt_basic(tmp_path):
    p = tmp_path / "bio.txt"
    p.write_text("A B-TITLE\nB I-TITLE\n\nC O\n", encoding="utf-8")
    sents = read_bio_txt(str(p))
    assert len(sents) == 2
    assert sents[0][1][0].startswith("B-")
