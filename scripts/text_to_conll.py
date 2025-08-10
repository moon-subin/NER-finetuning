# scripts/text_to_conll.py 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re, io, argparse, os

def simple_tokenize(text: str):
    # 공백 기준 + 이모지/문장부호를 토큰으로 분리
    # 필요하면 더 정교화 가능
    text = re.sub(r"(\s+)", " ", text.strip())
    # 공백 기준 split 후, 붙은 구두점은 띄우기
    out=[]
    for w in text.split():
        # 앞뒤 구두점 분리
        parts = re.findall(r"[A-Za-z0-9_@.#]+|[가-힣]+|[^\sA-Za-z0-9_가-힣]", w)
        out.extend(parts)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True, help="원본 txt (한 줄/여러 줄 상관없음)")
    ap.add_argument("--out_conll", required=True, help="토큰 당 한 줄, 문단 사이 빈 줄")
    args = ap.parse_args()

    with io.open(args.in_txt, "r", encoding="utf-8") as f:
        raw = f.read()

    # 빈 줄 기준으로 문단 분리
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]

    os.makedirs(os.path.dirname(args.out_conll) or ".", exist_ok=True)
    with io.open(args.out_conll, "w", encoding="utf-8") as g:
        for b in blocks:
            toks = simple_tokenize(b)
            for t in toks:
                g.write(f"{t}\n")          # 라벨은 없어도 됨(우리 predict가 첫 컬럼만 읽음)
            g.write("\n")
    print(f"[DONE] wrote {args.out_conll}  ({len(blocks)} blocks)")

if __name__ == "__main__":
    main()
