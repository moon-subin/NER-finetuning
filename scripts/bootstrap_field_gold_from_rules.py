#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, io, json, argparse, re, torch, torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from transformers import AutoTokenizer, AutoConfig
from src.models.koelectra_crf import ElectraCRF
from src.rules.postrules import merge_model_and_rules, schema_guard, load_lexicons

FIELD_KEYS = ["TITLE","DATE","TIME","PRICE","PRICE_ONSITE","PRICE_TYPE",
              "LINEUP","INSTAGRAM","VENUE","V_INSTA","TICKET_OPEN_DATE","TICKET_OPEN_TIME"]

def simple_tokenize(text: str):
    text = re.sub(r"\s+"," ",(text or "").strip())
    price = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\s*(?:원|won))?"
    weekday_paren = r"\((?:[월화수목금토일]|Mon|Tue|Wed|Thu|Fri|Sat|Sun|MON|TUE|WED|THU|FRI|SAT|SUN)\)"
    time_units = r"(?:(?:오전|오후|낮|밤|저녁)\s*\d{1,2}시(?:\s*\d{1,2}분)?|\d{1,2}:\d{2}(?:\s*(?:am|pm|AM|PM))?|\d{1,2}\s*(?:am|pm|AM|PM)|\d{1,2}시(?:\s*\d{1,2}분)?)"
    date_units = r"(?:\d{4}년|\d{1,2}월|\d{1,2}일|\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}|\d{1,2}\s*\.\s*\d{1,2}|\d{4}/\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2})"
    token_re = re.compile(rf"{weekday_paren}|{time_units}|{date_units}|{price}|@[A-Za-z0-9_.]+|#[^\s#@]+|[A-Za-z]+|[가-힣]+|\d+|[^\sA-Za-z0-9가-힣]")
    toks = token_re.findall(text)
    merged, i = [], 0
    while i < len(toks):
        t = toks[i]
        if i+1 < len(toks) and toks[i+1].lower() in ("원","won") and re.fullmatch(r"(?:\d{1,3}(?:,\d{3})+|\d+)", t):
            merged.append(t+"원"); i+=2; continue
        merged.append(t); i+=1
    return merged

def tokens_to_wp(tokenizer, tokens, max_len):
    ids=[tokenizer.cls_token_id]; align=[-1]
    for i,tok in enumerate(tokens):
        pieces=tokenizer.tokenize(tok) or [tokenizer.unk_token]
        ids+=tokenizer.convert_tokens_to_ids(pieces); align+=[i]*len(pieces)
    ids.append(tokenizer.sep_token_id); align.append(-1)
    if len(ids)>max_len: ids=ids[:max_len]; align=align[:max_len]
    attn=[1]*len(ids)
    return torch.tensor(ids).long(), torch.tensor(attn).long(), torch.tensor(align).long()

def decode_wp_to_token_labels(wp_pred, align, id2label):
    max_tok=int(align.max().item()) if (align>=0).any() else -1
    out=["O"]*(max_tok+1 if max_tok>=0 else 0); seen=set()
    for pos,lid in enumerate(wp_pred):
        ti=int(align[pos].item())
        if 0<=ti<len(out) and ti not in seen:
            out[ti]=id2label[lid]; seen.add(ti)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--texts", required=True, help="각 줄이 하나의 게시글 텍스트인 .txt")
    ap.add_argument("--out_jsonl", required=True, help="부트스트랩 결과 jsonl")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--artists_csv", default=None)
    ap.add_argument("--venues_csv", default=None)
    ap.add_argument("--max_len", type=int, default=256)
    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    config=AutoConfig.from_pretrained(args.model_dir)
    try:
        tokenizer=AutoTokenizer.from_pretrained(args.model_dir); backbone=args.model_dir
    except Exception:
        backbone=getattr(config,"_name_or_path",None) or "monologg/koelectra-base-v3-discriminator"
        tokenizer=AutoTokenizer.from_pretrained(backbone)
    with io.open(os.path.join(args.model_dir,"labels.txt"),"r",encoding="utf-8") as f:
        labels=[ln.strip() for ln in f if ln.strip()]
    model=ElectraCRF(backbone, num_labels=len(labels)).to(device)
    state=torch.load(os.path.join(args.model_dir,"pytorch_model.bin"), map_location=device)
    model.load_state_dict(state, strict=True); model.eval()

    thresholds={}
    if args.thresholds and os.path.exists(args.thresholds):
        with io.open(args.thresholds,"r",encoding="utf-8") as f:
            got=json.load(f); thresholds=got.get("thresholds",got)
    lex=load_lexicons(args.artists_csv, args.venues_csv)

    with io.open(args.texts,"r",encoding="utf-8") as fin, io.open(args.out_jsonl,"w",encoding="utf-8") as fout:
        for raw in fin:
            text=raw.strip("\n")
            if not text: continue
            tokens=simple_tokenize(text)
            input_ids,attn,align=tokens_to_wp(tokenizer,tokens,args.max_len)
            with torch.no_grad():
                outputs=model.backbone(input_ids=input_ids.unsqueeze(0).to(device), attention_mask=attn.unsqueeze(0).to(device))
                seq_out=model.dropout(outputs.last_hidden_state)
                logits=model.classifier(seq_out)
                paths=model.crf.decode(logits, mask=attn.unsqueeze(0).bool())
            pred_tok_labels=decode_wp_to_token_labels(paths[0], align, labels)

            merged=merge_model_and_rules(tokens, pred_tok_labels, confidences=None, thresholds=thresholds, lexicons=lex)
            merged=schema_guard(merged)
            # 골드 초안: 사람이 검수하기 쉽게 text와 필드만 남김
            out={"text": text}
            for k in FIELD_KEYS:
                out[k]=merged.get(k,[]) or []
            fout.write(json.dumps(out, ensure_ascii=False)+"\n")
    print(f"[DONE] bootstrap saved: {args.out_jsonl}")

if __name__=="__main__":
    main()
