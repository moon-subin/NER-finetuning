# scripts/evaluate.py
# 체크포인트 평가
import os, io, json, argparse, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig
from src.models.koelectra_crf import ElectraCRF
from src.data.collate import pad_batch
from src.data.load_bio import read_conll

class ConllDataset(Dataset):
    def __init__(self, sents, tags, tokenizer, label2id, max_len=256):
        self.sents=sents; self.tags=tags; self.tk=tokenizer; self.label2id=label2id; self.max_len=max_len
    def __len__(self): return len(self.sents)
    def __getitem__(self, idx):
        tokens=self.sents[idx]; labels=self.tags[idx]
        wp_ids=[self.tk.cls_token_id]; wp_labels=[self.label2id["O"]]
        for tok,lab in zip(tokens,labels):
            pieces = self.tk.tokenize(tok) or [self.tk.unk_token]
            ids = self.tk.convert_tokens_to_ids(pieces)
            wp_ids.extend(ids); wp_labels.append(self.label2id.get(lab, self.label2id["O"]))
            for _ in pieces[1:]: wp_labels.append(self.label2id["O"])
        wp_ids.append(self.tk.sep_token_id); wp_labels.append(self.label2id["O"])
        if len(wp_ids) > self.max_len:
            wp_ids=wp_ids[:self.max_len]; wp_labels=wp_labels[:self.max_len]
        attn=[1]*len(wp_ids)
        import torch
        return {"input_ids": torch.tensor(wp_ids), "attention_mask": torch.tensor(attn), "labels": torch.tensor(wp_labels)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--split_conll", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with io.open(args.labels, "r", encoding="utf-8") as f:
        labels=[ln.strip() for ln in f if ln.strip()]
    id2label={i:lab for i,lab in enumerate(labels)}
    label2id={lab:i for i,lab in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    sents, tags = read_conll(args.split_conll)
    ds=ConllDataset(sents,tags,tokenizer,label2id,args.max_len)
    loader=DataLoader(ds, batch_size=16, shuffle=False, collate_fn=lambda b: pad_batch(b, tokenizer.pad_token_id))

    model=ElectraCRF(args.model_dir, num_labels=len(labels)).to(device)
    state=torch.load(os.path.join(args.model_dir,"pytorch_model.bin"), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    total=0; correct=0
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            paths=model(batch["input_ids"], batch["attention_mask"])
            # flatten compare on masked positions
            for b, path in enumerate(paths):
                gold = [gid.item() for gid in batch["labels"][b] if gid.item()!=-1]
                pred = path[:len(gold)]
                correct += sum(int(p==g) for p,g in zip(pred,gold))
                total   += len(gold)
    acc = correct / max(1,total)
    print(json.dumps({"token_acc": acc}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
