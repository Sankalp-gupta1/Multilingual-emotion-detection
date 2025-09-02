# src/sarcasm_train.py
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

MODEL = "xlm-roberta-base"
EPOCHS = 2
BATCH = 64
LR = 2e-5
MAX_LEN = 128

# twitter datasset
ds = load_dataset("tweet_eval", "irony")

# tokenizer
tok = XLMRobertaTokenizer.from_pretrained(MODEL)
def tok_fn(b):
    return tok(b["text"], padding="max_length", truncation=True, max_length=MAX_LEN)
ds = ds.map(tok_fn, batched=True)


ds = ds.rename_column("label", "labels")
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = DataLoader(ds["train"], batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(ds["validation"], batch_size=BATCH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
opt = AdamW(model.parameters(), lr=LR)

def run_epoch(loader, train=True):
    if train: model.train()
    else: model.eval()

    tot, preds, gold = 0.0, [], []
    for batch in tqdm(loader, desc="train" if train else "eval"):
        if train: opt.zero_grad()
        inp = {k:v.to(device) for k,v in batch.items()}
        out = model(**inp)

        if train:
            out.loss.backward()
            opt.step()

        tot += float(out.loss)
        with torch.no_grad():
            p = out.logits.argmax(dim=1).cpu().numpy().tolist()
            g = inp["labels"].cpu().numpy().tolist()
            preds += p; gold += g
    return tot/len(loader), accuracy_score(gold,preds), f1_score(gold,preds)

for e in range(EPOCHS):
    tl, ta, tf1 = run_epoch(train_loader, True)
    vl, va, vf1 = run_epoch(val_loader, False)
    print(f"Epoch {e+1}: train_loss={tl:.4f} val_acc={va:.4f} val_f1={vf1:.4f}")

model.save_pretrained("models/sarcasm_xlmr")
tok.save_pretrained("models/sarcasm_xlmr")
print("Saved models/sarcasm_xlmr")
