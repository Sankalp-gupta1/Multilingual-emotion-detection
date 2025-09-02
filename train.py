import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

MODEL_NAME = "xlm-roberta-base"
NUM_LABELS = 6   
EPOCHS = 3
BATCH_SIZE = 64
LR = 2e-5
MAX_LEN = 128

dataset = load_dataset("emotion")

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=LR)

def train_epoch(model, dataloader):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="weighted")
    return acc, f1

train_losses = []
val_accuracies = []
val_f1s = []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader)
    acc, f1 = eval_epoch(model, val_loader)
    train_losses.append(train_loss)
    val_accuracies.append(acc)
    val_f1s.append(f1)
    print(f"Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

model.save_pretrained("saved_emotion_model")
tokenizer.save_pretrained("saved_emotion_model")
print(" Training complete. Model saved in 'saved_emotion_model'")

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.subplot(1,3,2)
plt.plot(range(1, EPOCHS+1), val_accuracies, marker='o', color="green", label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()

# F1 Score Plot
plt.subplot(1,3,3)
plt.plot(range(1, EPOCHS+1), val_f1s, marker='o', color="red", label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1 Score")
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()
print(" Training graphs saved as training_metrics.png")






























#transformers>=4.41.0
#torch>=2.2.0
#pandas>=2.0.0
#numpy>=1.25.0
#streamlit>=1.35.0
#tqdm>=4.65.0
#scikit-learn>=1.3.0
#python-dateutil>=2.8.2
#datasets>=2.19.0
