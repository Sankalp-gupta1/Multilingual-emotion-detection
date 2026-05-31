import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# ----- Load irony dataset from HuggingFace -----
ds = load_dataset("tweet_eval", "irony")  # train / validation / test splits

test_ds = ds["test"]
texts = test_ds["text"]
true_labels = test_ds["label"]  # 0 or 1

# ----- Load pre-trained sarcasm model -----
MODEL_PATH = r"C:\Users\hp pc\emotion-detection-whatsapp\models\sarcasm_xlmr"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

# ----- Prediction -----
preds = []
batch_size = 16
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model(**enc)
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    preds.extend(batch_preds)

pred_labels = np.array(preds)

# ----- Confusion Matrix -----
cm = confusion_matrix(true_labels, pred_labels)
classes = ["Not Sarcastic", "Sarcastic"]

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Irony (Sarcasm) Model")

for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.colorbar(im)
plt.tight_layout()
plt.show()

# ----- Classification Report -----
print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=classes))
