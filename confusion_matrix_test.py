import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1. Model + Tokenizer Load
# -------------------------
MODEL_PATH = r"C:\Users\hp pc\emotion-detection-whatsapp\saved_emotion_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

# -------------------------
# 2. Load Test Dataset
# -------------------------
df = pd.read_csv(r"C:\Users\hp pc\emotion-detection-whatsapp\data\test.csv")
texts = df["text"].tolist()
true_labels = df["label"].tolist()

# -------------------------
# 3. Label Mapping
# -------------------------
# -------------------------
# 5. Confusion Matrix (Matplotlib only)
# -------------------------
label_map = {
    0: "Happy",
    1: "Sad",
    2: "Angry",
    3: "Neutral",
    4: "Fear",
    5: "Surprise"   # <-- Add this
}


# -------------------------
# 4. Prediction Function
# -------------------------
def predict(texts, batch_size=16):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    return np.array(all_preds)

# -------------------------
# 5. Run Predictions
# -------------------------
pred_labels = predict(texts)

# -------------------------
# 6. Confusion Matrix with Emotion Names
# -------------------------
cm = confusion_matrix(true_labels, pred_labels)
classes = [label_map[i] for i in np.unique(true_labels)]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")

# Show all ticks and label them with emotion names
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=45)
ax.set_yticklabels(classes)

plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")
plt.title("Confusion Matrix - XLM-RoBERTa Emotion Model")

# Annotate cells with values
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.colorbar(im)
plt.tight_layout()
plt.show()

# -------------------------
# 7. Classification Report with Emotion Names
# -------------------------
print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=[label_map[i] for i in sorted(label_map.keys())]))
