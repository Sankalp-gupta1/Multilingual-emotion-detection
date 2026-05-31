from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

app = FastAPI()

class Message(BaseModel):
    text: str

# Load models
emo_tok = XLMRobertaTokenizer.from_pretrained(r"C:\Users\hp pc\emotion-detection-whatsapp\saved_emotion_model")
emo_model = XLMRobertaForSequenceClassification.from_pretrained(r"C:\Users\hp pc\emotion-detection-whatsapp\saved_emotion_model")
emo_model.eval()

sarc_tok = XLMRobertaTokenizer.from_pretrained(r"C:\Users\hp pc\emotion-detection-whatsapp\models\sarcasm_xlmr")
sarc_model = XLMRobertaForSequenceClassification.from_pretrained(r"C:\Users\hp pc\emotion-detection-whatsapp\models\sarcasm_xlmr")
sarc_model.eval()

@app.post("/predict")
def predict(msg: Message):
    # Emotion
    inputs = emo_tok(msg.text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        emo_out = emo_model(**inputs)
    emo_pred = torch.argmax(emo_out.logits, dim=1).item()
    emo_map = {0:"Sadness 😢",1:"Joy 😀",2:"Love ❤️",3:"Anger 😡",4:"Fear 😨",5:"Surprise 😲"}

    # Sarcasm
    inputs_s = sarc_tok(msg.text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        sarc_out = sarc_model(**inputs_s)
    sarc_pred = torch.argmax(sarc_out.logits, dim=1).item()
    sarc_map = {0:"Not Sarcastic 🙂", 1:"Sarcastic 🙃"}

    return {"emotion": emo_map[emo_pred], "sarcasm": sarc_map[sarc_pred]}
