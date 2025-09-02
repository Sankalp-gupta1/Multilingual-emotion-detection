
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import re


# Streamlit Config

st.set_page_config(
    page_title="Multilingual Emotion + Sarcasm Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
.stButton>button {
    background-color: #ff4b2b;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
.card {
    background: rgba(255, 255, 255, 0.1);
    padding: 15px;
    margin: 10px 0;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Multilingual Emotion + Sarcasm Detection")
st.caption("Type your message and get predictions for Emotion & Sarcasm.")


# Load Models

@st.cache_resource
def load_models():
    emo_tokenizer = XLMRobertaTokenizer.from_pretrained(
        r"C:\Users\hp pc\emotion-detection-whatsapp\saved_emotion_model"
    )
    emo_model = XLMRobertaForSequenceClassification.from_pretrained(
        r"C:\Users\hp pc\emotion-detection-whatsapp\saved_emotion_model"
    )
    emo_model.eval()

    sarc_tokenizer = XLMRobertaTokenizer.from_pretrained(
        r"C:\Users\hp pc\emotion-detection-whatsapp\models\sarcasm_xlmr"
    )
    sarc_model = XLMRobertaForSequenceClassification.from_pretrained(
        r"C:\Users\hp pc\emotion-detection-whatsapp\models\sarcasm_xlmr"
    )
    sarc_model.eval()

    return emo_tokenizer, emo_model, sarc_tokenizer, sarc_model

emo_tok, emo_model, sarc_tok, sarc_model = load_models()


# Prediction Functions

def predict_emotion(text: str):
    inputs = emo_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = emo_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0:"Sadness 😢",1:"Joy 😀",2:"Love ❤️",3:"Anger 😡",4:"Fear 😨",5:"Surprise 😲"}
    return label_map[pred]

def predict_sarcasm(text: str):
    inputs = sarc_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = sarc_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0:"Not Sarcastic 🙂",1:"Sarcastic 🙃"}
    return label_map[pred]

# WhatsApp Chat Parsing (simple)

PATTERN = re.compile(r"^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),\s+(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\s+-\s+(.*?):\s+(.*)$")
def parse_whatsapp(txt_path):
    lines = txt_path.splitlines()
    rows = []
    for line in lines:
        m = PATTERN.match(line.strip())
        if m:
            date, time, sender, msg = m.groups()
            msg = msg.strip()
            if msg and msg.lower() != "<media omitted>":
                rows.append({"datetime": f"{date} {time}", "sender": sender.strip(), "message": msg})
        else:  
            if line.strip():
                rows.append({"datetime":"", "sender":"", "message": line.strip()})
    return rows


st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Dashboard", "Upload Chat", "Results", "Graphs"])


if menu == "Dashboard":
    user_text = st.text_area("Type your message here:", height=150)
    if user_text:
        with st.spinner("Analyzing..."):
            emo = predict_emotion(user_text)
            sarc = predict_sarcasm(user_text)
        st.markdown(f'<div class="card"><h3>Emotion: {emo}</h3></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><h3>Sarcasm: {sarc}</h3></div>', unsafe_allow_html=True)


elif menu == "Upload Chat":
    st.subheader("Upload WhatsApp/Telegram Chat (.txt)")
    uploaded_file = st.file_uploader("Choose a file", type=["txt"])
    if uploaded_file:
        st.success("File uploaded successfully!")
        chat_content = uploaded_file.read().decode("utf-8-sig")
        st.info("Processing messages...")

        chat_rows = parse_whatsapp(chat_content)
        results = []
        for row in chat_rows:
            msg = row["message"]
            emo = predict_emotion(msg)
            sarc = predict_sarcasm(msg)
            results.append({"message": msg, "emotion": emo, "sarcasm": sarc})

     
        st.session_state["chat_results"] = results

        st.subheader("Prediction Results (Upload Chat)")
        for r in results:
            st.markdown(f'<div class="card"><b>Message:</b> {r["message"]}<br>'
                        f'<b>Emotion:</b> {r["emotion"]}<br>'
                        f'<b>Sarcasm:</b> {r["sarcasm"]}</div>', unsafe_allow_html=True)


elif menu == "Results":
    st.subheader("Prediction Results")
    if "chat_results" in st.session_state:
        for r in st.session_state["chat_results"]:
            st.markdown(f'<div class="card"><b>Message:</b> {r["message"]}<br>'
                        f'<b>Emotion:</b> {r["emotion"]}<br>'
                        f'<b>Sarcasm:</b> {r["sarcasm"]}</div>', unsafe_allow_html=True)
    else:
        st.info("Here you will see the processed emotions & sarcasm from uploaded chats (coming soon).")


elif menu == "Graphs":
    st.subheader("Graphs & Analytics")
    if "chat_results" in st.session_state:
        df = pd.DataFrame(st.session_state["chat_results"])
        # Emotion distribution
        fig1, ax1 = plt.subplots()
        df['emotion'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title("Emotion Distribution")
        st.pyplot(fig1)
        # Sarcasm distribution
        fig2, ax2 = plt.subplots()
        df['sarcasm'].value_counts().plot(kind='bar', ax=ax2, color='orange')
        ax2.set_title("Sarcasm Distribution")
        st.pyplot(fig2)
    else:
        st.info("Upload a chat first to see graphs.")
