🤖 Multilingual Emotion & Sarcasm Detection
📖 ABSTRACT

Understanding human communication requires more than just literal text analysis.
Emotions and sarcasm are crucial components of natural language but are challenging for machines to interpret, especially in multilingual settings.

This project presents a Streamlit-based interactive system that integrates fine-tuned transformer models (XLM-RoBERTa) for multilingual emotion classification and sarcasm detection.

Supports real-time inference

Allows bulk analysis of WhatsApp/Telegram chats

Provides visualized analytics dashboards

🧩 MOTIVATION

Ambiguity in Language
👉 Example: “Wow, great job 👏” can either be genuine praise or sarcasm depending on context.

Social Media Complexity
👉 Platforms like WhatsApp, Twitter, and Telegram contain highly informal and emotionally nuanced text.

Multilingual Challenge
👉 Existing emotion/sarcasm systems are often restricted to English.

Our Approach
👉 Leverages XLM-RoBERTa for cross-lingual generalization.

🎯 OBJECTIVES

Build a robust emotion classifier capable of detecting six emotions: Joy, Love, Anger, Fear, Sadness, Surprise.

Develop a sarcasm detection model trained on irony datasets.

Provide a user-friendly interface for single-message predictions & chat-level analytics.

Demonstrate the power of transfer learning with multilingual transformers.

📚 RELATED WORK

Emotion Detection

Traditional methods (SVMs, Naïve Bayes) are weak in multilingual settings.

Transformers like BERT and XLM-RoBERTa achieve state-of-the-art results.

Sarcasm Detection

Prior research shows sarcasm is context-heavy, often requiring multimodal signals.

Our project focuses on text-based sarcasm detection using TweetEval - Irony Dataset.

🚀 APPLICATIONS

Sentiment Analysis

Mental Health Monitoring

Conversational AI

Social Media Moderation


                ┌──────────────────────────┐
                │  User Input / Chat File  │
                └──────────────┬───────────┘
                               │
                      ┌────────▼─────────┐
                      │ Preprocessing    │
                      │ (Tokenizer, PAD) │
                      └────────┬─────────┘
                               │
                ┌──────────────▼─────────────┐
                │ Fine-tuned XLM-RoBERTa     │
                │ Emotion Model (6 classes)  │
                │ Sarcasm Model (binary)     │
                └──────────────┬─────────────┘
                               │
                 ┌─────────────▼─────────────┐
                 │ Predictions (Emotion+Irony)│
                 └─────────────┬─────────────┘
                               │
             ┌─────────────────▼─────────────────┐
             │ Streamlit Dashboard (UI + Graphs) │
             └───────────────────────────────────┘

⚙️ METHODOLOGY
1️⃣ Datasets

Emotion Classification → HuggingFace Emotion Dataset (6-class)

Sarcasm Detection → TweetEval - Irony Dataset (binary)

2️⃣ Model Architecture

Base Model → xlm-roberta-base

Fine-tuning Strategy:

Max sequence length: 128

Batch size: 64

Optimizer: AdamW

Learning rate: 2e-5

Epochs: 2–3

3️⃣ Training Pipeline

Tokenization with XLMRobertaTokenizer

Data batching & shuffling

Loss → CrossEntropy

Evaluation → Accuracy & Weighted F1

4️⃣ Deployment

Frontend → Streamlit UI

Backend → Fine-tuned XLM-R models

Features:
✅ Real-time inference
✅ Chat file uploader
✅ Analytics dashboards

🔬 EXPERIMENTS & RESULTS
📌 Emotion Model (6-class)
Metric	Value
Validation Accuracy	~90%
Weighted F1 Score	~0.88
📌 Sarcasm Model (Binary)
Metric	Value
Validation Accuracy	~85%
F1 Score	~0.82

📊 Training Curves:
<img width="1200" height="400" alt="training_metrics" src="https://github.com/user-attachments/assets/2e6de7ae-b7a0-4eeb-92d3-689b306f8fae" />



📊 ANALYTICS

Emotion Distribution → Bar charts of detected emotions

Sarcasm Distribution → Ratio of sarcastic vs non-sarcastic text

Chat Insights → Long-term sentiment/sarcasm trends in conversations

📌 LIMITATIONS

⚠️ Text-only sarcasm detection (no tone/emoji/video context)
⚠️ Requires GPU resources for training
⚠️ Fine-tuned mostly on Twitter data (domain-specific)

🚧 FUTURE WORK

✅ Incorporate multimodal sarcasm detection (emoji, audio, video cues)
✅ Add context-aware conversation modeling
✅ Deploy on HuggingFace Spaces / Streamlit Cloud
✅ Integrate with live messaging apps

🧑‍💻 TECH STACK

Language → Python 3.10+

Framework → Streamlit

Models → HuggingFace Transformers (XLM-RoBERTa)

Visualization → Matplotlib, Pandas

📚 REFERENCES

Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL.

Mohammad, S., et al. (2018). SemEval-2018 Task 1: Affect in Tweets.

Barbieri, F., et al. (2020). TweetEval: Unified Benchmark for Tweet Classification.
