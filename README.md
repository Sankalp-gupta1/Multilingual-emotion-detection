#🤖 Multilingual Emotion & Sarcasm Detection


##📖 Abstract

Understanding human communication requires more than just literal text analysis. Emotions and sarcasm are crucial components of natural language but are challenging for machines to interpret, especially in multilingual settings.

This project presents a Streamlit-based interactive system that integrates fine-tuned transformer models (XLM-RoBERTa) for multilingual emotion classification and sarcasm detection. The system supports real-time inference, as well as bulk analysis of WhatsApp/Telegram chats, with results visualized through analytics dashboards.

##🧩 Motivation

Ambiguity in Language

Example: “Wow, great job 👏” can either be genuine praise or sarcasm depending on context.

Social Media Complexity

Platforms like WhatsApp, Twitter, and Telegram contain highly informal and emotionally nuanced text.

Multilingual Challenge

Existing emotion/sarcasm systems are often restricted to English.

Our approach leverages XLM-RoBERTa for cross-lingual generalization.

##🎯 Objectives

Build a robust emotion classifier capable of detecting six emotions: Joy, Love, Anger, Fear, Sadness, Surprise.

Develop a sarcasm detection model trained on irony datasets.

Provide a user-friendly interface for both single-message predictions and chat-level analytics.

Demonstrate the effectiveness of transfer learning using pre-trained multilingual transformers.

##📚 Related Work

Emotion Detection

Traditional methods (SVMs, Naïve Bayes) are weak in multilingual settings.

Transformers like BERT and XLM-RoBERTa achieve state-of-the-art results.

Sarcasm Detection

Prior research highlights sarcasm as context-heavy, often requiring multimodal signals.

This project focuses on text-based sarcasm detection using TweetEval - Irony Dataset.

##Applications

Sentiment analysis

Mental health monitoring

Conversational AI

Social media moderation


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


##⚙️ Methodology
1. Datasets

Emotion Classification → HuggingFace Emotion Dataset
 (6-class).

Sarcasm Detection → TweetEval - Irony Dataset
 (binary).

##2. Model Architecture

Base Model: xlm-roberta-base

Fine-tuning Strategy:

Max sequence length: 128

Batch size: 64

Optimizer: AdamW

Learning rate: 2e-5

Epochs: 2–3

##3. Training Pipeline

Tokenization with XLMRobertaTokenizer

Data batching & shuffling

CrossEntropy loss optimization

Evaluation via Accuracy & Weighted F1

##4. Deployment

Frontend: Streamlit UI

Backend: Fine-tuned XLM-R models

##Features:

Real-time inference

Chat file uploader

Analytics dashboards

##🔬 Experiments & Results
📌 Emotion Model (6-class)
Metric	Value
Validation Accuracy	~90%
Weighted F1 Score	~0.88
##📌 Sarcasm Model (Binary)
Metric	Value
Validation Accuracy	~85%
F1 Score	~0.82

📊 Training curves:
<<img width="1200" height="400" alt="training_metrics" src="https://github.com/user-attachments/assets/20d5c6e1-e334-4742-8105-3cb85043ffeb" />
 />



##📊 Analytics

Emotion Distribution → Bar charts of detected emotions.

Sarcasm Distribution → Ratio of sarcastic vs non-sarcastic text.

Chat Insights → Long-term sentiment/sarcasm trends in conversations.

##📌 Limitations

Text-only sarcasm detection (no tone/emoji/video context).

Training requires GPU resources.

Fine-tuned mostly on Twitter data (domain-specific).

##🚧 Future Work

✅ Incorporate multimodal sarcasm detection (emoji, audio, video cues).

✅ Add context-aware conversation modeling.

✅ Deploy on HuggingFace Spaces / Streamlit Cloud.

✅ Integrate with live messaging apps.

##🧑‍💻 Tech Stack

Language: Python 3.10+

Framework: Streamlit

Models: HuggingFace Transformers (XLM-RoBERTa)

Visualization: Matplotlib, Pandas

#📚 References

Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. ACL.

Mohammad, S., et al. (2018). SemEval-2018 Task 1: Affect in Tweets.

Barbieri, F., et al. (2020). TweetEval: Unified Benchmark for Tweet Classification.
