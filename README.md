🤖 Multilingual Emotion & Sarcasm Detection
📖 Abstract

Understanding human communication requires more than just literal text analysis. Emotions and sarcasm are crucial components of natural language but are challenging for machines to interpret, especially in multilingual settings.

This project presents a Streamlit-based interactive system that integrates fine-tuned transformer models (XLM-RoBERTa) for multilingual emotion classification and sarcasm detection. The system is capable of real-time inference, as well as bulk analysis of WhatsApp/Telegram chats, with results visualized through analytics dashboards.

🧩 Motivation

Ambiguity in Language: Phrases like “Wow, great job 👏” can convey genuine praise or sarcasm depending on context.

Social Media Complexity: Platforms like WhatsApp, Twitter, and Telegram are rich in emotional & sarcastic communication.

Multilingual Challenge: Existing systems are often limited to English; our approach leverages XLM-RoBERTa for multilingual generalization.

🎯 Objectives

Build a robust emotion classifier capable of detecting six emotions: Joy, Love, Anger, Fear, Sadness, Surprise.

Develop a sarcasm detection system trained on irony datasets.

Provide a user-friendly interface for both single-message predictions and chat-level analytics.

Demonstrate the effectiveness of transfer learning using pre-trained multilingual transformers.

📚 Related Work

Emotion Detection: Traditional methods (SVMs, Naïve Bayes) struggle with multilinguality. Transformers like BERT and XLM-RoBERTa have shown state-of-the-art results on benchmark datasets.

Sarcasm Detection: Prior works highlight the difficulty of sarcasm, often requiring contextual or multimodal signals. We focus on text-based sarcasm detection using Twitter’s TweetEval - Irony Dataset.

Applications: Sentiment analysis, mental health monitoring, conversational AI, social media moderation.


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
⚙️ Methodology
1. Datasets

Emotion Classification: HuggingFace Emotion Dataset
 (6-class).

Sarcasm Detection: HuggingFace TweetEval - Irony
 (binary).

2. Model Architecture

Base Model: xlm-roberta-base

Fine-tuning Strategy:

Max sequence length: 128

Batch size: 64

Optimizer: AdamW

Learning rate: 2e-5

Epochs: 2–3

3. Training Pipeline

Tokenization (XLMRobertaTokenizer)

DataLoader with batching & shuffling

Loss minimization (CrossEntropy)

Evaluation metrics: Accuracy & Weighted F1

4. Deployment

Streamlit provides interactive front-end

Real-time inference with cached model loading

File uploader for chat analysis

🔬 Experiments & Results
Emotion Model (6-class)
Metric	Value
Validation Accuracy	~90%
Weighted F1 Score	~0.88
Sarcasm Model (Binary)
Metric	Value
Validation Accuracy	~85%
F1 Score	~0.82

📊 Training curves available in training_metrics.png.
![Metrics](https://raw.githubusercontent.com/username/repo/main/assets/training_metrics.png)
