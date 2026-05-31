# Real-Time Multilingual Emotion Detection & Privacy-Preserving Distributed Chat System

A real-time AI-based communication platform that detects emotions from multilingual chat messages and facial expressions.  
The system combines NLP, Computer Vision, Transformer-based emotion detection, CNN facial emotion recognition, FastAPI backend, Streamlit frontend, and distributed synchronization.

---

## Project Overview

This project is designed to make online communication more emotion-aware.

Normal chat systems only transfer messages, but they do not understand whether a user is happy, sad, angry, confused, or sarcastic.  
This system analyzes both:

- Text messages
- Facial expressions through webcam

and displays live emotional analytics in a distributed multi-user dashboard.

---

## Key Highlight

The main technical highlight of this project is the use of a Transformer-based multilingual emotion detection model with Cosine Similarity Attention.

Instead of relying only on traditional dot-product based attention, the system uses cosine similarity logic to improve semantic understanding between multilingual sentences.

Example:

```text
I am happy
Main khush hu
