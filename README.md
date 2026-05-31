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


Features
Real-time multilingual text emotion detection
Sarcasm detection from chat messages
Facial emotion recognition using webcam
CNN-based face emotion prediction
Transformer-based NLP emotion module
Cosine Similarity Attention mechanism
FastAPI backend for AI model serving
Streamlit frontend dashboard
Multi-user distributed chat system
Shared camera synchronization
Privacy-preserving webcam access
Live emotional analytics dashboard
JSON-based synchronization
Real-time user tracking
Tech Stack
Technology	Purpose
Python	Core programming language
Streamlit	Frontend dashboard
FastAPI	Backend API communication
XLM-RoBERTa	Multilingual text emotion model
Transformer	Text understanding
Cosine Similarity	Semantic attention comparison
CNN	Facial emotion recognition
OpenCV	Webcam and face detection
TensorFlow / Keras	Deep learning model execution
PyTorch	Transformer model support
JSON	Multi-user synchronization
NumPy / Pandas	Data processing
System Architecture
User Input
   |
   |-- Text Message
   |-- Webcam Frame
   |
Streamlit Frontend
   |
FastAPI Backend
   |
   |-- Transformer NLP Model
   |      |-- Emotion Detection
   |      |-- Sarcasm Detection
   |
   |-- CNN Facial Emotion Model
          |-- Face Detection
          |-- Facial Emotion Prediction
   |
Synchronization Layer
   |
Live Dashboard + Multi-User Analytics
Internal Workflow
User sends a message or activates webcam.
Streamlit frontend sends data to FastAPI backend.
Text messages are processed by the Transformer NLP model.
Webcam frames are processed using OpenCV and CNN.
Emotion predictions are generated in real time.
Backend synchronizes results across connected users.
Dashboard displays live emotional analytics.
Text Emotion Detection Module

The text emotion module analyzes chat messages and predicts emotional meaning.

Working Steps
User Message
   ↓
Text Preprocessing
   ↓
Tokenization
   ↓
Transformer Model
   ↓
Cosine Similarity Attention
   ↓
Emotion Prediction
Output Emotions
Joy
Anger
Sadness
Fear
Love
Neutral
Surprise
Cosine Transformer Attention

Traditional Transformer models use dot-product attention.

In this project, cosine similarity is used to compare semantic meaning between vectors.

Formula
Cosine(Q, K) = (Q · K) / (||Q|| × ||K||)
Why Cosine Similarity?
Better multilingual meaning comparison
Handles Hindi-English mixed text
Helps similar meaning sentences stay closer
Improves sarcasm and emotional context understanding
Reduces dependency on vector magnitude
Facial Emotion Detection Module

The facial emotion module detects emotions from webcam input.

Working Steps
Webcam Frame
   ↓
OpenCV Face Detection
   ↓
Face Crop & Resize
   ↓
CNN Model
   ↓
Facial Emotion Prediction
   ↓
Dashboard Display
Facial Emotion Classes
Happy
Sad
Angry
Fear
Neutral
Surprise
Disgust
Privacy-Preserving Shared Camera System

A major feature of this project is shared camera synchronization.

Problem

If multiple users try to access the webcam at the same time, hardware conflict can occur.

Solution
Only one authenticated user gets direct camera access.
Other users receive synchronized shared video stream.
Backend controls camera allocation.
Emotional analytics are shared without giving direct camera access to all users.
Benefits
Prevents webcam conflict
Improves privacy
Supports multi-user monitoring
Keeps synchronization stable
Model Training
Text Emotion Model
Dataset collected from Hugging Face
Transformer model used for multilingual text understanding
Emotion and sarcasm datasets used
Text preprocessing performed before training
Facial Emotion Model
Dataset collected from Kaggle facial emotion datasets
CNN model trained on facial expression images
Image preprocessing included resizing, normalization, and face extraction
Results
Module	Approximate Performance
Text Emotion Detection	78%
Sarcasm Detection	Around 80%
Facial Emotion Detection	Around 62%

Facial emotion accuracy is comparatively lower because real-time webcam prediction depends on:

Lighting condition
Face angle
Camera quality
Live movement
Expression clarity
Project Folder Structure
emotion-detection-whatsap/
│
├── chat_client/
├── chat_server/
├── data/
├── facial_emotion/
│   └── face_model/
│       └── face_emotion_model_v2.keras
│
├── model_service/
│   ├── fastapi_model_service.py
│   ├── models/
│   └── saved_emotion_model/
│
├── web_ui/
│   ├── app.py
│   ├── webcam.py
│   ├── camera_manager.py
│   ├── global_chat.json
│   ├── online_users.json
│   ├── shared_camera.jpg
│   └── shared_emotion.json
│
├── requirements.txt
└── README.md
Installation
1. Clone the Repository
git clone https://github.com/your-username/your-repository-name.git
cd emotion-detection-whatsap
2. Create Virtual Environment
python -m venv clean_env
3. Activate Environment

For Windows:

.\clean_env\Scripts\activate

If PowerShell gives execution policy error:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\clean_env\Scripts\activate
Install Dependencies
python -m pip install --upgrade pip
python -m pip install streamlit fastapi uvicorn transformers tensorflow keras opencv-python numpy pandas scikit-learn torch torchvision torchaudio pillow requests python-multipart sentencepiece tiktoken
Run the Project
Terminal 1: Run FastAPI Backend
cd model_service
python -m uvicorn fastapi_model_service:app --reload --host 0.0.0.0 --port 8000

Backend will run on:

http://127.0.0.1:8000
Terminal 2: Run Streamlit Frontend
cd web_ui
streamlit run app.py --server.address 0.0.0.0

Frontend will run on:

http://localhost:8501

For mobile access on same WiFi, use:

http://YOUR_IPV4_ADDRESS:8501

Example:

http://192.168.29.223:8501
How to Find IPv4 Address

Run:

ipconfig

Use the IPv4 address shown under Wi-Fi adapter.

Example:

IPv4 Address: 192.168.29.223

Then open on mobile:

http://192.168.29.223:8501
Important Firewall Note

If mobile device cannot open the Streamlit URL, allow port 8501 and 8000 in firewall.

Temporary testing command:

netsh advfirewall set allprofiles state off

Turn firewall back on after testing:

netsh advfirewall set allprofiles state on
API Flow
Frontend Request
   ↓
FastAPI Endpoint
   ↓
AI Model Processing
   ↓
JSON Response
   ↓
Streamlit Dashboard Update

Example API use:

User Message → FastAPI → Transformer Model → Emotion Result → Frontend
Main Modules
1. Frontend Module

Built using Streamlit.

Handles:

User interface
Chat input
Webcam display
Emotion dashboard
Live analytics
2. Backend Module

Built using FastAPI.

Handles:

API requests
AI model loading
Emotion prediction
Synchronization response
3. NLP Module

Uses Transformer model.

Handles:

Multilingual emotion detection
Sarcasm detection
Cosine similarity based semantic comparison
4. Facial Emotion Module

Uses CNN and OpenCV.

Handles:

Webcam frame capture
Face detection
Emotion classification
5. Synchronization Module

Uses JSON storage.

Handles:

Online users
Shared chat
Shared webcam frame
Emotion analytics
Camera status
Applications
Online interview monitoring
Smart classrooms
Emotion-aware chat systems
AI-based customer support
Mental health monitoring
Virtual meeting analytics
Human-computer interaction research
AI surveillance and monitoring systems
Future Enhancements
Voice emotion detection
Speech tone analysis
Mobile application support
Cloud deployment
GPU acceleration
Federated learning
Advanced analytics dashboard
Encrypted communication
Biometric authentication
AI-based recommendation system
Challenges Faced
Real-time webcam synchronization
Multi-user shared camera control
CPU-based processing limitations
Facial emotion prediction under low lighting
API response handling
Frontend-backend synchronization
Model compatibility issues during deployment
Team Members
Sankalp Gupta
Suryansh Sharma
Pranshu Yadav
Guide

Er. Shesh Mani Tiwari
Department of Computer Science
CSJM University, Kanpur

Conclusion

This project demonstrates a real-time emotion-aware distributed communication system that combines NLP, Computer Vision, Deep Learning, and synchronization techniques.

The system can detect emotions from both text and facial expressions, synchronize results across multiple users, and display live emotional analytics through an interactive dashboard.

It provides a practical foundation for future intelligent communication systems where emotional understanding plays an important role.

License

This project is developed for academic and learning purposes.



```text

I am happy
Main khush hu
