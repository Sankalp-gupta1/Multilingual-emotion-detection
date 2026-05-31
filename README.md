### Real-Time Multilingual Emotion Detection & Distributed Communication System
# Overview

Modern communication platforms allow users to exchange messages and video streams, but they are unable to understand the emotional context behind conversations. A user may be happy, frustrated, stressed, sarcastic, or confused, yet traditional chat systems treat every message as plain text.

To address this limitation, we developed a real-time multilingual emotion detection and distributed communication platform capable of analyzing both textual and facial emotions. The system combines Natural Language Processing (NLP), Computer Vision, and Deep Learning techniques to provide live emotional insights during communication.

The platform supports multiple connected users, real-time chat, facial emotion recognition through webcam input, sarcasm detection, and synchronized emotional analytics through an interactive dashboard.



## Problem Statement

Most existing communication systems focus only on message delivery and do not understand user emotions.

Some systems perform text analysis while others focus on facial recognition, but very few combine both approaches into a single platform. In addition, traditional systems face challenges such as:

Lack of real-time emotion understanding
No sarcasm detection capability
Limited multilingual support
No synchronized emotional analytics
Webcam conflicts in multi-user environments
Lack of privacy-aware monitoring

These limitations reduce the ability of communication platforms to understand human behavior and emotional context.

## Proposed Solution

The proposed system integrates multilingual text emotion detection and facial emotion recognition into a single distributed platform.

The system continuously analyzes user messages and facial expressions to generate emotional insights in real time. These predictions are synchronized across connected users and visualized through a live analytics dashboard.

# Key capabilities include:

Real-time multilingual emotion detection
Sarcasm identification
Facial emotion recognition
Distributed synchronization
Shared camera management
Live emotional analytics
Privacy-preserving communication
Key Innovation

A major contribution of this project is the modification of the traditional Transformer attention mechanism.

Traditional Transformer models use Dot-Product Attention for semantic comparison between vectors. In this project, Cosine Similarity Attention was introduced to improve multilingual semantic understanding.

Instead of focusing primarily on vector magnitude, cosine similarity measures the directional relationship between vectors, helping the model better understand similar meanings expressed in different languages.

For example:

"I am happy"
"Main khush hu"

Although the words are different, both sentences express similar emotions. Cosine similarity helps bring such semantically related sentences closer during prediction.

## System Features
Text Emotion Detection

The NLP module analyzes user messages and predicts emotions such as:

Joy
Sadness
Anger
Fear
Love
Surprise
Neutral
Sarcasm Detection

The system identifies sarcastic messages that may otherwise be misunderstood by conventional sentiment analysis models.

Example:

Wow, great! I failed my exam.

Although positive words are used, the actual emotional meaning is negative.

## Facial Emotion Recognition

A CNN-based facial emotion recognition model processes webcam frames and predicts facial emotions in real time.

Supported facial emotions include:

Happy
Sad
Angry
Fear
Neutral
Surprise
Disgust
Distributed Multi-User Communication

Multiple users can connect simultaneously through different devices or browsers and participate in synchronized communication.

Privacy-Preserving Shared Camera

To prevent webcam conflicts, only one authenticated user is granted direct camera access while other users receive synchronized video updates.

This approach improves both privacy and resource management.

## Real-Time Analytics Dashboard

The Streamlit dashboard continuously displays:

Active users
Message statistics
Detected emotions
Facial emotion predictions
Camera status
Live analytics
Technology Stack
Technology	Purpose
Python	Core development
Streamlit	Frontend dashboard
FastAPI	Backend APIs
XLM-RoBERTa	Multilingual NLP
Transformer	Text understanding
Cosine Similarity Attention	Semantic comparison
PyTorch	NLP model execution
TensorFlow / Keras	Facial emotion model
CNN	Facial emotion recognition
OpenCV	Webcam processing
JSON	Synchronization
NumPy & Pandas	Data handling
## System Architecture

```text
Users
   │
   ▼
Streamlit Frontend
   │
   ▼
FastAPI Backend
   │
   ├── NLP Module
   │     ├── Emotion Detection
   │     └── Sarcasm Detection
   │
   ├── CNN Facial Emotion Module
   │     └── Webcam Analysis
   │
   ├── Synchronization Layer
   │
   ▼
Live Analytics Dashboard
```
## Live Analytics Dashboard
Workflow
Users connect through the distributed communication platform.
Messages and webcam frames are received in real time.
Text messages are processed by the Transformer-based NLP model.
Webcam frames are processed using OpenCV and CNN.
Emotion predictions are generated.
Results are synchronized across connected users.
Dashboard displays real-time emotional analytics.


<img width="1919" height="964" alt="Screenshot 2026-05-22 125053" src="https://github.com/user-attachments/assets/d319fbb7-0475-4462-ae38-9624e3abe462" />

<img width="1920" height="1080" alt="Screenshot (1807)" src="https://github.com/user-attachments/assets/ed651cae-45cd-42dc-9ca0-b14c73a22826" />

<img width="856" height="574" alt="image" src="https://github.com/user-attachments/assets/f58ba51f-d791-48fb-9763-59a499912f69" />


## Datasets Used

# Text Emotion Dataset

The multilingual emotion dataset was collected from publicly available Hugging Face resources and emotion classification datasets.

# Sarcasm Dataset

 Sarcasm detection data was collected from publicly available irony and sarcasm datasets.

# Facial Emotion Dataset

 Facial emotion datasets were collected from Kaggle sources containing thousands of facial expression images across multiple emotion classes.
 ![Uploading image.png…]()


### Model Training
# NLP Model

The text emotion module was developed using XLM-RoBERTa and trained on multilingual emotion datasets.

The model was fine-tuned to identify emotional context from multilingual conversations and sarcasm-rich text.

## Facial Emotion Model

The facial emotion recognition model was trained using CNN architecture on facial expression datasets.

Images were preprocessed through resizing, normalization, and augmentation before training.

## Results

The system successfully performs:

Real-time emotion detection
Sarcasm detection
Facial emotion recognition
Multi-user synchronization
Live emotional analytics

The platform demonstrates stable performance in distributed communication environments and provides meaningful emotional insights during conversations.

## Applications

This project can be used in:

Online interview monitoring
Smart classroom environments
Mental health support systems
Customer behavior analysis
Emotion-aware chat platforms
AI-assisted communication systems
Virtual meeting analytics
Future Scope
![Uploading image.png…]()


## Future improvements may include:

Voice emotion recognition
Speech tone analysis
Mobile application support
Cloud deployment
Advanced emotional analytics
Federated learning integration
End-to-end encrypted communication
Team Members
![Uploading image.png…]()


Sankalp Gupta
AI/ML Development, Transformer Modification, Backend Integration

Suryansh Sharma
Frontend Development, Dashboard Design, Synchronization

Pranshu Yadav
Facial Emotion Recognition, CNN Integration, Webcam Processing

## Conclusion

This project presents a practical implementation of an emotion-aware distributed communication platform that combines Natural Language Processing and Computer Vision techniques.

By integrating multilingual emotion detection, sarcasm recognition, facial emotion analysis, and synchronized real-time analytics, the system moves beyond traditional communication platforms and provides a more intelligent understanding of human interactions.

