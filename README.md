 🧠 Mental Health Mood Prediction System

An AI-powered system that predicts a user's emotional state from text and speech input using Deep Learning and NLP models.  
Built using PyTorch and HuggingFace Transformers.

## 🚀 Features

- Mood prediction from text input
- Voice-to-text conversion using speech recognition
- Pretrained Hugging Face model integration
- Real-time sentiment analysis
- Modular and scalable project structure

## 🛠️ Tech Stack

- Python
- PyTorch
- Transformers (HuggingFace)
- BERT (bert-base-uncased)
- SpeechRecognition

📂 Project Structure

MentalHealthPrediction/
│
├── src/
│   ├── emotion_prediction_textbased.py
│   ├──mood_prediction_voicebased.py
│
├── data/
├── requirements.txt
├── README.md  

## 🧠 Model Details

- Model: nateraw/bert-base-uncased-emotion
- Architecture: BERT (bert-base-uncased)
- Task: Multi-class Emotion Classification
- Framework: PyTorch
- Library: Hugging Face Transformers
- Technique: Transfer Learning
  
## ⚙️ Installation
 Install dependencies:
   pip install -r requirements.txt

## ▶️ Usage

Run text-based prediction:
   emotion_prediction_textbased.py

Run voice-based prediction:
  mood_prediction_voicebased.py    
