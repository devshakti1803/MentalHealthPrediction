import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load pre-trained model & tokenizer from HuggingFace
model_name = "nateraw/bert-base-uncased-emotion"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Build emotion detection pipeline
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Function to predict emotion
def detect_emotion(text):
    result = emotion_classifier(text)[0]  # Get first result (list of dicts)
    result = sorted(result, key=lambda x: x['score'], reverse=True)
    for r in result:
        print(f"{r['label']}: {r['score']:.4f}")
    return result[0]['label']

# Example usage
user_input = input("Enter your message: ")
predicted_emotion = detect_emotion(user_input)
print(f"Predicted Emotion: {predicted_emotion}")
