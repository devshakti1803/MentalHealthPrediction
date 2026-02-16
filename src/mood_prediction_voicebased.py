import speech_recognition as sr
from transformers import pipeline

# Initialize recognizer
recognizer = sr.Recognizer()

def record_voice(prompt):
    """Record voice from microphone and convert to text"""
    with sr.Microphone() as source:
        print(prompt)
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("Recognized:", text)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "API unavailable"

# Step 1: Take voice inputs
patient_text = record_voice("🎤 Speak now (Patient): ")
doctor_text = record_voice("🎤 Speak now (Doctor): ")

# Step 2: Combine inputs
conversation = f"Patient: {patient_text}\nDoctor: {doctor_text}"
print("\nConversation:\n", conversation)

# Step 3: Prediction model (using HuggingFace pipeline)
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

result = classifier(conversation)
print("\nPredicted Output:", result)