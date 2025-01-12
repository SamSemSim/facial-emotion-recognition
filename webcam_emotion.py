import cv2
import torch
import numpy as np
from torchvision import transforms
from train_model import EmotionCNN  # Import our model architecture

# Load the trained model
model = EmotionCNN()
model.load_state_dict(torch.load('best_emotion_model.pth'))
model.eval()  # Set to evaluation mode

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionary for emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Image transformation (same as test transform in training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_face(face):
    # Resize to match training size
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Convert to PIL Image format
    face_tensor = torch.from_numpy(face).float().unsqueeze(0).unsqueeze(0)
    face_tensor = face_tensor / 255.0  # Normalize to [0,1]
    face_tensor = (face_tensor - 0.5) / 0.5  # Normalize to [-1,1]
    return face_tensor

def detect_emotion(face_tensor):
    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100
        return predicted_class, confidence

# Start video capture
cap = cv2.VideoCapture(0)

print("Starting webcam capture. Press 'q' to quit.")

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # For each face
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess face
        face_tensor = preprocess_face(face_roi)
        
        # Detect emotion
        emotion_idx, confidence = detect_emotion(face_tensor)
        emotion = emotions[emotion_idx]
        
        # Add text with emotion and confidence
        text = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()

print("Webcam capture ended.") 