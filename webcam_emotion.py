import cv2
import torch
import numpy as np
from torchvision import transforms
from train_model import EmotionCNN, IMG_HEIGHT, IMG_WIDTH  # Import constants from training
import sys
import traceback
import os
import urllib.request

print("Starting application...")
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load the trained model
try:
    print("Initializing model...")
    device = torch.device('cpu')
    model = EmotionCNN()  # Fixed typo
    print("Loading model weights...")
    try:
        state_dict = torch.load('best_emotion_model.pth', map_location=device, weights_only=True)  # Added weights_only
        print("Model state loaded successfully")
    except Exception as e:
        print(f"Error loading model file: {str(e)}")
        raise
    
    try:
        model.load_state_dict(state_dict)
        print("State dict loaded into model")
    except Exception as e:
        print(f"Error loading state dict into model: {str(e)}")
        raise
        
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Traceback:")
    traceback.print_exc()
    raise

# Download cascade file if not exists
cascade_file = 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_file):
    print("Downloading face cascade classifier...")
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(url, cascade_file)
    print("Downloaded cascade file successfully")

# Load the face cascade classifier
try:
    print("Loading face cascade classifier...")
    face_cascade = cv2.CascadeClassifier(cascade_file)
    if face_cascade.empty():
        raise Exception("Error: Cascade classifier not loaded properly")
    print("Face cascade classifier loaded successfully")
except Exception as e:
    print(f"Error loading face cascade: {str(e)}")
    raise

# Dictionary for emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Image transformation (same as test transform in training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_face(face):
    try:
        # Convert to grayscale first
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Ensure correct size
        face = cv2.resize(face, (IMG_HEIGHT, IMG_WIDTH))
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face = clahe.apply(face)
        
        # Add Gaussian blur to reduce noise
        face = cv2.GaussianBlur(face, (3, 3), 0)
        
        # Convert to tensor and normalize
        face_tensor = torch.from_numpy(face).float().unsqueeze(0).unsqueeze(0)
        face_tensor = face_tensor / 255.0  # Normalize to [0,1]
        face_tensor = (face_tensor - 0.5) / 0.5  # Normalize to [-1,1]
        return face_tensor
    except Exception as e:
        print(f"Error in preprocess_face: {str(e)}")
        raise

def detect_emotion(face_tensor):
    try:
        with torch.no_grad():
            face_tensor = face_tensor.to(device)
            outputs = model(face_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get probabilities for all emotions
            probs = probabilities[0].cpu().numpy()
            
            # Print all probabilities for debugging
            print("\nEmotion Probabilities:")
            for emotion, prob in zip(emotions, probs):
                print(f"{emotion}: {prob*100:.1f}%")
            print("-" * 30)
            
            # Apply some rules for more balanced predictions
            max_prob = np.max(probs)
            
            # If no emotion is very confident, return neutral
            if max_prob < 0.4:
                return emotions.index('Neutral'), max_prob * 100, probs
            
            # Get top 2 emotions and their probabilities
            top2_idx = np.argsort(probs)[-2:][::-1]
            top2_probs = probs[top2_idx]
            
            # If the difference between top 2 emotions is small, prefer more common emotions
            if (top2_probs[0] - top2_probs[1]) < 0.15:
                # Bias towards more common emotions (Happy, Neutral, Surprise)
                common_emotions = ['Happy', 'Neutral', 'Surprise']
                for emotion in common_emotions:
                    idx = emotions.index(emotion)
                    if probs[idx] > max_prob * 0.7:  # If probability is close enough to max
                        return idx, probs[idx] * 100, probs
            
            # Return the highest probability emotion
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class] * 100
            
            return predicted_class, confidence, probs
    except Exception as e:
        print(f"Error in detect_emotion: {str(e)}")
        raise

# Start video capture
print("Initializing video capture...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera opened successfully")
print("Starting webcam capture. Press 'q' to quit.")

try:
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create debug info window
        debug_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        y_offset = 30
        cv2.putText(debug_frame, "Debug Information:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detect faces with adjusted parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=8,
            minSize=(60, 60),
            maxSize=(400, 400)
        )
        
        # For each face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face ROI with some margin
            margin = 20
            y1 = max(y - margin, 0)
            y2 = min(y + h + margin, frame.shape[0])
            x1 = max(x - margin, 0)
            x2 = min(x + w + margin, frame.shape[1])
            face_roi = frame[y1:y2, x1:x2]
            
            # Preprocess face
            face_tensor = preprocess_face(face_roi)
            
            # Detect emotion
            emotion_idx, confidence, all_probs = detect_emotion(face_tensor)
            emotion = emotions[emotion_idx]
            
            # Add text with emotion and confidence
            text = f"{emotion} ({confidence:.1f}%)"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Update debug window with probabilities
            y_offset = 60
            for emotion, prob in zip(emotions, all_probs):
                text = f"{emotion}: {prob*100:.1f}%"
                cv2.putText(debug_frame, text, (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 30
        
        # Display both frames
        cv2.imshow('Emotion Detection', frame)
        cv2.imshow('Debug Info', debug_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit key pressed")
            break

except Exception as e:
    print(f"Error in main loop: {str(e)}")
finally:
    # Release everything
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam capture ended.") 