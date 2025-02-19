import cv2
import numpy as np
import pickle
import torch
from torchvision import models, transforms
# import time # for timing predict time

# Load the pre-trained MobileNetV2 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mobilenet_v2 = models.mobilenet_v2(pretrained=True).to(device)
mobilenet_v2.eval()

# Input image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load trained Random Forest model
with open('rf_model.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

# Confidence threshold for "Unknown"
confidence_threshold = 0.2

# Copied function from train
def get_face_embedding(face_image):
    face_image = transform(face_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = mobilenet_v2.features(face_image)
        embedding = embedding.view(embedding.size(0), -1)
    return embedding.cpu().numpy().flatten()

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Measure the time prediction
    # start_time = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces_rects = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_rects:
        face_image = frame[y:y+h, x:x+w]
        embedding = get_face_embedding(face_image)

        # Predict using the trained Random Forest model
        predicted_label = rf_classifier.predict([embedding])
        predicted_probabilities = rf_classifier.predict_proba([embedding])  # Get class probabilities

        # Confidence: Maximum probability for the predicted class
        confidence = np.max(predicted_probabilities)

        # Classify unknown faces
        if confidence < confidence_threshold:
            label = "Unknown"
        else:
            label = predicted_label[0]
        
        # Draw a bounding box around the face and show the predicted label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # end_time = time.time()  # End timing
    # inference_time = end_time - start_time  # Calculate inference time
    # Print the inference time for this frame
    # print(f"Inference time: {inference_time:.4f} seconds")

    cv2.imshow("Real-Time Face Recognition", frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
