import cv2
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
import torch.nn.functional as F

# Define the MLP model class (same as in your training script)
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=None, dropout=0.5):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Load the pre-trained MobileNetV2 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mobilenet_v2 = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device)
mobilenet_v2.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize the face to the expected size of MobileNetV2
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the label encoder to decode predicted labels
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the trained MLP model architecture and state_dict
input_size = 62720  # This corresponds to the flattened output size of the MobileNetV2 feature map (1280*7*7)
output_size = len(label_encoder.classes_)  # Number of classes in the dataset
mlp_model = MLPModel(input_size=input_size, hidden_size=64, output_size=output_size).to(device)

# Load the saved weights into the model
mlp_model.load_state_dict(torch.load('best_mlp_face_recognition_model.pth', map_location=device))

# Set the model to evaluation mode
mlp_model.eval()

# Confidence threshold for "Unknown"
confidence_threshold = 0.2  # Adjust this threshold as needed

# Function to extract face embeddings using MobileNetV2 in PyTorch
def get_face_embedding(face_image):
    face_image = transform(face_image).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension
    with torch.no_grad():
        embedding = mobilenet_v2.features(face_image)  # Extract features
        embedding = embedding.view(embedding.size(0), -1)  # Flatten feature map
    return embedding.flatten().cpu().numpy()  # Return as numpy array

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces_rects = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_rects:
        face_image = frame[y:y+h, x:x+w]

        # Ensure face is resized to (224, 224) and processed
        face_embedding = get_face_embedding(face_image)

        # Convert embedding to tensor for the MLP model
        embedding_tensor = torch.tensor(face_embedding, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict using the trained MLP model
        with torch.no_grad():
            outputs = mlp_model(embedding_tensor)
            predicted_probs = F.softmax(outputs, dim=1)  # Get probabilities from softmax
            predicted_label = torch.argmax(predicted_probs, dim=1).item()  # Get the predicted class label
            confidence = predicted_probs[0, predicted_label].item()  # Maximum probability (confidence)

        # Decode the label to the corresponding person's name
        if confidence < confidence_threshold:
            label = "Unknown"
        else:
            label = label_encoder.inverse_transform([predicted_label])[0]
        
        # Draw a bounding box around the face and show the predicted label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
