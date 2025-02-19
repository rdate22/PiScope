import cv2
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch
from torchvision import models, transforms
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained MobileNetV2 model
mobilenet_v2 = models.mobilenet_v2(pretrained=True).to(device)
mobilenet_v2.eval()

# Input image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Extract face embedding
def get_face_embedding(face_image):
    # Transform image for MobileNetV2 format
    face_image = transform(face_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = mobilenet_v2.features(face_image)  # Feature map from the model
        embedding = embedding.view(embedding.size(0), -1)  # Flatten feature map
    return embedding.cpu().numpy().flatten()

# Dataset format: dataset/personfolder/imgs
data_dir = 'dataset' 
faces = []
person_names = []

# Loop through each person's folder
for person_folder in os.listdir(data_dir):
    person_folder_path = os.path.join(data_dir, person_folder)
    
    if os.path.isdir(person_folder_path):
        # Load all images of the person
        for img_name in os.listdir(person_folder_path):
            img_path = os.path.join(person_folder_path, img_name)
            img = cv2.imread(img_path)
            
            # Skip non-image files (.DS_Store)
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces_rects = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces_rects:
                # Crop the face region
                face_image = img[y:y+h, x:x+w]
                
                # Extract the embedding
                embedding = get_face_embedding(face_image)
                faces.append(embedding)
                person_names.append(person_folder)

# Convert lists to numpy arrays for easier handling with k-fold
faces = np.array(faces)
person_names = np.array(person_names)

# Set up k-fold cross-validation
kf = KFold(n_splits=len(faces) - 1, shuffle=True, random_state=42)  # 5-fold cross-validation
accuracies = []

# Perform cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(faces)):
    print(f"Fold {fold+1}")
    
    # Split the data into train and test sets for this fold
    X_train, X_test = faces[train_index], faces[test_index]
    y_train, y_test = person_names[train_index], person_names[test_index]

    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    rf_classifier.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = rf_classifier.predict(X_test)
    
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Accuracy for fold {fold+1}: {accuracy:.4f}")

# Compute the average accuracy across all folds
average_accuracy = np.mean(accuracies)
print(f"\nAverage accuracy across all folds: {average_accuracy:.4f}")

# Optionally, save the model after training on all folds
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

# Save the embeddings and names (not needed yet)
with open('face_embeddings.pkl', 'wb') as file:
    pickle.dump((faces, person_names), file)

print("Cross-validation complete, Random Forest model saved.")
