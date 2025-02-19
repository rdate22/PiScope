import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import os
import cv2
import pickle
from torchvision import models, transforms
from torch.autograd import Variable

hyperparameters = {
    'lr': [0.001, 0.0001],
    'batch_size': [32, 64],
    'hidden_size': [64, 128],
    'dropout': [0.4, 0.5]
}

# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=None, dropout=0.5):
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

# Function for training the model
def train_model(X_train, y_train, model, criterion, optimizer, epochs=30):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

def evaluate(model, X_val_tensor, y_val):
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_val, predicted.cpu().numpy())
    return accuracy

# Cross-validation function
def cross_validate(hyperparams, X, y, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    best_model = None
    best_accuracy = 0
    
    # Try each combination of hyperparameters
    for lr in hyperparams['lr']:
        for batch_size in hyperparams['batch_size']:
            for hidden_size in hyperparams['hidden_size']:
                for dropout in hyperparams['dropout']:
                    print(f"Training with lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, dropout={dropout}")
                    fold_accuracies = []
                    
                    for train_index, val_index in kfold.split(X):
                        # Split the dataset
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]
                        
                        # Convert to torch tensors
                        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
                        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
                        
                        # Create the model and optimizer
                        model = MLPModel(input_size=X_train_tensor.shape[1], hidden_size=hidden_size, 
                                         output_size=len(np.unique(y)), dropout=dropout)
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        criterion = nn.CrossEntropyLoss()
                        
                        # Train the model
                        model = train_model(X_train_tensor, y_train_tensor, model, criterion, optimizer, epochs=10)
                        
                        # Evaluate the model
                        model.eval()
                        with torch.no_grad():
                            outputs = model(X_val_tensor)
                            _, predicted = torch.max(outputs, 1)
                            accuracy = accuracy_score(y_val, predicted.numpy())
                            fold_accuracies.append(accuracy)
                    
                    # Compute average accuracy for this combination of hyperparameters
                    avg_accuracy = np.mean(fold_accuracies)
                    print(f"Avg Accuracy: {avg_accuracy:.4f}")
                    
                    # Save the best model based on accuracy
                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy
                        best_model = model
                        best_hyperparams = {
                            'lr': lr,
                            'batch_size': batch_size,
                            'hidden_size': hidden_size,
                            'dropout': dropout
                        }
    
    return best_model, best_hyperparams, best_accuracy

# Prepare dataset (images organized in folders by person)
data_dir = 'dataset'  # Folder containing subdirectories for each person
faces = []
person_names = []  # Store names of people (instead of encoded labels)

# Load the pre-trained MobileNetV2 model
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
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

# Function to extract face embedding using MobileNetV2 in PyTorch
def get_face_embedding(face_image):
    face_image = transform(face_image).unsqueeze(0)  # Convert to tensor and add batch dimension
    with torch.no_grad():
        embedding = mobilenet_v2.features(face_image)  # Extract features
        embedding = embedding.view(embedding.size(0), -1)  # Flatten feature map
    return embedding.flatten().cpu().numpy()  # Return as numpy array

# Loop through each person's folder (subdirectory)
for person_folder in os.listdir(data_dir):
    person_folder_path = os.path.join(data_dir, person_folder)
    if os.path.isdir(person_folder_path):
        # Load all images of the person
        for img_name in os.listdir(person_folder_path):
            img_path = os.path.join(person_folder_path, img_name)
            img = cv2.imread(img_path)
            
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image using Haar Cascade
            faces_rects = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces_rects:
                face_image = img[y:y+h, x:x+w]
                
                # Extract the embedding of the face
                embedding = get_face_embedding(face_image)
                faces.append(embedding)
                person_names.append(person_folder)

# Encode the person names to numeric labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(person_names)

# Convert faces data to numpy array for cross-validation
faces_numpy = np.array(faces)
encoded_labels_numpy = np.array(encoded_labels)

# Run cross-validation and hyperparameter optimization (LOO CV)
best_model, best_hyperparams, best_accuracy = cross_validate(hyperparameters, faces_numpy, encoded_labels_numpy, len(faces_numpy) - 1)

# Output best hyperparameters and model
print(f"Best Hyperparameters: {best_hyperparams}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Save the best model
torch.save(best_model.state_dict(), 'best_mlp_face_recognition_model.pth')
print("Best model saved.")

# Optionally, save label encoder for future use
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)