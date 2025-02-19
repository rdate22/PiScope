import dlib
import cv2
import numpy as np
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier 

# Load the pre-trained face detector and face recognition model from dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download from dlib's model repo
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')  # Download from dlib's model repo

# Function to extract face embedding using dlib
def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)  # Detect faces
    
    if len(faces) == 0:
        return None

    embeddings = []
    for face in faces:
        landmarks = sp(gray, face)  # Detect landmarks
        face_descriptor = face_rec_model.compute_face_descriptor(image, landmarks)
        embeddings.append(np.array(face_descriptor))
    
    return embeddings

# Directory where the dataset is stored (organized by person subdirectories)
data_dir = 'dataset'  # Folder containing subdirectories for each person
faces = []
person_names = []  # Store names of people (instead of encoded labels)

# Loop through each person's folder (subdirectory) to collect images and labels
for person_folder in os.listdir(data_dir):
    person_folder_path = os.path.join(data_dir, person_folder)
    
    if os.path.isdir(person_folder_path):
        # Load all images of the person
        for img_name in os.listdir(person_folder_path):
            img_path = os.path.join(person_folder_path, img_name)
            img = cv2.imread(img_path)
            
            # Skip non-image files
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue

            embeddings = get_face_embedding(img)  # Get face embeddings for the image

            if embeddings:
                for embedding in embeddings:
                    faces.append(embedding)
                    person_names.append(person_folder)  # Label the embedding with the person's name

# Train a KNN classifier on the face embeddings
knn = KNeighborsClassifier(n_neighbors=1)  # k=1 for nearest neighbor classifier
knn.fit(faces, person_names)

# Save the trained KNN model for future use
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

# Save the face embeddings and names for future inference
with open('face_embeddings.pkl', 'wb') as file:
    pickle.dump((faces, person_names), file)

print("Training complete. KNN model and embeddings saved.")