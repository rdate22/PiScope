import dlib
import cv2
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree

# Load the pre-trained face detector and face recognition model from dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Function to extract face embedding using dlib
def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)  # Detect faces
    
    if len(faces) == 0:
        return None

    embeddings = []
    for face in faces:
        landmarks = sp(gray, face)
        face_descriptor = face_rec_model.compute_face_descriptor(image, landmarks)
        embeddings.append(np.array(face_descriptor))
    
    return embeddings

# Directory where the dataset is stored
data_dir = 'Dataset'
faces = []
person_names = []

# Loop through each person's folder (subdirectory) to collect images and labels
for person_folder in os.listdir(data_dir):
    person_folder_path = os.path.join(data_dir, person_folder)
    
    if os.path.isdir(person_folder_path):
        for img_name in os.listdir(person_folder_path):
            img_path = os.path.join(person_folder_path, img_name)
            img = cv2.imread(img_path)
            
            # Skip non-image files
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue

            embeddings = get_face_embedding(img)

            if embeddings:
                for embedding in embeddings:
                    faces.append(embedding)
                    person_names.append(person_folder)

# Apply PCA to reduce dimensionality of face embeddings
pca = PCA(n_components=30)  # Reduce to 30 components, adjust as needed
faces_pca = pca.fit_transform(faces)

confidence_threshold = 0.5
pruned_faces = []
pruned_person_names = []

for face, name in zip(faces_pca, person_names):
    distance = np.linalg.norm(face)  # Use L2 norm as a simple measure of confidence
    if distance < confidence_threshold:
        pruned_faces.append(face)
        pruned_person_names.append(name)

# Train a KNN classifier on the reduced face embeddings
ball_tree = BallTree(pruned_faces)

# Save the trained KNN model and PCA model for future use
with open('ball_tree.pkl', 'wb') as f:
    pickle.dump(ball_tree, f)

with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

# Save the face embeddings and names for future inference
with open('face_embeddings.pkl', 'wb') as file:
    pickle.dump((faces, person_names), file)

print("Training complete. Ball Tree model, PCA, and embeddings saved.")
