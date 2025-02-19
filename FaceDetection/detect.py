import cv2
import numpy as np
import pickle
import dlib

def get_face_embedding(gray, image):
    faces = detector(gray)  # Detect faces
    
    if len(faces) == 0:
        return None

    embeddings = []
    for face in faces:
        landmarks = sp(gray, face)  # Detect landmarks
        face_descriptor = face_rec_model.compute_face_descriptor(image, landmarks)
        embeddings.append(np.array(face_descriptor))
    
    return embeddings

# Load the pre-trained face detector, shape predictor, and face recognition model from dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download from dlib's model repo
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')  # Download from dlib's model repo

# Load the KNN classifier model
with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

# Load the saved face embeddings and names
with open('face_embeddings.pkl', 'rb') as file:
    faces, person_names = pickle.load(file)

# Confidence threshold for "Unknown"
confidence_threshold = 0.6  # Adjust this threshold as needed

# Start the webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    # Detect faces in the frame
    faces_rects = detector(gray)

    for face in faces_rects:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_image = frame[y:y+h, x:x+w]

        # Get face embedding
        embeddings = get_face_embedding(gray, frame)

        if embeddings:
            # Compare the face embedding with the ones from the training set using KNN
            embedding = embeddings[0]  # Assuming one face per frame
            distances, indices = knn.kneighbors([embedding])  # Find the closest match

            # Retrieve the name of the closest match
            closest_person_name = person_names[indices[0][0]]
            confidence = 1 / (distances[0][0] + 1e-5)  # Inverse of distance to get confidence score

            if confidence < confidence_threshold:
                label = "Unknown"
            else:
                label = closest_person_name

            # Draw a bounding box around the face and show the predicted label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
