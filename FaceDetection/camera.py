import cv2
import numpy as np
import dlib
import pickle
import os
import time

class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        while not self.cap.isOpened():
            time.sleep(0.5)
        print("Camera Initialized")

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


        # Load the pre-trained face detector and face recognition model from dlib
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        self.sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        self.modTime = os.path.getmtime('ball_tree.pkl')
        # Load the BallTree model and PCA model
        with open('ball_tree.pkl', 'rb') as f:
            self.ball_tree = pickle.load(f)
        with open('pca_model.pkl', 'rb') as f:
            self.pca = pickle.load(f)

        # Load the saved face embeddings and names
        with open('face_embeddings.pkl', 'rb') as file:
            faces, self.person_names = pickle.load(file)
        
        self.confidence_threshold = 0.8

    def __del__(self):
        self.cap.release()
    
    def get_face_embedding(self, gray, image, face):
        landmarks = self.sp(gray, face)
        face_descriptor = self.face_rec_model.compute_face_descriptor(image, landmarks)
        return np.array(face_descriptor)

    def get_frame(self):
        # Default returns
        label = None
        biggest_box_x = self.width / 2
        biggest_box_y = self.height / 2
        if self.modTime != os.path.getmtime('ball_tree.pkl'):
            with open('ball_tree.pkl', 'rb') as f:
                self.ball_tree = pickle.load(f)
            with open('pca_model.pkl', 'rb') as f:
                self.pca = pickle.load(f)
            with open('face_embeddings.pkl', 'rb') as file:
                _, self.person_names = pickle.load(file)
                self.modTime = os.path.getmtime('ball_tree.pkl')
        
        # Checking camera for proper initialization
        if not self.cap:
            return None, biggest_box_x, biggest_box_y, label

        ret, frame = self.cap.read()

        # Return None if the frame is invalid
        if not ret or frame is None:
            return None, biggest_box_x, biggest_box_y, label


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use OpenCV Haar Cascade for face detection
        faces_rects = self.detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50))

        if len(faces_rects) > 0:
            # Convert the detected bounding boxes from OpenCV format (x, y, w, h) to dlib format (left, top, right, bottom)
            dlib_faces = [dlib.rectangle(x, y, x + w, y + h) for (x, y, w, h) in faces_rects]

            # Find the largest face by area (width * height)
            largest_face = max(dlib_faces, key=lambda face: face.width() * face.height())
            largest_face_rect = faces_rects[dlib_faces.index(largest_face)]
            
            # Get face embedding
            embedding = self.get_face_embedding(gray, frame, largest_face)
            
            # Apply PCA for dimensionality reduction
            embedding_pca = self.pca.transform([embedding])
            
            # Query the BallTree to find the closest match
            distances, indices = self.ball_tree.query(embedding_pca)
            
            closest_person_name = self.person_names[indices[0][0]]
            confidence = 1 / (distances[0][0] + 1e-5)

            if confidence < self.confidence_threshold:
                label = None
            else:
                label = closest_person_name

            # Draw the bounding box and label for the largest face
            x, y, w, h = largest_face_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            biggest_box_x = x + (w / 2)
            biggest_box_y = y + (h / 2)

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes(), biggest_box_x, biggest_box_y, label
    
    