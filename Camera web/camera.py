import torch
import cv2
import numpy as np
from ultralytics import YOLO

class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()

        if not ret or frame is None:
            return None

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
    
    