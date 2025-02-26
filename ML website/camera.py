import torch
import cv2
import numpy as np
from ultralytics import YOLO

class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        # Load the pre-trained YOLOv5 model 
        self.model = YOLO("yolov5nu.pt")  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Storing data for frame skipping 
        self.previous_boxes = []  
        self.frame_counter = 0
        self.tracked_counter = 0  # Counter for unique tracking IDs
        self.previous_scores = []

    def __del__(self):
        self.cap.release()
    
    # Intersection over union
    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        inter_x1 = max(x1, x1_2)
        inter_y1 = max(y1, y1_2)
        inter_x2 = min(x2, x2_2)
        inter_y2 = min(y2, y2_2)

        # Return 0 if no overlap
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Calculate the area of both bounding boxes
        area_box1 = (x2 - x1) * (y2 - y1)
        area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate IoU
        union_area = area_box1 + area_box2 - intersection_area
        iou = intersection_area / union_area
        return iou

    def track_objects(self, detected_boxes, previous_boxes, iou_threshold=0.5, tracked_counter=0):
        current_ids = []
        tracked_ids = []
        
        # Loop through all detected boxes in the current frame
        for i, curr_box in enumerate(detected_boxes):
            max_iou = 0
            matched_id = None

            # Loop through all previous boxes to find the best match based on IoU
            for j, prev_box in enumerate(previous_boxes):
                iou = self.compute_iou(curr_box, prev_box)

                # Check for best match
                if iou > max_iou and iou >= iou_threshold: 
                    max_iou = iou
                    matched_id = j

            if matched_id is not None:
                tracked_ids.append(matched_id)
            else:
                tracked_ids.append(tracked_counter)  # New object (no match)
                tracked_counter += 1 

            current_ids.append(i)

        return current_ids, tracked_ids, tracked_counter

    def get_frame(self):
        # Checking camera for proper initialization
        if not self.cap:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None # Return None if the frame is invalid

        # Default values for person_boxes and person_scores
        person_boxes = []
        person_scores = []

        # Increment the frame counter and check if it's the 3rd frame
        self.frame_counter += 1

        if self.frame_counter % 3 == 0:
            # Run detection with YOLOv5
            results = self.model(frame)

            # Extract bounding boxes, labels, and scores
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Box coordinates
            labels = results[0].boxes.cls.cpu().numpy()  # Object labels (0 = person)
            scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores

            # Filter out only the detected people
            person_boxes = boxes[labels == 0]
            person_scores = scores[labels == 0]

            # Check if people are detected
            if person_scores.size > 0:
                # Filter out detections with confidence score below 0.5
                high_confidence_indices = person_scores >= 0.5  # Apply the threshold to person scores
                person_boxes = person_boxes[high_confidence_indices]  
                person_scores = person_scores[high_confidence_indices]
            
            # Tracking objects
            current_ids, tracked_ids, self.tracked_counter = self.track_objects(person_boxes, self.previous_boxes, self.tracked_counter)

            # Update the previous bounding boxes
            self.previous_boxes = person_boxes
            self.previous_scores = person_scores

        else:
            # Tracking objects
            current_ids, tracked_ids, self.tracked_counter = self.track_objects(self.previous_boxes, self.previous_boxes, self.tracked_counter)

            # Update the previous bounding boxes
            person_boxes = self.previous_boxes
            person_scores = self.previous_scores

        # Draw bounding boxes
        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Display the tracking ID
            # cv2.putText(frame, f"ID: {tracked_ids[i]}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Display confidence score rounded to 4 decimal places
            cv2.putText(frame, f"Confidence: {person_scores[i]:.4f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Find the biggest bounding box by area
        if len(person_boxes) > 0:
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in person_boxes]
            max_area_index = np.argmax(areas)
            biggest_box = person_boxes[max_area_index]
            biggest_box_x = biggest_box[0]  # x1 of the biggest box
            biggest_box_y = biggest_box[1]  # y1 of the biggest box
        else:
            biggest_box_x = None
            biggest_box_y = None
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), biggest_box_x, biggest_box_y, "Gayrick"
    
    