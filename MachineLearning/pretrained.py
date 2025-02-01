import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Intersection over union
def compute_iou(box1, box2):
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

def track_objects(detected_boxes, previous_boxes, iou_threshold=0.5, tracked_counter=0):
    current_ids = []
    tracked_ids = []
    
    # Loop through all detected boxes in the current frame
    for i, curr_box in enumerate(detected_boxes):
        max_iou = 0
        matched_id = None

        # Loop through all previous boxes to find the best match based on IoU
        for j, prev_box in enumerate(previous_boxes):
            iou = compute_iou(curr_box, prev_box)

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


def main():
    # Load the pre-trained YOLOv5 model 
    model = YOLO("yolov5n.pt")  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cap = cv2.VideoCapture(0)

    # Storing data for frame skipping 
    previous_boxes = []  
    frame_counter = 0
    tracked_counter = 0  # Counter for unique tracking IDs

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Increment frame counter
        frame_counter += 1

        # Checking every 3 frames
        if frame_counter % 3 == 0:
            # Run detection with YOLOv5
            results = model(frame)
            
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
            current_ids, tracked_ids, tracked_counter = track_objects(person_boxes, previous_boxes, tracked_counter)
            
            # Update the previous bounding boxes
            previous_boxes = person_boxes
        
        else:
            # Tracking objects
            current_ids, tracked_ids, tracked_counter = track_objects(previous_boxes, previous_boxes, tracked_counter)

            # Update the previous bounding boxes
            person_boxes = previous_boxes

        # Draw bounding boxes
        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Display the tracking ID
            # cv2.putText(frame, f"ID: {tracked_ids[i]}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Display confidence score rounded to 4 decimal places
            cv2.putText(frame, f"Confidence: {person_scores[i]:.4f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        
        # Display Camera
        cv2.imshow("Tracking", frame)

        # Break on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
