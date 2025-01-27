import cv2

cap = cv2.VideoCapture(0)  # Use 0 for webcam or adjust if using Raspberry Pi camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the frame
    cv2.imshow('Camera Feed', frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
