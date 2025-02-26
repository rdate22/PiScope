import threading
import time
import serial
from flask import Flask, render_template, Response
from camera import VideoCamera  # Correct Import

app = Flask(__name__)

# Create a Single Camera Instance
camera = VideoCamera()

@app.route('/')
def index():
    """Render the main webpage."""
    return render_template('index.html')

def gen(camera):
    """Continuously stream video frames to the browser."""
    while True:
        frame, x, y, name = camera.get_frame()
        if frame is None:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """Provide the video feed endpoint."""
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

def send_coordinates():
    """Continuously send coordinates to STM32 via UART without blocking Flask."""
    ser = None

    while True:
        # Attempt to establish serial connection if not already connected
        if ser is None:
            try:
                ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # CHANGE PORT IF NEEDED
                print("erial connection established with STM32")
            except serial.SerialException as e:
                print("Could not open serial port:", e)
                ser = None
                time.sleep(2)  # Retry every 2 seconds
                continue

        # Get frame and person detection data
        frame, x, y, name = camera.get_frame()
        if frame is None:
            continue

        # Construct a message: "x,y,name"
        message = f"{x},{y},{name}\n"
        try:
            ser.write(message.encode())
            print(f"📤 Sent to STM32: {message.strip()}")  # Debugging Output
        except Exception as e:
            print("Error sending data:", e)
            ser.close()
            ser = None  # Reset connection on error

        time.sleep(0.1)  # Adjust transmission speed if necessary

if __name__ == '__main__':
    # Start Flask Server in a Separate Thread
    flask_thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 8000}, daemon=True)
    flask_thread.start()

    # Start Serial Communication Without Blocking Flask
    send_coordinates()
