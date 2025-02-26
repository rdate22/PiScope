import threading
import time
import serial
from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame, x, y, name = camera.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def send_coordinates():
    try:
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        print("Serial connection established with STM32")
    except serial.SerialException as e:
        print("Could not open serial port:", e)
        return

    camera = VideoCamera()
    while True:
        frame, x, y, name = camera.get_frame()
        if frame is None:
            continue

        message = f"{x},{y},{name}\n"
        try:
            ser.write(message.encode())
            print("Sent:", message.strip())
        except Exception as e:
            print("Error sending data:", e)

        time.sleep(0.1)

if __name__ == '__main__':
    # Start Flask in a separate thread (WITHOUT debug=True)
    flask_thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 5001}, daemon=True)
    flask_thread.start()
    
    # Run serial communication in the main thread
    send_coordinates()
