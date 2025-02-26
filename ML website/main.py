import threading
import time
import serial
from flask import Flask, render_template, Response
from camera import VideoCamera  # ensure VideoCamera is defined in camera.py

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

# Send coordinates to STM32 via serial port
def send_coordinates():
    try:
        # Update the serial port to the one that's connected to STM32F4
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # CHANGE THIS PORT TO YOUR SYSTEM
    except serial.SerialException as e:
        print("Could not open serial port:", e)
        return

    camera = VideoCamera()
    while True:
        # Get a frame and its associated detection coordinates
        frame, x, y, name = camera.get_frame()
        if frame is None:
            continue

        # Construct a message. For example, send a CSV string: x,y,name
        message = f"{x},{y},{name}\n"
        try:
            ser.write(message.encode())
            print("Sent:", message.strip())  # Debugging - Confirmed sending to STM32
        except Exception as e:
            print("Error sending data:", e)
        
        # Delay between transmissions to match your desired update rate
        time.sleep(0.1)

if __name__ == '__main__':
    # Start the coordinate sending thread
    coord_thread = threading.Thread(target=send_coordinates, daemon=True)
    coord_thread.start()
    
    # Start the Flask app for video streaming
    app.run(host='0.0.0.0', port=5001, debug=True)
