import threading
import time
from flask import Flask, request, jsonify, render_template, Response
from camera import VideoCamera  # Ensure VideoCamera is defined in camera.py
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import serial
import serial.tools.list_ports

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'Dataset/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
SEND_SIGNAL = threading.Event()

# Initialize a single global camera instance
camera = VideoCamera()

# Global variable to store the latest frame & coordinates
latest_frame_data = {"frame": None, "x": None, "y": None, "name": None}

# Infinite loop to capture frames & update the global variable
def update_frame():
    global latest_frame_data
    while True:
        frame, x, y, name = camera.get_frame()
        SEND_SIGNAL.clear() # reset thread event
        if frame is not None:
            latest_frame_data["frame"] = frame
            latest_frame_data["x"] = x
            latest_frame_data["y"] = y
            latest_frame_data["name"] = name
            if name is not None: # allow coordinate sending if name/face is detected
                SEND_SIGNAL.set()
            # print(f"New frame data: x={x}, y={y}, name={name}, {SEND_SIGNAL}") # Debugging statement
            time.sleep(1) # Avoid CPU overload if needed

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create user-specific folder if it doesn't exist
def create_user_folder(user_name):
    user_folder = os.path.join(UPLOAD_FOLDER, user_name)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

@app.route('/')
def index():
    return render_template('index.html')

# Stream video frames for Flask
def gen():
    while True:
        if latest_frame_data["frame"] is None:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_data["frame"] +
               b'\r\n\r\n')

# Serial connection helper function for def send_coordinates
def get_serial_connection():
    try:
        ser = serial.Serial('/dev/serial0', 115200, timeout=1)  # Change port if needed (/dev/ttyACM0)
        print("Serial connection established with STM32")
        return ser
    except serial.SerialException as e:
        print(f"Could not open UART port: {e}")
        return None

# Continuously send coordinates to STM32 via serial port, implements threading pause
# if face is not detected to avoid unecessary cpu usage
def send_coordinates():
    ser = None
    while True:
        # SEND_SIGNAL.wait() # pauses thread until signal is set (name/face detected)
        if ser is None:
            ser = get_serial_connection()
            if ser is None:
                time.sleep(1) # Avoid cpu usage if needed
                continue

        if latest_frame_data["frame"] is None:
            time.sleep(0.1)  # Avoid busy waiting if needed
            continue
        
        message = f"{latest_frame_data['x']},{latest_frame_data['y']},{latest_frame_data['name']}\n"
        try:
            ser.write(message.encode())
            print(f"Sent to STM32: {message.strip()}") # Debugging statement
        except Exception as e:
            print("Error sending data:", e)
            ser.close()
            ser = None  # Reset connection on error
        SEND_SIGNAL.clear()

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_user', methods=['POST'])
def add_user():
    user_name = request.form.get('userName')
    if not user_name:
        return jsonify({'error': 'User name is required'}), 400

    user_folder = create_user_folder(user_name)

    if 'photos' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('photos')
    file_paths = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(user_folder, filename)
            file.save(file_path)
            file_paths.append(file_path)
        else:
            return jsonify({'error': f'Invalid file type for {file.filename}'}), 400

    return jsonify({
        'message': 'User added successfully!',
        'user_name': user_name,
        'uploaded_files': file_paths
    })

if __name__ == '__main__':
    # Start background thread for sending coordinates to STM32f4
    serial_thread = threading.Thread(target=send_coordinates, daemon=True)
    serial_thread.start()

    # Start background thread for updateing global frame variable
    update_thread = threading.Thread(target=update_frame, daemon=True)
    update_thread.start()

    # Run Flask in the main thread
    app.run(host="0.0.0.0", port=8000)
