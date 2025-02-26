import threading
import time
from flask import Flask, request, jsonify, render_template, Response
from camera import VideoCamera  # Ensure VideoCamera is defined in camera.py
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import serial

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'Dataset/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

# Initialize a single global camera instance
camera = VideoCamera()

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create user-specific folder if it doesn't exist
def create_user_folder(user_name):
    user_folder = os.path.join(UPLOAD_FOLDER, user_name)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

@app.route('/')
def index():
    """ Render the main webpage. """
    return render_template('index.html')

def gen():
    """ Stream video frames for Flask. """
    while True:
        frame, x, y, name = camera.get_frame()
        if frame is None:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """ Stream video from the single global camera instance. """
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_user', methods=['POST'])
def add_user():
    """ Upload images for a user and create their directory. """
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

def send_coordinates():
    """ Continuously send coordinates to STM32 via UART. """
    ser = None

    while True:
        if ser is None:
            try:
                ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)  # Change port if needed
                print("Serial connection established with STM32")
            except serial.SerialException as e:
                print("Could not open serial port:", e)
                ser = None
                time.sleep(2)
                continue

        frame, x, y, name = camera.get_frame()
        if frame is None:
            continue

        message = f"{x},{y},{name}\n"
        try:
            ser.write(message.encode())
            print(f"Sent to STM32: {message.strip()}")
        except Exception as e:
            print("Error sending data:", e)
            ser.close()
            ser = None  # Reset connection on error

        time.sleep(0.1)

if __name__ == '__main__':
    # Run Flask in a separate thread
    flask_thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 5001}, daemon=True)
    flask_thread.start()

    # Run serial communication separately
    serial_thread = threading.Thread(target=send_coordinates, daemon=True)
    serial_thread.start()

    # Keep the main thread alive
    while True:
        time.sleep(1)
