import threading
import time
from flask import Flask, request, jsonify, render_template, Response
from camera import VideoCamera  # ensure VideoCamera is defined in camera.py
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import serial

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'Dataset/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

# Function to check the allowed file extensions
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

def gen(camera):
    while True:
        frame, x, y, name = camera.get_frame()
        if frame is None:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')

@app.route('/add_user', methods=['POST'])
def add_user():
    # Get the userName from the form data
    user_name = request.form.get('userName')

    if not user_name:
        return jsonify({'error': 'User name is required'}), 400

    # Create a folder specific to the user
    user_folder = create_user_folder(user_name)

    # Check if the form contains files
    if 'photos' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    # Get the files from the form data
    files = request.files.getlist('photos')

    # List to store file paths
    file_paths = []

    # Handle and save each file
    for file in files:
        if file and allowed_file(file.filename):
            # Secure the filename and create the path
            filename = secure_filename(file.filename)
            file_path = os.path.join(user_folder, filename)
            file.save(file_path)
            file_paths.append(file_path)
        else:
            return jsonify({'error': f'Invalid file type for {file.filename}'}), 400

    # Return success message with file paths
    return jsonify({
        'message': 'User added successfully!',
        'user_name': user_name,
        'uploaded_files': file_paths
    })

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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
    flask_thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": 5001}, daemon=True)
    flask_thread.start()

    # Start Serial Communication Without Blocking Flask
    send_coordinates()