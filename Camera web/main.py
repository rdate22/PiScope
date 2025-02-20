from flask import Flask, render_template, Response
from flask_cors import CORS
from camera import VideoCamera

app = Flask(__name__)
cors = CORS(app, origins='*')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
            
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame
            + b'\r\n\r\n')

#Updated this main.py to display regular webcam feed on a seperate url endpoint
@app.route('/camera_web', methods = ['GET'])
def video_feed():
    return Response(gen(VideoCamera()),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port='5001', debug=True)
