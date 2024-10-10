import io
import cv2
import json
import numpy as np

from ultralytics import YOLO
from speed_detector import detect_speeds
from speed_estimator import twopoints_speed
from flask import Flask, request, Response

app = Flask(__name__)


def process_frame(im0):
    global cnt, model
    tracks = model.track(im0, persist=True, show=False, verbose=False)
    cnt += 1
    return detect_speeds(
        im0,
        tracks[0],
        cnt,
    )


def process_frame2(im0):
    global cnt, model
    tracks = model.track(im0, persist=True, show=False, verbose=False)
    cnt += 1
    list_of_objs = []

    for obj in tracks[0]:
        id = obj.boxes.id

        if not id:
            continue
        id = int(id[0])

        rect_i = obj.boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, rect_i)

        list_of_objs.append({'id': id, 'bbox': [x1, y1, x2, y2], 'speed': -1.0})

    return list_of_objs


def process_frame3(im0):
    global cnt, model
    tracks = model.track(im0, persist=True, show=False, verbose=False)
    cnt += 1
    list_of_objs = []

    for obj in tracks[0]:
        id = obj.boxes.id

        if not id:
            continue
        id = int(id[0])

        rect_i = obj.boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, rect_i)

        speed = twopoints_speed(id, x1, y1, abs(x2-x1), abs(y2-y1))

        list_of_objs.append({'id': id, 'bbox': [x1, y1, x2, y2], 'speed': speed})

    return list_of_objs


# Endpoint to receive and process video frames
@app.route('/process_video', methods=['POST'])
def process_video():    # don't work good (very slow)
    print('in server side')
    # Receive the video frame
    if 'frame' not in request.files:
        return "No frame found", 400

    # Read the image from the request
    file = request.files['frame']
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    npimg = np.fromstring(in_memory_file.getvalue(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Process the frame
    processed_frame = process_frame(frame)

    # Encode the processed frame to JPEG
    _, buffer = cv2.imencode('.jpg', processed_frame)

    # Create a response with the processed frame
    return Response(buffer.tobytes(), mimetype='image/jpeg')


# Route to receive video frames and send bounding box results back
@app.route('/process_frame', methods=['POST'])
def process_video_frame():
    try:
        # Read the frame data from the request
        frame_stream = request.files['frame'].read()

        # Convert the frame bytes into an OpenCV image
        frame_array = np.frombuffer(frame_stream, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        # Process the frame
        processed_data = process_frame2(frame)

        # Convert the result to JSON format
        result_json = json.dumps(processed_data)

        # Send back the result as a JSON response
        return Response(result_json, mimetype='application/json')
    except Exception as e:
        return str(e), 500


# Route to receive video frames and send bounding box results back
@app.route('/process_frame3', methods=['POST'])
def process_video_frame3():
    try:
        # Read the frame data from the request
        frame_stream = request.files['frame'].read()

        # Convert the frame bytes into an OpenCV image
        frame_array = np.frombuffer(frame_stream, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        # Process the frame
        processed_data = process_frame3(frame)

        # Convert the result to JSON format
        result_json = json.dumps(processed_data)

        # Send back the result as a JSON response
        return Response(result_json, mimetype='application/json')
    except Exception as e:
        return str(e), 500


@app.route('/init', methods=['POST'])
def init():
    global cnt, model
    try:
        model = YOLO("yolov8n.pt")  # Load the YOLO model
        cnt = -1  # Initialize counter
        print("Model loaded successfully")
        return "Model initialized", 200
    except Exception as e:
        print(f"Error loading model: {e}")
        return "Model initialization failed", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
