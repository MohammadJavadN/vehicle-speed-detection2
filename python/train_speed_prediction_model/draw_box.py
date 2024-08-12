import cv2
import json


def parse_json(json_path):
    # Initialize an empty dictionary to store vehicle data
    vehicles = {}

    # Reading the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Iterate through each item in the data
    for item in data:
        if 'name' in item:
            id = item.get('track_id', 0)
            if id == 0:
                continue

            speed = item.get('velocity', 0)
            if speed == 0 or speed is None:
                continue

            bbox = item.get('box', {})
            if bbox is None:
                continue

            iframe = item.get('frame', 0)
            cls = item.get('class', 0)

            # Unpack the bounding box coordinates
            x1 = bbox.get('x1', 0)
            y1 = bbox.get('y1', 0)
            x2 = bbox.get('x2', 0)
            y2 = bbox.get('y2', 0)

            # Store the extracted data in the vehicles dictionary
            vehicles[iframe] = (id, (x1, y1, x2, y2), speed, cls)

    return vehicles


js_path = '../../videos/FILE0010.json'
vid_path = js_path.replace('json', 'ASF')
out_path = js_path.replace('json', 'MP4')

vehicles = parse_json(js_path)
cap = cv2.VideoCapture(vid_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

label = ""
iframe = 0
while True:
    ret, frame = cap.read()
    iframe += 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not ret:
        break

    if iframe in vehicles:
        id, (x1, y1, x2, y2), speed, cls = vehicles[iframe]
        cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 2)

        # Prepare the text to display
        label = f'{cls}: {speed}'

        # Define the position for the label
        label_x = int((x1 + x2) * w / 2) - 10
        label_y = int((y1 + y2) * h / 2)

        # Draw the text on the frame
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    # else:
    #     cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    out.write(frame)
