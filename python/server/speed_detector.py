import cv2
import numpy as np
import xml.etree.ElementTree as ET

from joblib import load
from tensorflow.lite.python.interpreter import Interpreter


def load_tflite_model(path):
    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


ff = 0
step = 6
main_step = 6
fq = step * 6 + 2
frames = [None for _ in range(fq)]
list_of_obj_list = [None for _ in range(fq)]
frame_num = 0

model_path = '/content/drive/MyDrive/speed_prediction_model/'
model_path = 'speed_prediction_model/'
speed_prediction_model_path = model_path + \
    'speed_prediction_model_nobox.tflite'

si, sid, sod = load_tflite_model(speed_prediction_model_path)

sc = load(model_path + 'std_scaler_nobox.bin')

y_coeffs = np.linspace(0.5, 0.8, 5)
x_coeffs = np.linspace(0.15, 0.85, 7)

# Create an instance of the Lucas-Kanade optical flow algorithm
lk_params = dict(
    winSize=(100, 40),
    maxLevel=5,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


def detect_speeds(
        frame: np.array,
        detected_objects: list,
        frame_num,
        vehicle_speed=None,
):

    global frames, list_of_obj_list, si, sid, sod, sc, width, height
    global y_coeffs, x_coeffs, lk_params

    width, height = frame.shape[1], frame.shape[0]

    frames[frame_num % fq] = frame.copy()

    list_of_obj_list[frame_num % fq] = detected_objects  # todo: clone
    frame_num += 1

    if frame_num < fq:
        return frame

    canvas = frames[(frame_num - fq) % fq]
    g = True
    for obj in list_of_obj_list[(frame_num - fq) % fq]:
        id = obj.boxes.id

        if not id:
            continue
        id = int(id[0])
        # Filter class 2 (you can adjust this based on your needs)
        # if int(obj.boxes.cls) == 2:
        rect_i = obj.boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, rect_i)
        wi = np.abs(x2 - x1)
        hi = np.abs(y2 - y1)

        prev_pts = []
        for y_c in y_coeffs:
            for x_c in x_coeffs:
                prev_pts.append([x1 + x_c * (x2 - x1), y1 + y_c * (y2 - y1)])

        prev_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        prev_pts = np.array(prev_pts).astype(np.float32)
        prev_pts = np.float32(prev_pts)
        prev_pts = np.expand_dims(prev_pts, axis=1)

        rect_js = []
        for i in range(ff + 1, fq - 1, step):
            rect_j = get_id_for_frame(
                (frame_num - fq + i) % fq,
                id,
            )

            if rect_j is None:
                break
            else:
                rect_js.append(rect_j)

        if len(rect_js) < ((fq - ff) // step):
            speed = update_objects_speed(frame_num-fq, id, -1)
            if speed > 0:
                if vehicle_speed:
                    g = update_eval(
                        vehicle_speed, frame_num - fq, rect_i, speed)
                draw(canvas, rect_i, speed, id, g)
            continue

        data = [wi/width, hi/height]
        # for rect_j in rect_js:
        #     x1, y1, x2, y2 = map(int, rect_j)
        #     wj = np.abs(x2 - x1)
        #     hj = np.abs(y2 - y1)
        #     data.append((wj - wi) / wi)
        #     data.append((hj - hi) / hi)

        for i in range(ff + 1, fq - 1, step):
            frame_gray = cv2.cvtColor(
                frames[(frame_num - fq + i) % fq], cv2.COLOR_BGR2GRAY)

            # Calculate optical flow using Lucas-Kanade method
            next_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, frame_gray,
                prev_pts, None, **lk_params
            )

            dfp = next_pts[0][0] - prev_pts[0][0]

            data.extend(
                [
                    d2(next_pts[i], prev_pts[i] + dfp)/(wi)
                    for i in range(len(prev_pts))
                ]
            )

        # Prepare input data (example).
        input_data = np.array(
            sc.transform([data]),
            dtype=np.float32
        )

        # Set input tensor.
        si.set_tensor(
            sid[0]['index'], input_data)

        # Run inference.
        si.invoke()
        predicted_speed = si.get_tensor(sod[0]['index'])[0][0]
        predicted_speed *= main_step / step
        speed = update_objects_speed(frame_num-fq, id, predicted_speed)

        if vehicle_speed:
            g = update_eval(vehicle_speed, frame_num - fq, rect_i, speed)

        draw(canvas, rect_i, speed, id, g)

    return canvas


def d2(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


e_cnt = 0
e_sum = 0


def update_eval(vehicle_speed, frame_num, rect_p, speed_p) -> bool:
    global e_cnt, e_sum, width, height
    TH = 0.15 * width
    if frame_num in vehicle_speed:
        x1p, y1p, x2p, y2p = map(int, rect_p)
        id, (x1, y1, x2, y2), speed = vehicle_speed[frame_num]
        pos_err = abs(x1-x1p) + abs(x2-x2p) + abs(y1-y1p) + abs(y2-y2p)

        if pos_err < TH and (x1 < width//2) and (x2 > width//2) \
                and (y1 < height//2) and (y2 > height//2) and abs(speed) > 20:
            e_cnt += 1
            e_sum += abs(speed - speed_p)/speed
            return True
    return False


def get_e():
    global e_cnt, e_sum
    return e_cnt, e_sum


# Parse XML file
def parse_xml(xml_path):
    vehicle_speed = {}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for track in root.findall('track'):
        id = int(track.get('id'))
        cnt = 0
        for box in track.findall('box'):
            if cnt >= len(track.findall('box')) - 30 * 4:
                break
            cnt += 1
            if box.get('outside') == "0" and cnt % 4 == 0:
                iframe = (int(box.get('frame')) + 1) // 4 + 1
                x1 = float(box.get('xtl'))
                y1 = float(box.get('ytl'))
                x2 = float(box.get('xbr'))
                y2 = float(box.get('ybr'))

                speed = int(box.find('attribute').text)

                vehicle_speed[iframe] = (
                    id, (x1, y1, x2, y2), speed,
                )
    return vehicle_speed


# Constants
NUM_COLORS = 10     # Example value
TEXT_SIZE = 1       # Font scale for cv2
STROKE_WIDTH = 2    # Example stroke width
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Defining the colors (BGR format for OpenCV)
COLORS = [
    ((0, 0, 0), (255, 255, 255)),  # Black text, White background
    ((255, 255, 255), (255, 0, 255)),  # White text, Magenta background
    ((0, 0, 0), (211, 211, 211)),  # Black text, Light gray background
    ((255, 255, 255), (0, 0, 255)),  # White text, Red background
    ((255, 255, 255), (255, 0, 0)),  # White text, Blue background
    ((255, 255, 255), (169, 169, 169)),  # White text, Dark gray background
    ((0, 0, 0), (255, 255, 0)),  # Black text, Cyan background
    ((0, 0, 0), (0, 255, 255)),  # Black text, Yellow background
    ((255, 255, 255), (0, 0, 0)),  # White text, Black background
    ((0, 0, 0), (0, 255, 0)),  # Black text, Green background
]

# Prepare paint-like configurations for OpenCV
num_colors = len(COLORS)
text_paints = [None] * num_colors
box_paints = [None] * num_colors
label_paints = [None] * num_colors

for i in range(num_colors):
    text_color, background_color = COLORS[i]

    # Text paint configuration
    text_paints[i] = {
        'color': tuple(map(int, text_color)),
        'font': FONT,
        'font_scale': TEXT_SIZE,
        'thickness': STROKE_WIDTH
    }

    # Box paint configuration
    box_paints[i] = {
        'color': tuple(map(int, background_color)),
        'thickness': int(STROKE_WIDTH)
    }

    # Label paint configuration (similar to box but filled)
    label_paints[i] = {
        'color': tuple(map(int, background_color)),
        'fill': True
    }


def draw(canvas, rect, speed, obj_id, greed_flag=False):
    # Ensure obj_id and color_id are integers
    obj_id = int(obj_id)
    color_id = obj_id % NUM_COLORS

    if greed_flag:
        color_id = 9  # Black text, Green background
    else:
        color_id = 3  # White text, Red background

    # Check the color and thickness types
    try:
        # Access box paint colors and thickness
        box_color = tuple(map(int, box_paints[color_id]['color']))
        box_thickness = int(box_paints[color_id]['thickness'])
    except KeyError as e:
        print(f"KeyError: {e}. Check if color_id is in " +
              "range and correctly formatted.")
        return

    # Prepare text details
    text = f"ID: {obj_id}, Speed: {speed:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(
        text, FONT, TEXT_SIZE, STROKE_WIDTH)
    line_height = text_height + STROKE_WIDTH
    y_label_offset = -line_height

    # Adjust rect coordinates to ensure they are integers
    x0, y0, x1, y1 = map(int, rect)
    x0, x1 = min(x0, x1), max(x0, x1)
    rect = [x0, y0, x1, y1]

    if rect[1] + int(y_label_offset) < 0:
        y_label_offset = 0

    # Draw the rectangle (bounding box)
    cv2.rectangle(canvas, (rect[0], rect[1]),
                  (rect[2], rect[3]), box_color, box_thickness)

    # Draw the label background
    label_rect_top_left = (rect[0], rect[1] + int(y_label_offset))
    label_rect_bottom_right = (rect[0] + int(text_width) + 2 * STROKE_WIDTH,
                               label_rect_top_left[1] + int(line_height))
    label_color = tuple(map(int, label_paints[color_id]['color']))
    cv2.rectangle(canvas, label_rect_top_left,
                  label_rect_bottom_right, label_color, cv2.FILLED)

    # Draw the text
    text_position = (rect[0], rect[1] + line_height + int(y_label_offset))
    text_color = tuple(map(int, text_paints[color_id]['color']))
    text_thickness = int(text_paints[color_id]['thickness'])
    cv2.putText(canvas, text, text_position, FONT,
                TEXT_SIZE, text_color, text_thickness)


def get_id_for_frame(frame_num: int, id: int):
    global list_of_obj_list

    objects = list_of_obj_list[frame_num]
    for obj in objects:
        if obj.boxes.id == id:
            return obj.boxes.xyxy[0].cpu().numpy()

    return None


objects_speed = {}
car_speeds = {}


def update_objects_speed(frame_num: int, id: int, speed: int):
    global objects_speed, car_speeds

    if speed != -1:
        if id not in objects_speed:
            frame_speed = {}
            frame_speed[frame_num] = speed
            objects_speed[id] = frame_speed
        else:
            objects_speed[id][frame_num] = speed

        if id not in car_speeds:
            car_speeds[id] = [speed]
        else:
            car_speeds[id].append(speed)

    if id not in objects_speed:
        return -1

    if objects_speed[id] is None:
        return -1

    return mean_speed(car_speeds[id])


def mean_speed(car_speeds: list):
    N = 20
    speeds = [None for _ in range(N)]
    if len(car_speeds) == 0:
        return 0

    total_sum = 0
    count = 0

    for speed in car_speeds:
        if speed > 0:
            total_sum += speed
            speeds[count % N] = speed
            count += 1

    if count == 0:
        return 0
    if count < N:
        return total_sum / count

    return np.mean(speeds)
