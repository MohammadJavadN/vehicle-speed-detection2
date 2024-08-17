import ast
import cv2
import csv
import json
import numpy as np
from joblib import dump
import tensorflow as tf
import xml.etree.ElementTree as ET
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Parse XML file
def parse_xml(xml_paths):
    vehicles = {}
    boxes = {}

    offset_frame0 = 0
    offset_frame = 0
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        offset_frame += offset_frame0
        w = int(root.find('meta').find('original_size').find('width').text)
        h = int(root.find('meta').find('original_size').find('height').text)

        for track in root.findall('track'):
            id = track.get('id')
            cls = track.get('label')
            vehicles[id] = []
            if cls != 'car':
                continue
            num = 0
            cnt = 0
            for box in track.findall('box'):
                offset_frame0 = int(box.get('frame'))
                if cnt >= len(track.findall('box')) - 30 * 4:
                    break
                cnt += 1
                if box.get('outside') == "0" and cnt % 4 == 0:
                    iframe = (int(box.get('frame')) + 1) // 4 + 1
                    x1 = float(box.get('xtl'))/w
                    y1 = float(box.get('ytl'))/h
                    x2 = float(box.get('xbr'))/w
                    y2 = float(box.get('ybr'))/h

                    speed = int(box.find('attribute').text)

                    vehicles[id].append(
                        ((x2-x1, y2-y1), speed)
                    )

                    boxes[iframe + offset_frame] = (
                        id, (x1, y1, x2, y2), speed, cls, num
                    )
                    num += 1
    return vehicles, boxes


# Normalize the points
def normalize_points(points, minv=None, maxv=None):
    min_val = points.min(axis=0)
    max_val = points.max(axis=0)

    if minv is None:
        s = (max_val - min_val)
    else:
        s = (maxv - minv)

    norm_points = (points - min_val) / s
    return norm_points, min_val, max_val


# Print iterations progress
def printProgressBar(
    iteration,
    total,
    prefix='',
    suffix='',
    decimals=1,
    length=100,
    fill='█',
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:."+str(decimals)+"f}").format(100*(iteration/float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def extract_augmented_data3(
    video_paths,
    vehicles,
    boxes,
    fq=3,
    ff=0,
    step=7,
):

    y_coeffs = np.linspace(0.5, 0.8, 5)
    x_coeffs = np.linspace(0.15, 0.85, 7)

    # Create an instance of the Lucas-Kanade optical flow algorithm
    lk_params = dict(
        winSize=(100, 40),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    tfn = 0
    speed = 0

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)

        fn = 0
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_name = video_path.split("/")[-1]
        print(vid_name)

        prev_gray = None
        prev_pts = None

        frames = [None for _ in range(fq)]
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ret, frame = cap.read()
            frames[tfn % fq] = frame
            if not ret:
                break
            tfn += 1

            if fn % 300 == 0:
                printProgressBar(
                    4 * fn,
                    video_len,
                    prefix=f'Processing {vid_name}:',
                    suffix='Complete',
                    length=30,
                )
            fn += 1

            if fn < fq:
                continue

            if (tfn - fq + 1) in boxes:
                prev_pts = []
                id, (x1, y1, x2, y2), speed, cls, num = boxes[tfn - fq + 1]
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            else:
                continue

            vehicle = vehicles[id]
            if num + fq > len(vehicle):
                continue

            for y_c in y_coeffs:
                for x_c in x_coeffs:
                    prev_pts.append([x1 + x_c * (x2 - x1), y1 + y_c * (y2 - y1)])
            # Convert the frame to grayscale for optflow calculation
            prev_gray = cv2.cvtColor(
                frames[(tfn - fq) % fq], cv2.COLOR_BGR2GRAY)
            # prev_gray = frame_gray.copy()

            prev_pts = np.array(prev_pts).astype(np.float32)
            prev_pts = np.float32(prev_pts)
            prev_pts = np.expand_dims(prev_pts, axis=1)

            distances = [np.abs(x2-x1)/w, np.abs(y2-y1)/h]

            (wi, hi), si = vehicle[num]
            for j in range(num + step, num + fq, step):
                (wj, hj), _ = vehicle[j]
                distances.extend([(wj - wi/wi), (hj - hi)/hi])

            for i in range(ff, fq - 1):
                frame_gray = cv2.cvtColor(
                    frames[(tfn - fq + i + 1) % fq], cv2.COLOR_BGR2GRAY)

                # Calculate optical flow using Lucas-Kanade method
                next_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, frame_gray,
                    prev_pts, None, **lk_params
                )

                dfp = next_pts[0][0] - prev_pts[0][0]

                distances.extend(
                    [
                        d2(next_pts[i], prev_pts[i] + dfp)/(w)
                        for i in range(len(prev_pts))
                    ]
                )
                # # Calculate Euclidean distances
                # optflw[:, :, 2 * (i - ff):2 * (i - ff + 1)] = (
                #         (next_pts - prev_pts).reshape(5, 7, 2) - dfp
                #     ) / (w, h)

            if int(id) % 8 == 0:
                X_test.append(distances)
                y_test.append(speed)
            else:
                X_train.append(distances)
                y_train.append(speed)

        printProgressBar(
            4 * fn,
            video_len,
            prefix=f'Processing {vid_name}:',
            suffix='Complete',
            length=30,
        )
        cap.release()

    return X_train, X_test, y_train, y_test


def d2(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def save_features_in_file(X, Y, path='data/data.csv'):
    # Define field names (column headers)
    field_names = ['X', 'speed']

    # Create a list of dictionaries (rows)
    rows = []
    for i in range(len(X)):
        rows.append(
            {
                'X': X[i],
                'speed': Y[i],
            }
        )

    # Write data to CSV file
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()  # Write column headers
        writer.writerows(rows)  # Write data rows

    print(f"CSV file {path} created successfully!")


def load_features_from_file(path='data/data.csv'):
    X, Y = [], []

    # Read data from CSV file
    with open(path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Use ast.literal_eval to safely parse string representations of lists
            X.append(ast.literal_eval(row['X']))
            Y.append(ast.literal_eval(row['speed']))

    return X, Y


def train(X_train, X_test, y_train, y_test):
    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.15, random_state=42
    # )

    print(len(X_train), len(X_test))
    # Normalize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the neural network model
    layers = [
        Dense(1024, input_shape=(len(X_train[0]),), activation='relu'),
        Dense(1024, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ]
    model = Sequential(layers)

    # for layer in layers:
    #     print(layer)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Print model summary
    model.summary()

    # Train the model
    model.fit(
        X_train_scaled, np.array(y_train),
        epochs=30, batch_size=32, verbose=1
    )

    # Make predictions on the test set
    predictions = model.predict(X_test_scaled).flatten()

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)

    precision = 0
    y_p = []
    for i, p in enumerate(predictions):
        # print(y_test[i], p)
        y_p.append(p)
        precision += (1 - (abs(y_test[i] - p)/p))

    print('precision=', precision/len(predictions))

    save_features_in_file(y_test, y_p, path='test.csv')

    return model, scaler


def save_model(model, model_path):
    model.export("pb_model", "tf_saved_model")
    converter = tf.lite.TFLiteConverter.from_saved_model("pb_model")
    tflite_model = converter.convert()
    with open(model_path, "wb") as f:
        f.write(tflite_model)

    print(f"The model {model_path} saved successfully!")


# Main function
def main():
    regen = True
    videos = [
        # 'FILE0001.ASF',
        # 'FILE0002.ASF',
        # 'FILE0004.ASF',
        # 'FILE0005.ASF',
        # 'FILE0008.ASF',
        # 'FILE0009.ASF',
        # 'FILE0010.ASF',
        # 'FILE0019.ASF',
        # 'FILE0026.ASF',
        # 'FILE0027.ASF',
        'FILE0030.ASF',
        # 'FILE0031.ASF',
    ]
    videos_path = '../../videos/'
    # videos_path = '/media/javad/24D69A46D69A17DE/code/vehicle_speed_project_2/videos/'
    video_paths = [videos_path + v for v in videos]
    json_paths = [vp.replace('ASF', 'xml') for vp in video_paths]

    model_path = '../../speed_prediction_model/'
    speed_prediction_model_path = model_path + 'speed_prediction_model.tflite'

    features_path = 'features/data.csv'
    train_features_path = 'features/train_data.csv'
    test_features_path = 'features/test_data.csv'

    vehicles, boxes = parse_xml(json_paths)
    print('json file was parsed successfully!\n')

    if regen:
        fq = 35
        ff = 30

        print(f'fq={fq}, ff={ff}')

        X_train, X_test, y_train, y_test = extract_augmented_data3(
            video_paths=video_paths,
            vehicles=vehicles,
            boxes=boxes,
            fq=fq,
            ff=ff,  # number of frame involved: fq - ff - 1
        )

        print('\naugmented data extracted successfully!')

        save_features_in_file(X_train, y_train, path=train_features_path)
        save_features_in_file(X_test, y_test, path=test_features_path)
    else:
        X_train, y_train = load_features_from_file(path=train_features_path)
        X_test, y_test = load_features_from_file(path=test_features_path)
        print('\naugmented data loaded successfully!')

    print('\ntraining model started...')

    model, scaler = train(X_train, X_test, y_train, y_test)

    # dump(scaler, model_path + 'std_scaler.bin', compress=True)

    # print('\nModel trained!')

    # save_model(model, speed_prediction_model_path)


if __name__ == "__main__":
    main()
