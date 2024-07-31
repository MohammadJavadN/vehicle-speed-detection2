import ast
import cv2
import csv
import json
import numpy as np
from joblib import dump
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def parse_json(json_paths):
    # Initialize an empty dictionary to store vehicle data
    vehicles = {}

    offset_frame0 = 0
    offset_frame = 0
    for json_path in json_paths:
        # Reading the JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)
        offset_frame += offset_frame0

        # Iterate through each item in the data
        for item in data:
            offset_frame0 = item.get('frame', 0)
            if 'name' in item:
                cls = item.get('class', 0)
                # if cls not in [2, 5, 7]:
                #     continue
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

                # Unpack the bounding box coordinates
                x1 = bbox.get('x1', 0)
                y1 = bbox.get('y1', 0)
                x2 = bbox.get('x2', 0)
                y2 = bbox.get('y2', 0)

                # Store the extracted data in the vehicles dictionary
                vehicles[iframe + offset_frame] = (id, (x1, y1, x2, y2), speed, cls)
                # print(str(iframe) + ' + ' + str(offset_frame))

    return vehicles


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


def extract_augmented_data(video_paths, vehicles, modelpath='', real_data_coef=50,
                           verbose=0, max_frames=None, fq=3, ff=0,
                           pts_num=50):
    
    # interpreter = Interpreter(model_path=modelpath)
    # interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # height = input_details[0]['shape'][1]
    # width = input_details[0]['shape'][2]

    # float_input = (input_details[0]['dtype'] == np.float32)

    # input_mean = 127.5
    # input_std = 127.5

    # Create an instance of the Lucas-Kanade optical flow algorithm
    lk_params = dict(
        winSize=(100, 40),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    X = []
    Y = []
    frame_num = 0
    speed = 0

    for video_path in video_paths:
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        if not max_frames:
            max_frames = 999999999

        prev_gray = None
        prev_pts = None

        frames = [None for _ in range(fq)]
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while (frame_num < max_frames):
            ret, frame = cap.read()
            frames[frame_num % fq] = frame
            if not ret:
                break
            frame_num += 1
            # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # imH, imW, _ = frame.shape
            # image_resized = cv2.resize(image_rgb, (width, height))
            # input_data = np.expand_dims(image_resized, axis=0)

            # # Normalize pixel values if using a floating model
            # # (i.e. if model is non-quantized)
            # if float_input:
            #     input_data = (np.float32(input_data) - input_mean) / input_std

            # # Perform the detection by running the model with the frame as input
            # interpreter.set_tensor(input_details[0]['index'], input_data)
            # interpreter.invoke()

            # ...

            # output = interpreter.get_tensor(output_details[0]['index'])
            # output = output[0]
            # output = output.T

            # Get coordinates of bounding box, first 4 columns of output tensor
            # boxes_xywh, scores = output[..., :4], output[..., 4]

            # # Threshold Setting
            # threshold = 0.09

            if frame_num < fq:
                continue

            if (frame_num - fq + 1) in vehicles:
                prev_pts = []
                id, (x1, y1, x2, y2), speed, cls = vehicles[frame_num - fq + 1]
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            else:
                continue

            for _ in range(pts_num):
                x = np.random.randint(x1, x2)
                y = np.random.randint(y1, y2)
                prev_pts.append([x, y])

            # Convert the frame to grayscale for optflow calculation
            prev_gray = cv2.cvtColor(
                frames[(frame_num - fq) % fq], cv2.COLOR_BGR2GRAY)
            # prev_gray = frame_gray.copy()

            prev_pts = np.array(prev_pts).astype(np.float32)
            prev_pts = np.float32(prev_pts)
            prev_pts = np.expand_dims(prev_pts, axis=1)

            distances = [np.abs(x2-x1), np.abs(y2-y1)]
            for i in range(ff, fq - 1):
                frame_gray = cv2.cvtColor(
                    frames[(frame_num - fq + i + 1) % fq], cv2.COLOR_BGR2GRAY)

                # Calculate optical flow using Lucas-Kanade method
                next_pts, status, errors = cv2.calcOpticalFlowPyrLK(
                    prev_gray, frame_gray,
                    prev_pts, None, **lk_params
                )

                # Check if errors is not None and has the right shape
                if errors is not None and errors.size > 0:
                    # Flatten errors and sort them to get indices of the best points
                    errors_flat = errors.flatten()
                    sorted_indices = np.argsort(errors_flat)  # Indices of sorted errors

                    # Get indices of top 20 best points
                    top_20_indices = sorted_indices[:20]

                    # Select top 20 points based on sorted indices
                    best_prev_pts = prev_pts[top_20_indices]
                    best_next_pts = next_pts[top_20_indices]
                    # best_errors = errors_flat[top_20_indices]

                    # Normalize points assuming the image size is known
                    prev_pts_n, minv, maxv = normalize_points(best_prev_pts)
                    next_pts_n, _, _ = normalize_points(best_next_pts, minv, maxv)

                    # Calculate Euclidean distances
                    distances.extend([
                        np.sqrt(np.sum((prev_pts_n[i] - next_pts_n[i]) ** 2))
                        for i in range(len(best_prev_pts))
                    ])

            X.append(distances)
            Y.append(speed)

        cap.release()

    return X, Y


def save_data_in_file(X, Y, path='data/data.csv'):
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


def load_data_from_file(path='data/data.csv'):
    X, Y = [], []

    # Read data from CSV file
    with open(path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Use ast.literal_eval to safely parse string representations of lists
            X.append(ast.literal_eval(row['X']))
            Y.append(ast.literal_eval(row['speed']))

    return X, Y


def train(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print(len(X_train), len(X_test))
    # Normalize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the neural network model
    layers = [
        Dense(512, input_shape=(len(X_train[0]),), activation='relu'),
        Dense(512, activation='relu'),
        # Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ]
    model = Sequential(layers)

    # for layer in layers:
    #     print(layer)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(
        X_train_scaled, np.array(y_train),
        epochs=100, batch_size=32, verbose=0
    )

    # Make predictions on the test set
    predictions = model.predict(X_test_scaled).flatten()

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)

    precision = 0
    for i, p in enumerate(predictions):
        # print(y_test[i], p)
        precision += (1 - (abs(y_test[i] - p)/p))

    print('precision=', precision/len(predictions))

    return model, scaler


def save_model(model, model_path):
    model.export("model/pb_model", "tf_saved_model")
    converter = tf.lite.TFLiteConverter.from_saved_model("model/pb_model")
    tflite_model = converter.convert()
    with open(model_path, "wb") as f:
        f.write(tflite_model)

    print(f"The model {model_path} saved successfully!")


# Main function
def main():
    regen = False
    video_paths = [
        'FILE0002.ASF',
        # 'FILE0005.ASF',
        'FILE0010.ASF',
        'FILE0019.ASF',
        'FILE0031.ASF',
    ]
    json_paths = [vp.replace('ASF', 'json') for vp in video_paths]

    speed_prediction_model_path = 'models/speed_prediction_model.tflite'

    vehicles = parse_json(json_paths)
    print('json file was parsed successfully!\n')

    if regen:
        fq = 6
        ff = 4

        print(f'fq={fq}, ff={ff}')
        X, y = extract_augmented_data(
            video_paths=video_paths,
            vehicles=vehicles,
            # modelpath=license_plate_detector_model_path,
            # max_frames=5000,
            fq=fq,
            ff=ff, # number of frame involved: fq - ff - 1
            verbose=1,
        )

        print('\naugmented data extracted successfully!')

        save_data_in_file(X, y)
    else:
        X, y = load_data_from_file()
        print('\naugmented data loaded successfully!')

    print('\ntraining model started...')

    model, scaler = train(X, y)

    # dump(scaler, 'models/std_scaler.bin', compress=True)

    # print('\nModel trained!')

    # save_model(model, speed_prediction_model_path)


if __name__ == "__main__":
    main()
