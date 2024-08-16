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
                vehicles[iframe + offset_frame] = (
                    id, (x1, y1, x2, y2), speed, cls
                )
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


def extract_augmented_data(
    video_paths,
    vehicles,
    modelpath='',
    max_frames=None,
    fq=3,
    ff=0,
    pts_num=50,
    best_num=10,
):

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
    tfn = 0
    speed = 0

    for video_path in video_paths:
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        if not max_frames:
            max_frames = 999999999

        fn = 0
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_name = video_path.split("/")[-1]

        prev_gray = None
        prev_pts = None

        frames = [None for _ in range(fq)]
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while (tfn < max_frames):
            ret, frame = cap.read()
            frames[tfn % fq] = frame
            if not ret:
                break
            tfn += 1

            if fn % 300 == 0:
                printProgressBar(
                    fn,
                    video_len,
                    prefix=f'Processing {vid_name}:',
                    suffix='Complete',
                    length=30,
                )
            fn += 1

            if fn < fq:
                continue

            if (tfn - fq + 1) in vehicles:
                prev_pts = []
                id, (x1, y1, x2, y2), speed, cls = vehicles[tfn - fq + 1]
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            else:
                continue

            for _ in range(pts_num):
                x = np.random.randint(x1, x2)
                y = np.random.randint(y1, y2)
                prev_pts.append([x, y])

            # Convert the frame to grayscale for optflow calculation
            prev_gray = cv2.cvtColor(
                frames[(tfn - fq) % fq], cv2.COLOR_BGR2GRAY)
            # prev_gray = frame_gray.copy()

            prev_pts = np.array(prev_pts).astype(np.float32)
            prev_pts = np.float32(prev_pts)
            prev_pts = np.expand_dims(prev_pts, axis=1)

            distances = [np.abs(x2-x1), np.abs(y2-y1)]
            for i in range(ff, fq - 1):
                frame_gray = cv2.cvtColor(
                    frames[(tfn - fq + i + 1) % fq], cv2.COLOR_BGR2GRAY)

                # Calculate optical flow using Lucas-Kanade method
                next_pts, status, errors = cv2.calcOpticalFlowPyrLK(
                    prev_gray, frame_gray,
                    prev_pts, None, **lk_params
                )

                # Check if errors is not None and has the right shape
                if errors is not None and errors.size > 0:
                    # get indices of the best points
                    errors_flat = errors.flatten()
                    sorted_indices = np.argsort(errors_flat)

                    # Get indices of top best_num best points
                    top_indices = sorted_indices[:best_num]

                    # Select top best_num points based on sorted indices
                    best_prev_pts = prev_pts[top_indices]
                    best_next_pts = next_pts[top_indices]
                    # best_errors = errors_flat[top_20_indices]

                    # Normalize points assuming the image size is known
                    prev_pts_n, minv, maxv = normalize_points(best_prev_pts)
                    next_pts_n, _, _ = normalize_points(
                        best_next_pts, minv, maxv
                    )

                    # Calculate Euclidean distances
                    distances.extend(sorted(
                            [
                                d2(prev_pts_n[i], next_pts_n[i])
                                for i in range(len(best_prev_pts))
                            ],
                        )
                    )

            X.append(distances)
            Y.append(speed)

        printProgressBar(
            fn,
            video_len,
            prefix=f'Processing {vid_name}:',
            suffix='Complete',
            length=30,
        )
        cap.release()

    return X, Y


def extract_augmented_data2(
    video_paths,
    vehicles,
    max_frames=None,
    fq=3,
    ff=0,
):

    y_coeffs = np.linspace(0.5, 0.8, 5)
    x_coeffs = np.linspace(0.15, 0.85, 7)

    # Create an instance of the Lucas-Kanade optical flow algorithm
    lk_params = dict(
        winSize=(100, 40),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    X = []
    Y = []
    tfn = 0
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
        while (tfn < max_frames):
            ret, frame = cap.read()
            frames[tfn % fq] = frame
            if not ret:
                break
            tfn += 1

            if tfn < fq:
                continue

            if (tfn - fq + 1) in vehicles:
                prev_pts = []
                id, (x1, y1, x2, y2), speed, cls = vehicles[tfn - fq + 1]
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            else:
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

            optflw = np.zeros((5, 7, 2 * (fq - ff - 1)))
            for i in range(ff, fq - 1):
                frame_gray = cv2.cvtColor(
                    frames[(tfn - fq + i + 1) % fq], cv2.COLOR_BGR2GRAY)

                # Calculate optical flow using Lucas-Kanade method
                next_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, frame_gray,
                    prev_pts, None, **lk_params
                )

                dfp = next_pts[0][0] - prev_pts[0][0]

                # Calculate Euclidean distances
                optflw[:, :, 2 * (i - ff):2 * (i - ff + 1)] = (
                        (next_pts - prev_pts).reshape(5, 7, 2) - dfp
                    ) / (w, h)
                
                # pixels = optflw[:, :, 2 * (i - ff)]
                # # Normalize to range [0, 1]
                # min_val = np.min(pixels)
                # max_val = np.max(pixels)
                # normalized_pixels = (pixels - min_val) / (max_val - min_val)
                # print(pixels)
                # cv2.imshow('frame', normalized_pixels)
                # k = cv2.waitKey(0) & 0xff
                # if k == 27:
                #     break

                # pixels = optflw[:, :, 2 * (i - ff)+1]
                # # Normalize to range [0, 1]
                # min_val = np.min(pixels)
                # max_val = np.max(pixels)
                # normalized_pixels = (pixels - min_val) / (max_val - min_val)
                # print(pixels)
                # cv2.imshow('frame', normalized_pixels)

            X.append(optflw)
            Y.append(speed)

        cap.release()

    return X, Y


def extract_augmented_data3(
    video_paths,
    vehicles,
    max_frames=None,
    fq=3,
    ff=0,
):

    y_coeffs = np.linspace(0.5, 0.8, 5)
    x_coeffs = np.linspace(0.15, 0.85, 7)

    # Create an instance of the Lucas-Kanade optical flow algorithm
    lk_params = dict(
        winSize=(100, 40),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    X = []
    Y = []
    tfn = 0
    speed = 0

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not max_frames:
            max_frames = 999999999

        fn = 0
        video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_name = video_path.split("/")[-1]
        print(vid_name)

        prev_gray = None
        prev_pts = None

        frames = [None for _ in range(fq)]
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while (tfn < max_frames):
            ret, frame = cap.read()
            frames[tfn % fq] = frame
            if not ret:
                break
            tfn += 1

            if fn % 300 == 0:
                printProgressBar(
                    fn,
                    video_len,
                    prefix=f'Processing {vid_name}:',
                    suffix='Complete',
                    length=30,
                )
            fn += 1

            if fn < fq:
                continue

            if (tfn - fq + 1) in vehicles:
                prev_pts = []
                id, (x1, y1, x2, y2), speed, cls = vehicles[tfn - fq + 1]
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            else:
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

            X.append(distances)
            Y.append(speed)

        printProgressBar(
            fn,
            video_len,
            prefix=f'Processing {vid_name}:',
            suffix='Complete',
            length=30,
        )
        cap.release()

    return X, Y


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
        Dense(256, input_shape=(len(X_train[0]),), activation='relu'),
        Dense(512, activation='relu'),
        # Dense(32, activation='relu'),
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
        epochs=20, batch_size=64, verbose=1
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


def train2d(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print(len(X_train), len(X_test))

    print(X_train[0].shape)
    layers = [
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train[0].shape),
        MaxPooling2D((2, 2)),
        # Conv2D(64, (3, 3), activation='relu', padding='same'),
        # MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
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
        np.array(X_train), np.array(y_train),
        epochs=20, batch_size=64, verbose=1
    )

    # Make predictions on the test set
    predictions = model.predict(np.array(X_test)).flatten()

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

    return model


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
        'FILE0001.ASF',
        'FILE0002.ASF',
        'FILE0004.ASF',
        'FILE0005.ASF',
        'FILE0008.ASF',
        'FILE0009.ASF',
        'FILE0010.ASF',
        'FILE0019.ASF',
        'FILE0026.ASF',
        'FILE0027.ASF',
        'FILE0030.ASF',
        'FILE0031.ASF',
    ]
    videos_path = '../../videos/'
    videos_path = '/media/javad/24D69A46D69A17DE/code/vehicle_speed_project_2/videos/'
    video_paths = [videos_path + v for v in videos]
    json_paths = [vp.replace('ASF', 'json') for vp in video_paths]

    model_path = '../../speed_prediction_model/'
    speed_prediction_model_path = model_path + 'speed_prediction_model.tflite'

    features_path = 'features/data4_6.csv'

    vehicles = parse_json(json_paths)
    print('json file was parsed successfully!\n')

    if regen:
        fq = 20
        ff = 15

        print(f'fq={fq}, ff={ff}')
        # X, y = extract_augmented_data2(
        #     video_paths=video_paths,
        #     vehicles=vehicles,
        #     # modelpath=license_plate_detector_model_path,
        #     fq=fq,
        #     ff=ff,  # number of frame involved: fq - ff - 1
        #     best_num=20,
        #     hist_bins=[0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
        # )

        X, y = extract_augmented_data3(
            video_paths=video_paths,
            vehicles=vehicles,
            # modelpath=license_plate_detector_model_path,
            fq=fq,
            ff=ff,  # number of frame involved: fq - ff - 1
        )

        print('\naugmented data extracted successfully!')

        save_features_in_file(X, y, path=features_path)
    else:
        X, y = load_features_from_file(path=features_path)
        print('\naugmented data loaded successfully!')

    print('\ntraining model started...')

    model, scaler = train(X, y)

    # dump(scaler, model_path + 'std_scaler.bin', compress=True)

    # print('\nModel trained!')

    # save_model(model, speed_prediction_model_path)


# Main function
def main2():
    regen = True
    videos = [
        # 'FILE0001.ASF',
        # 'FILE0002.ASF',
        # 'FILE0004.ASF',
        # 'FILE0005.ASF',
        # 'FILE0008.ASF',
        # 'FILE0009.ASF',
        'FILE0010.ASF',
        # 'FILE0019.ASF',
        # 'FILE0026.ASF',
        # 'FILE0027.ASF',
        'FILE0030.ASF',
        # 'FILE0031.ASF',
    ]
    videos_path = '../../videos/'
    videos_path = '/media/javad/24D69A46D69A17DE/code/vehicle_speed_project_2/videos/'
    video_paths = [videos_path + v for v in videos]
    json_paths = [vp.replace('ASF', 'json') for vp in video_paths]

    model_path = '../../speed_prediction_model/'
    speed_prediction_model_path = model_path + 'speed_prediction_model.tflite'

    features_path = 'features/data4_6.csv'

    vehicles = parse_json(json_paths)
    print('json file was parsed successfully!\n')

    if regen:
        fq = 20
        ff = 10

        print(f'fq={fq}, ff={ff}')
        X, y = extract_augmented_data2(
            video_paths=video_paths,
            vehicles=vehicles,
            fq=fq,
            ff=ff,  # number of frame involved: fq - ff - 1
        )

        print('\naugmented data extracted successfully!')

        np.save('features/X.npy', X)
        np.save('features/Y.npy', y)
    else:
        X = np.load('features/X.npy')
        y = np.load('features/Y.npy')
        print('\naugmented data loaded successfully!')

    print('\ntraining model started...')

    model = train2d(X, y)

    print('\nModel trained!')

    # save_model(model, speed_prediction_model_path)


if __name__ == "__main__":
    main()
