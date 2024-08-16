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
            cnt = 1
            for box in track.findall('box'):
                offset_frame0 = int(box.get('frame'))
                cnt += 1
                if box.get('outside') == "0" and cnt % 4 == 2:
                    x1 = float(box.get('xtl'))/w
                    y1 = float(box.get('ytl'))/h
                    x2 = float(box.get('xbr'))/w
                    y2 = float(box.get('ybr'))/h

                    speed = int(box.find('attribute').text)

                    vehicles[id].append(
                        ((x2-x1, y2-y1), speed)
                    )
    return vehicles


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
    vehicles: dict,
    step=4,
    fq=3,
):

    X = []
    Y = []

    for vehicle in vehicles.values():
        for i in range(len(vehicle) - fq * step):
            (wi, hi), si = vehicle[i]
            data = []
            for j in range(i + step, i + fq * step, step):
                (wj, hj), _ = vehicle[j]

                data.extend([(wj - wi/wi), (hj - hi)/hi])

            X.append(data)
            Y.append(si)

    return X, Y


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
        Dense(512, input_shape=(len(X_train[0]),), activation='relu'),
        Dense(1024, activation='relu'),
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

    features_path = 'features/data_box.csv'

    vehicles = parse_xml(json_paths)
    print('json file was parsed successfully!\n')

    if regen:
        fq = 5
        step = 8
        print(f'fq={fq}, step={step}')

        X, y = extract_augmented_data(
            vehicles=vehicles,
            fq=fq,
            step=step,
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


if __name__ == "__main__":
    main()
