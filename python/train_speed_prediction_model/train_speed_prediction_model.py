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
            if id not in vehicles:
                vehicles[id] = []
            speed = item.get('velocity', 0)
            if speed == 0 or speed is None:
                continue
            iframe = item.get('frame', 0)
            bbox = item.get('box', {})
            cls = item.get('class', 0)

            # Unpack the bounding box coordinates
            x1 = bbox.get('x1', 0)
            y1 = bbox.get('y1', 0)
            x2 = bbox.get('x2', 0)
            y2 = bbox.get('y2', 0)

            # Calculate width and height from bounding box coordinates
            w = x2 - x1
            h = y2 - y1

            # Store the extracted data in the vehicles dictionary
            vehicles[id].append((iframe, (x1, y1, w, h), speed, cls))

    return vehicles


def preprocess(vehicles):

    deep = 3
    X = []
    Y = []

    for valus in vehicles.values():
        if len(valus) < deep:
            continue

        for i in range(len(valus) - (deep-1)):
            # if deep == 3:
            iframe0, (x0, y0, w0, h0), speed0, cls = valus[i]
            iframe1, (x1, y1, w1, h1), speed1, _ = valus[i + 1]
            iframe2, (x2, y2, w2, h2), speed1, _ = valus[i + 2]

            # X.append(
            #     (
            #         x0, y0, w0, h0,
            #         iframe1 - iframe0,
            #         x1, y1, w1, h1,
            #         iframe2 - iframe1,
            #         x2, y2, w2, h2,
            #     )
            # )

            X.append(
                (
                    iframe1 - iframe0,
                    (w1 - w0)/w0,
                    (h1 - h0)/h0,
                    iframe2 - iframe1,
                    (w2 - w1)/w1,
                    (h2 - h1)/h1,
                )
            )
            Y.append((speed0 + speed1)/2)

    return X, Y


def save_data_in_file(X, Y, path='data/data.csv'):
    # Define field names (column headers)
    field_names = ['x', 'y', 'pixel_speed', 'real_speed']

    # Create a list of dictionaries (rows)
    rows = []
    for i, (x, y, ps) in enumerate(X):
        rows.append(
            {
                'x': x,
                'y': y,
                'pixel_speed': ps,
                'real_speed': Y[i],
            }
        )

    # Write data to CSV file
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()  # Write column headers
        writer.writerows(rows)  # Write data rows

    print(f"CSV file {path} created successfully!")


def train(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    print(len(X_train), len(X_test))
    # Normalize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the neural network model
    model = Sequential([
        Dense(1024, input_shape=(6,), activation='relu'),
        Dense(1024, activation='relu'),
        Dense(1)  # Output layer
    ])

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
    json_path = 'FILE0004.json'

    speed_prediction_model_path = 'models/speed_prediction_model.tflite'

    vehicles = parse_json(json_path)
    print('json file was parsed successfully!\n')

    X, y = preprocess(
        vehicles=vehicles,
    )
    print('\naugmented data extracted successfully!')

    # save_data_in_file(X, y)

    print('\ntraining model started...')

    model, scaler = train(X, y)

    # dump(scaler, 'models/std_scaler.bin', compress=True)

    # print('\nModel trained!')

    # save_model(model, speed_prediction_model_path)


if __name__ == "__main__":
    main()
