import numpy as np
from numpy.linalg import inv, norm, solve
import math
import matplotlib.pyplot as plt
import cv2


class TransformationCalculator:
    def __init__(self):
        self.CP = None
        self.CI = None
        self.h = None
        self.H_inverse = None
        self.H = None
        self.t = 1
        self.s = 1
        self.meanX = None
        self.meanY = None
        self.meanU = None
        self.meanV = None
        self.initiated = False

    def init(self, x, y, u, v):
        self.meanX = np.mean(x)
        self.meanY = np.mean(y)
        self.meanU = np.mean(u)
        self.meanV = np.mean(v)

        sxy = 0
        suv = 0
        for i in range(4):
            x[i] -= self.meanX
            y[i] -= self.meanY
            u[i] -= self.meanU
            v[i] -= self.meanV

            sxy += math.sqrt(x[i] ** 2 + y[i] ** 2)
            suv += math.sqrt(u[i] ** 2 + v[i] ** 2)

        self.CP = 4 * math.sqrt(2) / sxy
        self.CI = 4 * math.sqrt(2) / suv

        for i in range(4):
            x[i] *= self.CP
            y[i] *= self.CP
            u[i] *= self.CI
            v[i] *= self.CI

        # Define the 8x8 matrix K
        matrix_data = [
            [x[0], y[0], 1, 0, 0, 0, -x[0] * u[0], -y[0] * u[0]],
            [x[1], y[1], 1, 0, 0, 0, -x[1] * u[1], -y[1] * u[1]],
            [x[2], y[2], 1, 0, 0, 0, -x[2] * u[2], -y[2] * u[2]],
            [x[3], y[3], 1, 0, 0, 0, -x[3] * u[3], -y[3] * u[3]],
            [0, 0, 0, x[0], y[0], 1, -x[0] * v[0], -y[0] * v[0]],
            [0, 0, 0, x[1], y[1], 1, -x[1] * v[1], -y[1] * v[1]],
            [0, 0, 0, x[2], y[2], 1, -x[2] * v[2], -y[2] * v[2]],
            [0, 0, 0, x[3], y[3], 1, -x[3] * v[3], -y[3] * v[3]]
        ]
        K = np.array(matrix_data)

        # Define the 8x1 vector b
        vector_data = [u[0], u[1], u[2], u[3], v[0], v[1], v[2], v[3]]
        b = np.array(vector_data)

        # Solve for h
        self.h = solve(K, b)

        # Construct the 3x3 projective transformation matrix H
        matrix_data_H = [
            [self.h[0], self.h[1], self.h[2]],
            [self.h[3], self.h[4], self.h[5]],
            [self.h[6], self.h[7], 1.0]
        ]
        self.H = np.array(matrix_data_H)

        # Calculate the inverse of H
        self.H_inverse = inv(self.H)

        # Update t value
        self.t = self.s * (self.h[6] * x[0] + self.h[7] * y[0] + 1.0)
        self.initiated = True

    def calculate_real_L(self, u1, v1, u2, v2):
        u_c1 = self.CI * (u1 - self.meanU)
        v_c1 = self.CI * (v1 - self.meanV)
        u_c2 = self.CI * (u2 - self.meanU)
        v_c2 = self.CI * (v2 - self.meanV)

        vector_data1 = np.array([u_c1 * self.t, v_c1 * self.t, self.t])
        p1 = self.H_inverse @ vector_data1

        s1 = p1[2]
        x_c1 = p1[0] / s1
        y_c1 = p1[1] / s1

        vector_data2 = np.array([u_c2 * self.t, v_c2 * self.t, self.t])
        p2 = self.H_inverse @ vector_data2

        s2 = p2[2]
        x_c2 = p2[0] / s2
        y_c2 = p2[1] / s2

        x1 = (x_c1 / self.CP) + self.meanX
        y1 = (y_c1 / self.CP) + self.meanY
        x2 = (x_c2 / self.CP) + self.meanX
        y2 = (y_c2 / self.CP) + self.meanY

        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def predict(self, u1, v1, u2, v2):
        if not self.initiated:
            return -1
        L = self.calculate_real_L(u1, v1, u2, v2)
        T = 1.0
        return L / T * 3600.0 / 1000.0  # TODO: 29.10.24 calc T


# Store the selected points
selected_points = []


# Mouse callback function
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:

        cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
        strXY = '(' + str(x) + ',' + str(y) + ')'
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, strXY, (x+10, y-10), font, 1, (255, 255, 255))
        cv2.imshow("image", img)

        selected_points.append((x, y))
        print(f"Point selected: ({x}, {y})")
    elif event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((-1, -1))


def select_points(image):
    cv2.imshow("image", image)
    cv2.setMouseCallback("image", select_point)

    # Wait until 4 points are selected
    while len(selected_points) < 5:
        cv2.waitKey(1)  # Small delay to keep window responsive

    cv2.destroyAllWindows()
    selected_points.pop()
    u = []
    v = []
    x = []
    y = []
    # Display the selected points
    print("Selected points:", selected_points)
    minX = 10000
    minY = 10000
    for i in range(4):
        if selected_points[i][0] < minX:
            minX = selected_points[i][0]
        if selected_points[i][1] < minY:
            minY = selected_points[i][1]
    print(minX, minY)
    for i in range(4):
        u.append(selected_points[i][0])
        v.append(selected_points[i][1])

        print(f"p[{i}]={selected_points[i]}, ({selected_points[i][0]-minX}, {selected_points[i][1]-minY})")

        x.append(float(input(f"x[{i}] =")))
        y.append(float(input(f"y[{i}] =")))

    return x, y, u, v


t = TransformationCalculator()
cap = cv2.VideoCapture("out_top_no_plate.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("fps= ", fps)
cnt = 0
global last_frame, last_u, last_v
last_u = None
last_v = None
last_frame = None
while cap.isOpened():

    success, img = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    cnt += 1
    if cnt == 1:
        t.init(select_points(image=img))

    cv2.imshow('image', img)

    def printCoordinate(event, u, v, flags, params):
        global last_frame, last_u, last_v
        if event == cv2.EVENT_LBUTTONDOWN:

            cv2.circle(img, (u, v), 3, (255, 255, 255), -1)
            strXY = '(' + str(u) + ',' + str(v) + ')'
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img, strXY, (u+10, v-10), font, 1, (255, 255, 255))

            if last_frame is not None:
                speed = t.predict(last_u, last_v, u, v) / (cnt - last_frame) * fps
                print("df = ", cnt - last_frame)

                cv2.putText(img, f"speed = {speed}", (50, 50), font, 2, (0, 255, 0))

            cv2.imshow("image", img)
            last_frame = cnt
            last_u = u
            last_v = v
            # file = open('pixel_position.txt', 'a')     
            # file.write(str(x) + " " + str(y) + "\n")

    cv2.setMouseCallback("image", printCoordinate)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
