import numpy as np
import math
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

    def init(self, x0, y0, H, xr, yr, Hr, l, f, m, n):

        self.l = l
        self.f = f
        self.m = m
        self.n = n

        alpha_1 = np.arctan(y0 / H)
        alpha = math.degrees(alpha_1 * (np.pi/180))
        betha_1 = np.arctan(x0 / np.sqrt(y0*y0 + H*H))
        betha = math.degrees(betha_1 * (np.pi/180))
        Rx = np.array(
            [
                [1,0,0],
                [0, np.cos(alpha), np.sin(alpha)],
                [0,-np.sin(alpha), np.cos(alpha)]
            ],
        )
        Ry = np.array(
            [
                [np.cos(betha), 0, np.sin(betha)],
                [0, 1, 0],
                [-np.sin(betha),0,np.cos(betha)]
            ],
        )
        A = np.dot(Ry, Rx)
        self.A_inv = np.linalg.inv(A) 

        if xr == 0:
            nx = 0
        else:
            nx = 1

        if yr == 0:
            ny = 0
        else:
            ny = 1

        if Hr == 0:
            nz = 0
        else:
            nz = 1

        R_r = np.array(
            [
                [xr],
                [yr],
                [Hr]
            ],
        )
        L = np.dot(A, R_r)
        self.x1 = L[0, 0]
        self.y1 = L[1, 0]
        self.z1 = L[2, 0]

        # T = (a, b, c)
        n_r = np.array(
            [
                [nx],
                [ny],
                [nz]
            ],
        )
        T = np.dot(A, n_r)
        self.a = T[0, 0]
        self.b = T[1, 0]
        self.c = T[2, 0]

        self.initiated = True

    def calculate_real_L(self, u1, v1, u2, v2):

        x_p = (1 / (100 * self.l)) * (u1 - (self.m/2))
        y_p = (1 / (100 * self.l)) * ((self.n/2) - v1)

        numerator = self.a*self.x1 + self.b*self.y1 + self.c*self.z1
        denominator = self.a*x_p + self.b*y_p + self.c*self.f
        if denominator == 0:
            return 0
        t = numerator / denominator

        w = np.array(
            [
                [x_p*t],
                [y_p*t],
                [f*t]
            ],
        )
        Pos=np.dot(self.A_inv, w)
        x_real1 = Pos[0, 0]
        y_real1 = Pos[1, 0]
        H_real1 = Pos[2, 0]

        x_p = (1 / (100 * self.l)) * (u2 - (self.m/2))
        y_p = (1 / (100 * self.l)) * ((self.n/2) - v2)
        numerator = self.a*self.x1 + self.b*self.y1 + self.c*self.z1
        denominator = self.a*x_p + self.b*y_p + self.c*self.f
        if denominator == 0:
            return 0
        t = numerator / denominator
        w = np.array(
            [
                [x_p*t],
                [y_p*t],
                [f*t]
            ],
        )
        Pos=np.dot(self.A_inv, w)
        x_real2 = Pos[0, 0]
        y_real2 = Pos[1, 0]
        H_real2 = Pos[2, 0]

        x_o = x_real2 - x_real1
        y_o = y_real2 - y_real1

        L = np.sqrt(x_o*x_o + y_o*y_o)

        return L

    def predict(self, u1, v1, u2, v2):
        if not self.initiated:
            return -1
        L = self.calculate_real_L(u1, v1, u2, v2)
        T = 1.0
        return L / T * 3600.0 / 1000.0  # TODO: 29.10.24 calc T


# Store the selected points
selected_points = []


def select_points(image):
    cv2.imshow("image", image)
    print("Close image to enter parameters.")
    cv2.waitKey(0)
    # Close the window
    cv2.destroyAllWindows()

    print("Please set the coordination of Q=(X0,Y0,H)")
    x0 = float(input("X0 = "))
    y0 = float(input("Y0 = "))
    H = float(input("H = "))

    print("Please set the coordination of P=(X,Y,H)")
    X_R = float(input("X = "))
    Y_R = float(input("Y = "))
    H_R = float(input("H = "))

    print("Please enter the focal length (m) of the camera")
    f = float(input("Focal length = "))

    height, width= image.shape[:2]
    
    Diagonal = np.sqrt(width*width + height*height) * 0.0264583333
    l = np.sqrt((width*width + height*height)) / (Diagonal)

    m = float(width)
    n = float(height)

    print("Please enter the pixel density of the camera")
    l = float(input("lambda = "))

    return x0, y0, H, X_R, Y_R, H_R, l, f, m, n


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
