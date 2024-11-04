package com.example.myapplication;

import static com.example.myapplication.OptSpeedDetector.getGridSpeed;
import static com.example.myapplication.OptSpeedDetector.getTwoPointsSpeed;
import static com.example.myapplication.OptSpeedDetector.gridSpeeds;
import static com.example.myapplication.OptSpeedDetector.gridSpeedsInit;
import static java.lang.Math.min;
import static java.lang.Math.sqrt;

import android.graphics.Bitmap;
import android.graphics.Canvas;

import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions;

import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;

public class CPSpeedDetector extends SpeedDetector {
    final static int fq = 2;
    double alpha;   // Example: 30 degrees converted to radians
    double beta;   // Example: 45 degrees converted to radians
    double lambda = 1.0 / 0.0264583333;                   // Example value
    double f = 50.0;                       // Example focal length or another parameter
    double x1, y1, z1;
    double a, b, c;
    RealMatrix A_inv;
    private int imW;
    private int imH;
    private int frameNum = 0;
    private Mat prevGray;
    private TermCriteria criteria;
    private Size winSize;
    private List<Mat> grayFrames;
    public CPSpeedDetector() {
        super(fq);
    }

    public void init(double f, double X_0, double Y_0, double H, double lambda, double XR, double YR, double HR) {
        alpha= Math.atan(Y_0 / H);
        beta = Math.atan(X_0 /sqrt(Y_0 * Y_0 + H * H));
//        alpha = Math.toDegrees(alpha);
//        beta = Math.toDegrees(beta);
        this.lambda = lambda;
        this.f = f;
        // Step 1: Define Rx matrix
        double[][] RxData = {
                {1, 0, 0},
                {0, Math.cos(this.alpha), Math.sin(this.alpha)},
                {0, -Math.sin(this.alpha), Math.cos(this.alpha)}
        };
        RealMatrix Rx = MatrixUtils.createRealMatrix(RxData);

        // Step 2: Define Ry matrix
        double[][] RyData = {
                {Math.cos(beta), 0, Math.sin(beta)},
                {0, 1, 0},
                {-Math.sin(beta), 0, Math.cos(beta)}
        };
        RealMatrix Ry = MatrixUtils.createRealMatrix(RyData);

        // Step 3: Compute A = Ry * Rx
        RealMatrix A = Ry.multiply(Rx);

        // Step 4: Compute A_inv (Inverse of A)
        try {
            DecompositionSolver solver = new LUDecomposition(A).getSolver();
            if (!solver.isNonSingular()) {
                System.err.println("Matrix A is singular and cannot be inverted.");
                return;
            }
            A_inv = solver.getInverse();
        } catch (Exception e) {
            System.err.println("Error computing inverse of matrix A: " + e.getMessage());
            return;
        }

        // Step 5: Determine nx, ny, nz based on X_R, Y_R, H_R
        int nx = (XR == 0) ? 0 : 1;
        int ny = (YR == 0) ? 0 : 1;
        int nz = (HR == 0) ? 0 : 1;

        nx = 0; ny = 0; nz = 1;

        // Step 6: Define R_r matrix
        double[][] R_rData = {
                {XR},
                {YR},
                {HR}
        };
        RealMatrix R_r = MatrixUtils.createRealMatrix(R_rData);

        // Step 7: Compute L = A * R_r
        RealMatrix L = A.multiply(R_r);
        x1 = L.getEntry(0, 0);
        y1 = L.getEntry(1, 0);
        z1 = L.getEntry(2, 0);

        // Step 8: Define n_r matrix
        double[][] n_rData = {
                {nx},
                {ny},
                {nz}
        };
        RealMatrix n_r = MatrixUtils.createRealMatrix(n_rData);

        // Step 9: Compute T = A * n_r
        RealMatrix T = A.multiply(n_r);
        a = T.getEntry(0, 0);
        b = T.getEntry(1, 0);
        c = T.getEntry(2, 0);

        System.out.println("@@@ Ry = " + Ry);
        System.out.println("@@@ Rx = " + Rx);
        System.out.println("@@@ A = " + A);
        System.out.println("@@@ R_r = " + R_r);
        System.out.println("@@@ alpha = " + alpha + ", beta=" + beta);
        System.out.println("@@@ a = " + a + ", b=" + b + ", c=" + c);
        System.out.println("@@@ x1 = " + x1 + ", y1=" + y1 + ", z1=" + z1);
        grayFrames = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            grayFrames.add(new Mat());
        }
        winSize = new Size(10, 10);
        criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
                200,
                0.001);
        prevGray = null;
        frameNum = 0;

    }

    public void detectSpeeds(Mat frame, List<DetectedObject> detectedObjects, Canvas canvas) {
        frames.set(frameNum % fq, frame.clone());

        Mat grayFrame = new Mat();
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        grayFrames.set(frameNum % fq, grayFrame);

        listOfObjList.set(frameNum % fq, new ArrayList<>(detectedObjects));
        frameNum++;

        if (frameNum < fq) {
            imW = frame.cols();
            imH = frame.rows();
            bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(frame, bitmap);

            if (canvas != null) {
                // Render the frame onto the canvas
                int w = canvas.getWidth();
                int h = canvas.getHeight();
                Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, w, h, false);
                canvas.drawBitmap(scaledBitmap, 0, 0, null);
            }
            return;
        }
        prevGray = grayFrames.get((frameNum - fq) % fq);
        gridSpeedsInit();
        predictSpeed(grayFrames.get((frameNum - fq + 1) % fq));

        bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888); // todo: comment this line
        Utils.matToBitmap(frames.get((frameNum - fq) % fq).clone(), bitmap);
//        Utils.matToBitmap(frame, bitmap);
        double sx, sy;
        double hf = frame.height();
        double wf = frame.width();
        if (canvas != null) {
            // Render the frame onto the canvas
            int w = canvas.getWidth();
            int h = canvas.getHeight();

            sx = w / wf;
            sy = h / hf;

            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, w, h, false);
            canvas.drawBitmap(scaledBitmap, 0, 0, null);
        } else return;

        for (DetectedObject object : listOfObjList.get((frameNum - fq) % fq)) {
            int id = getId(object);
            android.graphics.Rect rectI = object.getBoundingBox();

            if (MainActivity.DETECTION_MODE == ObjectDetectorOptions.SINGLE_IMAGE_MODE)
                setIdForFrame(
                        (frameNum - fq + 1) % fq,
                        rectI,
                        id
                );
            else
                getIdForFrame(
                        (frameNum - fq + 1) % fq,
                        rectI,
                        id
                );
            int speed;

            speed = getGridSpeed(rectI);
            System.out.println("### speed = " + speed);
            speed = updateObjectsSpeed(frameNum, id, speed);

            android.graphics.Rect scaledBBox = new android.graphics.Rect(
                    (int) (rectI.left * sx),
                    (int) (rectI.top * sy),
                    (int) (rectI.right * sx),
                    (int) (rectI.bottom * sy)
            );
            System.out.println("### speed = " + speed);
            int twoSpeed = getTwoPointsSpeed(id, rectI);
            System.out.println("### twoSpeed= " + twoSpeed + ", speed= " + speed);

            if (twoSpeed != -1)
                if (twoSpeed < 20)
                    speed = 0;

            List<DetectedObject.Label> labels = object.getLabels();
            if (!labels.isEmpty()) {
                if (Float.parseFloat(labels.get(0).getText()) != -1)
                    speed = (int) Float.parseFloat(labels.get(0).getText());
            }
            speed = min(120, speed);
            draw(canvas, scaledBBox, speed, id);
        }
    }

    public int predict(double xp1, double yp1, double xp2, double yp2) {
        // Step 11: Initial computations using the first data point
        double x_p = (1.0 / (100.0 * lambda)) * (xp1 - (imW / 2.0));
        double y_p = (1.0 / (100.0 * lambda)) * ((imH / 2.0) - yp1);

        System.out.println("!@# imW= " + imW + ", imH = " + imH);
        System.out.println("!@# xp1= " + xp1 + ", yp1 = " + yp1);
        System.out.println("!@# xp2= " + xp2 + ", yp2 = " + yp2);
        System.out.println("!@# x_p= " + x_p + ", y_p = " + y_p);

        double denominator = (a * x_p + b * y_p + c * f);
        if (denominator == 0) {
            System.err.println("Denominator in t calculation is zero. Cannot divide by zero.");
            return -1;
        }

        double t = (a * x1 + b * y1 + c * z1) / denominator;
        System.out.println("!@# t= " + t);

        double[][] wData = {
                {x_p * t},
                {y_p * t},
                {f * t}
        };
        RealMatrix w = MatrixUtils.createRealMatrix(wData);

        // Step 12: Compute Pos = A_inv * w
        RealMatrix Pos = A_inv.multiply(w);
        double x_real = Pos.getEntry(0, 0);
        double y_real = Pos.getEntry(1, 0);
        // double H_real = Pos.getEntry(2, 0); // Uncomment if needed

        // Compute x_p and y_p for the next data point
        x_p = (1.0 / (100.0 * lambda)) * (xp2 - (imW / 2.0));
        y_p = (1.0 / (100.0 * lambda)) * ((imH / 2.0) - yp2);
        System.out.println("!@# x_p2= " + x_p + ", y_p2= " + y_p);

        denominator = (a * x_p + b * y_p + c * f);
        if (denominator == 0) {
            System.err.println("Denominator in t calculation is zero.");
            return -1;
        }

        t = (a * x1 + b * y1 + c * z1) / denominator;

        // Update w matrix
        wData[0][0] = x_p * t;
        wData[1][0] = y_p * t;
        wData[2][0] = f * t;
        w = MatrixUtils.createRealMatrix(wData);

        // Compute Pos = A_inv * w
        Pos = A_inv.multiply(w);
        double x_real1 = Pos.getEntry(0, 0);
        double y_real1 = Pos.getEntry(1, 0);

        // Compute differences and distance
        double x_o = x_real1 - x_real;
        double y_o = y_real1 - y_real; // Corrected from x_real1 - y_real
        double L_val = Math.sqrt(x_o * x_o + y_o * y_o);
        System.out.println("!@# L_val= " + L_val);

        return (int) ((L_val * 3600.0) / 1000.0);
    }

    private void predictSpeed(Mat frameGray) {

        try {
            MatOfPoint prevPtsMat_ = new MatOfPoint();

            Imgproc.goodFeaturesToTrack(prevGray, prevPtsMat_, 150, 0.1, 5, new Mat(), 7, false, 0.04);

            MatOfPoint2f prevPtsMat = new MatOfPoint2f(prevPtsMat_.toArray());

            MatOfPoint2f nextPts = new MatOfPoint2f();
            // Calculate optical flow using Lucas-Kanade method
            MatOfByte status = new MatOfByte();
            MatOfFloat err = new MatOfFloat();
            Video.calcOpticalFlowPyrLK(
                    prevGray, frameGray, prevPtsMat,
                    nextPts, status, err, winSize,
                    0, criteria);

            byte[] StatusArr = status.toArray();
            Point[] p0Arr = prevPtsMat.toArray();
            Point[] p1Arr = nextPts.toArray();

            for (int i = 0; i < StatusArr.length; i++) {
                if (StatusArr[i] == 1) {
                    Point newPt = p0Arr[i];

                    Point oldPt = p1Arr[i];

                    double a = newPt.x;
                    double b = newPt.y;
                    double c = oldPt.x;
                    double d = oldPt.y;

                    double pixelSpeed = Math.sqrt(Math.pow(a - c, 2) + Math.pow(b - d, 2));

                    if (pixelSpeed > 15) {
                        float predictedSpeed = predict(a, b, c, d); // TODO: 31.10.24 apply fps
                        List<Integer> gridSpeed = new ArrayList<>();
                        gridSpeed.add((int) (a));
                        gridSpeed.add((int) (b));
                        gridSpeed.add((int) predictedSpeed);
                        gridSpeeds.add(gridSpeed);
                    }
                }
            }

        } catch (Exception e) {
            System.out.println(e);
        }

//        prevGray = frameGray.clone();

        // Return the processed frame
    }

    public void setImSize(int imW, int imH) {
        this.imW = imW;
        this.imH = imH;
    }


//    List<List<Integer>> gridSpeeds;
//    private void gridSpeedsInit() {
//        gridSpeeds = new ArrayList<>();
//    }
//
//    private int getGridSpeed(Rect rect) {
//        int sumSpeed = 0;
//        int cnt = 0;
//        for (List<Integer> gridSpeed: gridSpeeds) {
//            int x = gridSpeed.get(0);
//            int y = gridSpeed.get(1);
//            if (rect.contains(x, y)) {
//                cnt++;
//                sumSpeed += gridSpeed.get(2);
//            }
//        }
//
//        if (cnt == 0)
//            return 0;
//        return sumSpeed/cnt;
//    }
}
