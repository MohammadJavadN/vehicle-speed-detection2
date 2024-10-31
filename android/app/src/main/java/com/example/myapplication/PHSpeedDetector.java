package com.example.myapplication;

import static com.example.myapplication.OptSpeedDetector.getGridSpeed;
import static com.example.myapplication.OptSpeedDetector.getTwoPointsSpeed;
import static com.example.myapplication.OptSpeedDetector.gridSpeeds;
import static com.example.myapplication.OptSpeedDetector.gridSpeedsInit;
import static java.lang.Math.min;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.view.View;
import android.widget.EditText;

import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
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

public class PHSpeedDetector extends SpeedDetector {
    final static int fq = 2;
    private int imW;
    private int imH;
    private int frameNum = 0;
    private Mat prevGray;
    private TermCriteria criteria;
    private Size winSize;
    private List<Mat> grayFrames;
    public PHSpeedDetector() {
        super(fq);
    }

    double CP, CI;
    RealVector h;
    RealMatrix H_inverse, H;
    double t = 1, s = 1;
    double meanX, meanY, meanU, meanV;
    boolean initiated;

    public void init(double[] x, double[] y, double[] u, double[] v) {

        meanX = (x[0] + x[1] + x[2] + x[3]) / 4;
        meanY = (y[0] + y[1] + y[2] + y[3]) / 4;
        meanU = (u[0] + u[1] + u[2] + u[3]) / 4;
        meanV = (v[0] + v[1] + v[2] + v[3]) / 4;

        double sxy = 0, suv = 0;
        for (int i = 0; i < 4; i++) {
            x[i] = x[i] - meanX;
            y[i] = y[i] - meanY;
            u[i] = u[i] - meanU;
            v[i] = v[i] - meanV;

            sxy += sqrt(x[i]*x[i] + y[i]*y[i]);
            suv += sqrt(u[i]*u[i] + v[i]*v[i]);
        }

        CP = 4 * sqrt(2) / sxy;
        CI = 4 * sqrt(2) / suv;

        for (int i = 0; i < 4; i++) {
            x[i] = CP * x[i];
            y[i] = CP * y[i];
            u[i] = CI * u[i];
            v[i] = CI * v[i];
        }

        // Define the 8x8 matrix K
        double[][] matrixData = {
                {x[0],  y[0],   1,      0,      0,      0,  -x[0]*u[0],     -y[0]*u[0]},
                {x[1],  y[1],   1,      0,      0,      0,  -x[1]*u[1],     -y[1]*u[1]},
                {x[2],  y[2],   1,      0,      0,      0,  -x[2]*u[2],     -y[2]*u[2]},
                {x[3],  y[3],   1,      0,      0,      0,  -x[3]*u[3],     -y[3]*u[3]},
                {0,     0,      0,      x[0],   y[0],   1,  -x[0]*v[0],     -y[0]*v[0]},
                {0,     0,      0,      x[1],   y[1],   1,  -x[1]*v[1],     -y[1]*v[1]},
                {0,     0,      0,      x[2],   y[2],   1,  -x[2]*v[2],     -y[2]*v[2]},
                {0,     0,      0,      x[3],   y[3],   1,  -x[3]*v[3],     -y[3]*v[3]}
        };

        // Define the 8x1 vector b
        double[] vectorData = {
                u[0],
                u[1],
                u[2],
                u[3],
                v[0],
                v[1],
                v[2],
                v[3]
        };

        // Create RealMatrix and RealVector objects for K and b
        RealMatrix K = new Array2DRowRealMatrix(matrixData);
        RealVector b = new ArrayRealVector(vectorData);

        // Solve for h
        DecompositionSolver solver = new LUDecomposition(K).getSolver();
        h = solver.solve(b);

        // Construct the 3x3 projective transformation matrix H
        double[][] matrixDataH = {
                { h.getEntry(0), h.getEntry(1), h.getEntry(2) },
                { h.getEntry(3), h.getEntry(4), h.getEntry(5) },
                { h.getEntry(6), h.getEntry(7), 1.0 }
        };

        // Create RealMatrix object for H
        H = new Array2DRowRealMatrix(matrixDataH);

        // Calculate the inverse of H
        H_inverse = new LUDecomposition(H).getSolver().getInverse();

        t = s * (h.getEntry(6) * x[0] + h.getEntry(7) * y[0] + 1.0);

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

        initiated = true;
    }

    private double calculateRealL(double u1, double v1, double u2, double v2){
        double u_c1 = CI * (u1 - meanU);
        double v_c1 = CI * (v1 - meanV);

        double u_c2 = CI * (u2 - meanU);
        double v_c2 = CI * (v2 - meanV);

        t = 1;
        double[] vectorData1 = {u_c1*t, v_c1*t, t};
        RealVector i1 = new ArrayRealVector(vectorData1);

        RealVector p1 = H_inverse.operate(i1);

        double s1 = p1.getEntry(2);
        double x_c1 = p1.getEntry(0) / s1;
        double y_c1 = p1.getEntry(1) / s1;

        double[] vectorData2 = {u_c2*t, v_c2*t, t};
        RealVector i2 = new ArrayRealVector(vectorData2);

        RealVector p2 = H_inverse.operate(i2);

        double s2 = p2.getEntry(2);
        double x_c2 = p2.getEntry(0) / s2;
        double y_c2 = p2.getEntry(1) / s2;

        double x1 = (x_c1 / CP) + meanX;
        double y1 = (y_c1 / CP) + meanY;
        double x2 = (x_c2 / CP) + meanX;
        double y2 = (y_c2 / CP) + meanY;

        return sqrt(pow(x2-x1, 2) + pow(y2-y1, 2));
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

    public int predict(double u1, double v1, double u2, double v2) {
        if (!initiated)
            return -1;
        double L = calculateRealL(u1, v1, u2, v2);
        double T = 1.0;
        return (int) (L/T * 3600.0 / 1000.0); // TODO: 29.10.24 calc T
    }

    private void predictSpeed(Mat frameGray) {

        try {
            MatOfPoint prevPtsMat_ = new MatOfPoint();

            Imgproc.goodFeaturesToTrack(prevGray, prevPtsMat_, 1000, 0.3, 7, new Mat(), 7, false, 0.04);

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


}
