package com.example.myapplication;

import static java.lang.Math.abs;
import static java.lang.Math.min;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.security.keystore.StrongBoxUnavailableException;

import com.example.myapplication.ml.SpeedPredictionModel;
import com.example.myapplication.ml.SpeedPredictionModelSideView;
import com.example.myapplication.ml.SpeedPredictionTopViewNoPlateModel;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class OptSpeedDetector extends SpeedDetector {

    final static int fq = 2;
    private static final Map<Integer, Rect> twoPointsSpeed = new HashMap<>();
    static List<List<Integer>> gridSpeeds;
    final boolean isSide;
    float[] laneSpeeds = {0, 0, 0, 0};
    Mat mask;
    private int imW;
    private int imH;
    private int frameNum = 0;
    private Mat prevGray;
    private TermCriteria criteria;
    private Size winSize;
    private SpeedPredictionTopViewNoPlateModel speedPredictionTopViewNoPlateModel;
    private SpeedPredictionModelSideView speedPredictionModelSideView;
    private TensorBuffer topSpeedInputFeature2;
    private TensorBuffer sideSpeedInputFeature;
    private List<Mat> grayFrames;

    public OptSpeedDetector(TensorBuffer sideSpeedInputFeature, SpeedPredictionModelSideView speedPredictionModelSideView) {
        super(fq);
        this.sideSpeedInputFeature = sideSpeedInputFeature;
        this.speedPredictionModelSideView = speedPredictionModelSideView;
        isSide = true;
        init();
    }

    public OptSpeedDetector(TensorBuffer topSpeedInputFeature2, SpeedPredictionTopViewNoPlateModel speedPredictionTopViewNoPlateModel) {
        super(fq);
        this.topSpeedInputFeature2 = topSpeedInputFeature2;
        this.speedPredictionTopViewNoPlateModel = speedPredictionTopViewNoPlateModel;
        isSide = false;
        init();
    }

    public static int getTwoPointsSpeed(int id, Rect rect2) {
        int speed = -1;
        int x1, y1, w1, h1;
        int x2, y2, w2, h2;
        if (twoPointsSpeed.containsKey(id)) {
            Rect rect1 = twoPointsSpeed.get(id);
            assert rect1 != null;
            x1 = rect1.left;
            y1 = rect1.top;
            w1 = rect1.width();
            h1 = rect1.height();
            x2 = rect2.left;
            y2 = rect2.top;
            w2 = rect2.width();
            h2 = rect2.height();
            speed = getTwoPointsSpeed2(x1, y1, w1, h1, x2, y2, w2, h2);
        }
        twoPointsSpeed.put(id, rect2);
        return speed;
    }

    private static int getTwoPointsSpeed2(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2) {
        double dPixel = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
        double w = (double) (w1 + w2) / 2;
        double h = (double) (h1 + h2) / 2;
        double ix = w / h;
        double ixp = ix / 2.9;
        double ppm = (w * ixp) / 4.2;
        double dMeters = dPixel / ppm;
        double timeConstant = 30 * 3.6;
        double speed = dMeters * timeConstant * 4;
        return (int) speed;
    }

    public static int getGridSpeed(Rect rect) {
        System.out.println("@@@@@@ rect= " + rect);

        int sumSpeed = 0;
        int cnt = 0;
        for (List<Integer> gridSpeed : gridSpeeds) {
            int x = gridSpeed.get(0);
            int y = gridSpeed.get(1);
            if (rect.contains(x, y)) {
                cnt++;
                sumSpeed += gridSpeed.get(2);
            }
        }

        if (cnt == 0)
            return 0;
        return sumSpeed / cnt;
    }

    public static void gridSpeedsInit() {
        gridSpeeds = new ArrayList<>();
    }

    private void init() {
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
        if (isSide) {
            predictSide(grayFrames.get((frameNum - fq + 1) % fq), frames.get((frameNum - fq) % fq));
        } else {
            predictTop(grayFrames.get((frameNum - fq + 1) % fq), frames.get((frameNum - fq) % fq));
        }

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
            if (isSide)
                speed = (int) laneSpeeds[0];
            else
                speed = (int) laneSpeeds[getLane(rectI.centerX(), rectI.centerY()) - 1];

            speed = getGridSpeed(rectI);
            // Draw label text

            android.graphics.Rect scaledBBox = new android.graphics.Rect(
                    (int) (rectI.left * sx),
                    (int) (rectI.top * sy),
                    (int) (rectI.right * sx),
                    (int) (rectI.bottom * sy)
            );
            int twoSpeed = getTwoPointsSpeed(id, rectI);
            System.out.println("### twoSpeed= " + twoSpeed + ", speed= " + speed);
            if (!isSide)
                twoSpeed = speed;
            if (twoSpeed != -1)
                if (speed == 0)
                    speed = twoSpeed;
            speed = (twoSpeed + speed) / 2;
            speed = updateObjectsSpeed(frameNum, id, speed);
            System.out.println("### speed = " + speed);

            if (twoSpeed != -1)
                if (twoSpeed < 15)
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

    public int getLane(int x, int y) {
        if (x < 0.307 * imW) {
            return 1;
        }
        if (x < 0.49 * imW + 0.37 * y) {
            return 2;
        }
        if (x < 0.6875 * imW + 0.8 * y) {
            return 3;
        }
        return 4;
    }

    public void predictTop(Mat frameGray, Mat frame) {

        for (int i = 0; i < 4; i++) {
            laneSpeeds[i] = 0;
        }

        try {
            int[] cntOfLaneSpeeds = {0, 0, 0, 0};

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
            if (showOpt)
                mask = Mat.zeros(frame.size(), frame.type());

            for (int i = 0; i < StatusArr.length; i++) {
                if (StatusArr[i] == 1) {
                    Point newPt = p0Arr[i];

                    Point oldPt = p1Arr[i];

                    double a = newPt.x;
                    double b = newPt.y;
                    double c = oldPt.x;
                    double d = oldPt.y;

//                    if (b < 0.25 * imH || d < b || abs(a-c) > (double) imW /100 || abs(b-d) > imH*0.14)
//                        continue;

                    int lane = getLane((int) newPt.x, (int) newPt.y);

                    double pixelSpeed = Math.sqrt(Math.pow(a - c, 2) + Math.pow(b - d, 2));

                    double[] input_data = {a, b, pixelSpeed};

                    if (showOpt) {
                        Imgproc.line(mask, newPt, oldPt, new Scalar(255, 0, 0), 40);
                        Imgproc.circle(mask, newPt, (5 * imW) / 1920, new Scalar(255, 0, 0), -1);
                    }

                    if (pixelSpeed > 15) {
                        System.out.println("****** input_data" + Arrays.toString(input_data));
                        System.out.println("****** imW" + imW);
                        float predictedSpeed = predictSpeedNoPlate(input_data, false);
                        predictedSpeed = (float) (predictedSpeed * Math.pow(1920 / imW, 0.08) * 1.2);
                        laneSpeeds[lane - 1] += predictedSpeed;
                        cntOfLaneSpeeds[lane - 1]++;
                        List<Integer> gridSpeed = new ArrayList<>();
                        gridSpeed.add((int) (a));
                        gridSpeed.add((int) (b));
                        gridSpeed.add((int) predictedSpeed);
                        gridSpeeds.add(gridSpeed);
//                                Imgproc.line(mask, newPt, oldPt, new Scalar(255, 0, 0), 4 * imW / 1920);
//                                Imgproc.circle(mask, newPt, (5 * imW) / 1920, new Scalar(255, 0, 0), -1);
                    }
                }
            }

            for (int lane = 0; lane < 4; lane++) {
                // Draw lines and circles
                int speed = (int) (laneSpeeds[lane] / cntOfLaneSpeeds[lane]);

                if (speed < 5)
                    speed = 0;
                laneSpeeds[lane] = speed;
            }

            if (showOpt)
                Core.add(frame, mask, frame);

        } catch (Exception e) {
            System.out.println(e);
        }

//        prevGray = frameGray.clone();

        // Return the processed frame
    }

    public void predictSide(Mat frameGray, Mat frame) {

        laneSpeeds[0] = 0;

        try {
            int[] cntOfLaneSpeeds = {0, 0, 0, 0};

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
            if (showOpt)
                mask = Mat.zeros(frame.size(), frame.type());

            for (int i = 0; i < StatusArr.length; i++) {
                if (StatusArr[i] == 1) {
                    Point newPt = p0Arr[i];

                    Point oldPt = p1Arr[i];

                    double a = newPt.x;
                    double b = newPt.y;
                    double c = oldPt.x;
                    double d = oldPt.y;

//                    if (b < 0.25 * imH || d < b || abs(a-c) > (double) imW /100 || abs(b-d) > imH*0.14)
//                        continue;

                    double pixelSpeed = Math.sqrt(Math.pow(a - c, 2) + Math.pow(b - d, 2));

                    if (showOpt) {
                        Imgproc.line(mask, newPt, oldPt, new Scalar(255, 0, 0), 4);
                        Imgproc.circle(mask, newPt, (5 * imW) / 1920, new Scalar(255, 0, 0), -1);
                    }

                    // Get output tensor (predicted_speed_function)
                    if (pixelSpeed > 15) {
                        double[] input_data = {a, b, pixelSpeed};
                        float predictedSpeed = predictSpeedNoPlate(input_data, true);
                        laneSpeeds[0] += predictedSpeed;
                        cntOfLaneSpeeds[0]++;
                        List<Integer> gridSpeed = new ArrayList<>();
                        gridSpeed.add((int) (a));
                        gridSpeed.add((int) (b));
                        gridSpeed.add((int) predictedSpeed);
                        gridSpeeds.add(gridSpeed);
                    }
                }
            }
            // Draw lines and circles
            laneSpeeds[0] = (int) (laneSpeeds[0] / cntOfLaneSpeeds[0]);

            if (laneSpeeds[0] < 20)
                laneSpeeds[0] = 0;

            if (showOpt)
                Core.add(frame, mask, frame);

        } catch (Exception e) {
            System.out.println(e);
        }

//        prevGray = frameGray.clone();

        // Return the processed frame
    }

    private float predictSpeedNoPlate(double[] inputData, boolean sideView) {
        if (sideView) {
            double MEAN_A = 936.88328756;
            double MEAN_B = 256.01818182;
            double MEAN_P = 36.73230519;
            double VAR_A = 255.618194466;
            double VAR_B = 111.727442858;
            double VAR_P = 634.69753378;
            double[] normalized;
            normalized = new double[]{(inputData[0] - MEAN_A) / VAR_A,
                    (inputData[1] - MEAN_B) / VAR_B,
                    (inputData[2] - MEAN_P) / VAR_P};
            sideSpeedInputFeature.loadBuffer(doubleToByteBuffer(normalized));

            // Runs model inference and gets result.
            SpeedPredictionModelSideView.Outputs outputs = speedPredictionModelSideView.process(sideSpeedInputFeature);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            return outputFeature0.getFloatValue(0);
        } else {
            double MEAN_A = 917.25111607;
            double MEAN_B = 393.7375372;
            double MEAN_P = 48.96527362;
            double VAR_A = 516.638571083;
            double VAR_B = 299.871522795;
            double VAR_P = 26.931929277;
            double[] normalized;
            normalized = new double[]{(inputData[0] - MEAN_A) / VAR_A,
                    (inputData[1] - MEAN_B) / VAR_B,
                    (inputData[2] - MEAN_P) / VAR_P};
            topSpeedInputFeature2.loadBuffer(doubleToByteBuffer(normalized));

            // Runs model inference and gets result.
            SpeedPredictionTopViewNoPlateModel.Outputs outputs = speedPredictionTopViewNoPlateModel.process(topSpeedInputFeature2);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            return outputFeature0.getFloatValue(0);
        }
    }


}
