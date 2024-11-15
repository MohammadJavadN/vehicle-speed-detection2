package com.example.myapplication;

import static com.example.myapplication.OptSpeedDetector.getGridSpeed;
import static com.example.myapplication.OptSpeedDetector.getTwoPointsSpeed;
import static com.example.myapplication.OptSpeedDetector.gridSpeeds;
import static com.example.myapplication.OptSpeedDetector.gridSpeedsInit;
import static java.lang.Math.log10;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.sqrt;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Rect;

import com.example.myapplication.ml.SpeedPredictionModelSideView;
import com.example.myapplication.ml.SpeedPredictionTopViewNoPlateModel;
import com.google.mlkit.vision.GraphicOverlay;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions;

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
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RLSpeedDetector extends SpeedDetector {
    final static int fq = 2;
    private static final Map<Integer, Rect> twoPointsSpeed = new HashMap<>();
    static List<List<Integer>> gridSpeeds;
    float[] laneSpeeds = {0, 0, 0, 0};
    Mat mask;
    private int imW;
    private int imH;
    private int frameNum = 0;
    private Mat prevGray;
    private final TermCriteria criteria;
    private final Size winSize;
    private SpeedPredictionTopViewNoPlateModel speedPredictionTopViewNoPlateModel;
    private SpeedPredictionModelSideView speedPredictionModelSideView;
    private TensorBuffer topSpeedInputFeature2;
    private TensorBuffer sideSpeedInputFeature;
    private final List<Mat> grayFrames;

    public RLSpeedDetector() {
        super(fq);
        grayFrames = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            grayFrames.add(new Mat());
        }
        winSize = new Size(10, 10);
        criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
                2000,
                0.001);
        prevGray = null;
        frameNum = 0;
    }

    public static int getGridSpeed(Rect rect) {
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

    public void setImSize(int imW, int imH) {
        this.imW = imW;
        this.imH = imH;
    }

    @Override
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

//            if (twoSpeed != -1)
//                if (twoSpeed < 20)
//                    speed = 0;

            List<DetectedObject.Label> labels = object.getLabels();
            if (!labels.isEmpty()) {
                if (Float.parseFloat(labels.get(0).getText()) != -1)
                    speed = (int) Float.parseFloat(labels.get(0).getText());
            }
            speed = min(120, speed);
            draw(canvas, scaledBBox, speed, id);
        }
    }

    public float predict(double xp1, double yp1, double xp2, double yp2, int df) {
        System.out.println("*** RL predict: ");
        System.out.println("*** xp1=" + xp1 + ", yp1=" + yp1 + ", xp2=" + xp2 + ", yp2=" + yp2);
        Point tmpSpeed = GraphicOverlay.getOverlayInstance().roadLine.calculateSignSpeed(
                new Point(xp1 / imW, yp1 / imH),
                new Point(xp2 / imW, yp2 / imH)
        );
        System.out.println("*** xp1=" + xp1/ imW + ", yp1=" + yp1/ imH + ", xp2=" + xp2/ imW + ", yp2=" + yp2/ imH);

        System.out.println("df=" + df);
        double speedX = tmpSpeed.x/df, speedY = tmpSpeed.y/df;
        System.out.println("speedX=" + speedX + ", speedY=" + speedY);
        float speed1 = (float) sqrt(speedX * speedX + speedY * speedY) * 0.6f;
        System.out.println("speed1=" + speed1);

        double val1 = 90 - 40 * log10(900 / (speed1 - 10) - 12);
        if (speed1 < 14)
            val1 = 0;
        else if (speed1 > 84)
            val1 = 130;
        System.out.println("val1=" + val1);
        double val2 = 50 * log10(speed1 / 2);
        if (speed1 < 2)
            val2 = 0;
        System.out.println("val2=" + val2);
        float speed = (float) min(max(0, val1), val2);
        System.out.println("speed=" + speed);
//        if (speed < 10 || ((int) speed) < 20) {
//            speed = -1;
//        }
        return speed/25;
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
                        float predictedSpeed = predict(a, b, c, d, 1); // TODO: 31.10.24 apply fps
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
