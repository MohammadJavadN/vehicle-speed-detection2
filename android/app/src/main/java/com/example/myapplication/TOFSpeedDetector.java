package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.example.myapplication.ml.SpeedPredictionModel;
import com.google.mlkit.vision.objects.DetectedObject;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class TOFSpeedDetector  {

    final static int fq = 10;
    final static int ff = 5;
    final static int ptsNum = 50;
    private static final Size WIN_SIZE = new Size(100, 40);
    private static final TermCriteria CRITERIA = new TermCriteria(TermCriteria.COUNT | TermCriteria.EPS,
            10,
            0.003);
    private static final int MAX_LEVEL = 10;
    private static final int MAX_CORNERS = 20;
    private final TensorBuffer speedInputFeature;
    private final List<Mat> frames;
    private final List<Bitmap> bitmaps;
    private final List<List<DetectedObject>> listOfObjList;
    private final List<List<STrack>> listOfSTrackList;
    private final SpeedPredictionModel speedPredictionModel;
    private final Paint rectPaint;
    private final Paint textPaint;
    Bitmap bitmap;
    private int frameNum = 0;

    public TOFSpeedDetector(TensorBuffer speedInputFeature, SpeedPredictionModel speedPredictionModel) {
        this.speedInputFeature = speedInputFeature;
        this.speedPredictionModel = speedPredictionModel;
        frames = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            frames.add(new Mat());
        }

        bitmaps = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            bitmaps.add(null);
        }

        listOfObjList = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            listOfObjList.add(new ArrayList<>());
        }

        listOfSTrackList = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            listOfSTrackList.add(new ArrayList<>());
        }

        rectPaint = new Paint();
        rectPaint.setColor(Color.RED);
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(8.0f);

        textPaint = new Paint();
        textPaint.setTextSize(160);
        textPaint.setColor(Color.RED);
        textPaint.setStyle(Paint.Style.STROKE);
        textPaint.setStrokeWidth(16.0f);

    }

    public void detectSpeeds(Mat frame, List<DetectedObject> detectedObjects, Canvas canvas) {


        frames.set(frameNum % fq, frame.clone());
        listOfObjList.set(frameNum % fq, new ArrayList<>(detectedObjects));
        frameNum++;

        if (frameNum < fq) {
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

        bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
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
            Rect bBox = new Rect(object.getBoundingBox().left, object.getBoundingBox().top,
                    object.getBoundingBox().right, object.getBoundingBox().bottom);

            Random random = new Random();

            List<Point> prevPts = new ArrayList<>();
            for (int i = 0; i < ptsNum; i++) {
                double x = random.nextInt(bBox.width) + bBox.tl().x;
                double y = random.nextInt(bBox.height) + bBox.tl().y;
                prevPts.add(new Point(x, y));
            }

            Mat prevGray = new Mat();
            Imgproc.cvtColor(frames.get((frameNum - fq) % fq), prevGray, Imgproc.COLOR_BGR2GRAY);

            MatOfPoint2f prevPtsMat = new MatOfPoint2f();
            prevPtsMat.fromList(prevPts);

            MatOfByte status = new MatOfByte();
            MatOfFloat errors = new MatOfFloat();

            // Calculate optical flow
            List<Double> distances = new ArrayList<>();
            distances.add((double) Math.abs(bBox.width));
            distances.add((double) Math.abs(bBox.height));

            for (int i = ff; i < fq - 1; i++) {
                Mat frameGray = new Mat();
                Imgproc.cvtColor(frames.get((frameNum - fq + i + 1) % fq), frameGray, Imgproc.COLOR_BGR2GRAY);

                MatOfPoint2f nextPts = new MatOfPoint2f();
                Video.calcOpticalFlowPyrLK(
                        prevGray, frameGray, prevPtsMat, nextPts, status, errors, WIN_SIZE, MAX_LEVEL, CRITERIA);

                if (!errors.empty()) {
                    float[] errorsArray = errors.toArray();
                    int[] sortedIndices = sortIndices(errorsArray);

                    // Get indices of top 20 best points
                    int[] top20Indices = new int[Math.min(MAX_CORNERS, sortedIndices.length)];
                    System.arraycopy(sortedIndices, 0, top20Indices, 0, top20Indices.length);

                    List<Point> bestPrevPts = new ArrayList<>();
                    List<Point> bestNextPts = new ArrayList<>();
                    for (int index : top20Indices) {
                        bestPrevPts.add(prevPtsMat.toList().get(index));
                        bestNextPts.add(nextPts.toList().get(index));
                    }

                    // Normalize points
                    Point minV = new Point(-1, -1);
                    Point maxV = new Point(-1, -1);
                    List<Point> prevPtsNormalized = normalizePoints(bestPrevPts, minV, maxV);
                    List<Point> nextPtsNormalized = normalizePoints(bestNextPts, minV, maxV);

                    // Calculate Euclidean distances
                    for (int j = 0; j < bestPrevPts.size(); j++) {
                        Point prevPt = prevPtsNormalized.get(j);
                        Point nextPt = nextPtsNormalized.get(j);
                        double distance = Math.sqrt(Math.pow(prevPt.x - nextPt.x, 2) + Math.pow(prevPt.y - nextPt.y, 2));
                        distances.add(distance);
                    }
                }


                double[] input_data = listToArray(distances);

                // Get output tensor (predicted_speed_function)
                float predictedSpeed = predictSpeed(input_data);

                int speed = (int) predictedSpeed;
                // Draw label text

                android.graphics.Rect scaledBBox = new android.graphics.Rect(
                        (int) (bBox.tl().x * sx),
                        (int) (bBox.tl().y * sy),
                        (int) (bBox.br().x * sx),
                        (int) (bBox.br().y * sy)
                );
                canvas.drawRect(scaledBBox, rectPaint);
                canvas.drawText(Float.toString((float) speed / 10), scaledBBox.left + 16, scaledBBox.top + 160, textPaint);
            }

        }

    }

    public Bitmap dest;
    public List<DetectedObject> detectSpeeds(
            Bitmap originalCameraImage,
            List<DetectedObject> detectedObjects
    ) {

        Mat frame = new Mat();
        Utils.bitmapToMat(originalCameraImage, frame);

        frames.set(frameNum % fq, frame.clone());
        bitmaps.set(frameNum % fq, originalCameraImage.copy(originalCameraImage.getConfig(), true));
        listOfObjList.set(frameNum % fq, new ArrayList<>(detectedObjects));
        frameNum++;

        if (frameNum < fq) {
            dest = originalCameraImage;
            return detectedObjects;
        }
        dest = bitmaps.get((frameNum - fq) % fq);
        List<DetectedObject> detectedObjectsOut = new ArrayList<>(detectedObjects.size());

        for (DetectedObject object : listOfObjList.get((frameNum - fq) % fq)) {
            Rect bBox = new Rect(object.getBoundingBox().left, object.getBoundingBox().top,
                    object.getBoundingBox().right, object.getBoundingBox().bottom);

            Random random = new Random();

            List<Point> prevPts = new ArrayList<>();
            for (int i = 0; i < ptsNum; i++) {
                double x = random.nextInt(bBox.width) + bBox.tl().x;
                double y = random.nextInt(bBox.height) + bBox.tl().y;
                prevPts.add(new Point(x, y));
            }

            Mat prevGray = new Mat();
            Imgproc.cvtColor(frames.get((frameNum - fq) % fq), prevGray, Imgproc.COLOR_BGR2GRAY);

            MatOfPoint2f prevPtsMat = new MatOfPoint2f();
            prevPtsMat.fromList(prevPts);

            MatOfByte status = new MatOfByte();
            MatOfFloat errors = new MatOfFloat();

            // Calculate optical flow
            List<Double> distances = new ArrayList<>();
            distances.add((double) Math.abs(bBox.width));
            distances.add((double) Math.abs(bBox.height));

            for (int i = ff; i < fq - 1; i++) {
                Mat frameGray = new Mat();
                Imgproc.cvtColor(frames.get((frameNum - fq + i + 1) % fq), frameGray, Imgproc.COLOR_BGR2GRAY);

                MatOfPoint2f nextPts = new MatOfPoint2f();
                Video.calcOpticalFlowPyrLK(
                        prevGray, frameGray, prevPtsMat, nextPts, status, errors, WIN_SIZE, MAX_LEVEL, CRITERIA);

                if (!errors.empty()) {
                    float[] errorsArray = errors.toArray();
                    int[] sortedIndices = sortIndices(errorsArray);

                    // Get indices of top 20 best points
                    int[] top20Indices = new int[Math.min(MAX_CORNERS, sortedIndices.length)];
                    System.arraycopy(sortedIndices, 0, top20Indices, 0, top20Indices.length);

                    List<Point> bestPrevPts = new ArrayList<>();
                    List<Point> bestNextPts = new ArrayList<>();
                    for (int index : top20Indices) {
                        bestPrevPts.add(prevPtsMat.toList().get(index));
                        bestNextPts.add(nextPts.toList().get(index));
                    }

                    // Normalize points
                    Point minV = new Point(-1, -1);
                    Point maxV = new Point(-1, -1);
                    List<Point> prevPtsNormalized = normalizePoints(bestPrevPts, minV, maxV);
                    List<Point> nextPtsNormalized = normalizePoints(bestNextPts, minV, maxV);

                    // Calculate Euclidean distances
                    for (int j = 0; j < bestPrevPts.size(); j++) {
                        Point prevPt = prevPtsNormalized.get(j);
                        Point nextPt = nextPtsNormalized.get(j);
                        double distance = Math.sqrt(Math.pow(prevPt.x - nextPt.x, 2) + Math.pow(prevPt.y - nextPt.y, 2));
                        distances.add(distance);
                    }
                }
            }

            double[] input_data = listToArray(distances);
            System.out.println(Arrays.toString(input_data));

            // Get output tensor (predicted_speed_function)
            float predictedSpeed = predictSpeed(input_data);

            int speed = (int) predictedSpeed;

            List<DetectedObject.Label> labels = object.getLabels();
            if (labels == null)
                labels = new ArrayList<>();
            labels.add(new DetectedObject.Label(Float.toString(speed), -1, -1));

            detectedObjectsOut.add(
                    new DetectedObject(
                            object.getBoundingBox(),
                            object.getTrackingId(),
                            labels
                    )
            );

        }
        return detectedObjectsOut;
    }

    static double[] xCoeffs;
    static double[] yCoeffs;
    static {
        // Define the range and number of points
        double yStart = 0.5;
        double yEnd = 0.8;
        int yNumPoints = 5;

        double xStart = 0.15;
        double xEnd = 0.85;
        int xNumPoints = 7;

        // Calculate y_coeffs and x_coeffs
        yCoeffs = new double[yNumPoints];
        xCoeffs = new double[xNumPoints];

        for (int i = 0; i < yNumPoints; i++) {
            yCoeffs[i] = yStart + i * (yEnd - yStart) / (yNumPoints - 1);
        }

        for (int i = 0; i < xNumPoints; i++) {
            xCoeffs[i] = xStart + i * (xEnd - xStart) / (xNumPoints - 1);
        }
    }
    public List<DetectedObject> detectSpeeds2(
            Bitmap originalCameraImage,
            List<STrack> detectedObjects
    ) {

        Mat frame = new Mat();
        Utils.bitmapToMat(originalCameraImage, frame);

        frames.set(frameNum % fq, frame.clone());
        bitmaps.set(frameNum % fq, originalCameraImage.copy(originalCameraImage.getConfig(), true));
        listOfSTrackList.set(frameNum % fq, new ArrayList<>(detectedObjects));
        frameNum++;
        System.out.println("*** detectedObjects.size() = " + detectedObjects.size());

        if (frameNum < fq) {
            dest = originalCameraImage;
            System.out.println("*** frameNum < fq : " + frameNum + " < " + fq);
            return new ArrayList<>();
        }
        dest = bitmaps.get((frameNum - fq) % fq);
        List<DetectedObject> detectedObjectsOut = new ArrayList<>(detectedObjects.size());

        for (STrack sTrack : listOfSTrackList.get((frameNum - fq) % fq)) {
            int id = sTrack.trackId;

            Rect rectI = sTrack.getRect();

            List<Double> distances = new ArrayList<>();
            distances.add((double) Math.abs(rectI.width));
            distances.add((double) Math.abs(rectI.height));

            final int STEP = 7;
            for (int j = STEP; j < fq; j+= STEP) {
                Rect rectJ = null;
                for (STrack sTrackJ : listOfSTrackList.get((frameNum - fq + j) % fq)) {
                    if (sTrackJ.trackId == id) {
                        rectJ = sTrackJ.getRect();
                    }
                }
                if (rectJ == null)
                    break;

                distances.add((double) ((rectJ.width - rectI.width) / rectI.width));
                distances.add((double) ((rectJ.height - rectI.height) / rectI.height));
            }

            if (distances.size() < fq / STEP + 1)
                continue;

            List<Point> prevPts = new ArrayList<>();
            for (double yC : yCoeffs) {
                for (double xC : xCoeffs) {
                    double x = rectI.tl().x + xC * (rectI.width);
                    double y = rectI.tl().y + yC * (rectI.height);
                    prevPts.add(new Point(x, y));
                }
            }

            Mat prevGray = new Mat();
            Imgproc.cvtColor(frames.get((frameNum - fq) % fq), prevGray, Imgproc.COLOR_BGR2GRAY);

            MatOfPoint2f prevPtsMat = new MatOfPoint2f();
            prevPtsMat.fromList(prevPts);

            MatOfByte status = new MatOfByte();
            MatOfFloat errors = new MatOfFloat();

            // Calculate optical flow
            for (int i = ff; i < fq - 1; i++) {
                Mat frameGray = new Mat();
                Imgproc.cvtColor(frames.get((frameNum - fq + i + 1) % fq), frameGray, Imgproc.COLOR_BGR2GRAY);

                MatOfPoint2f nextPts = new MatOfPoint2f();
                Video.calcOpticalFlowPyrLK(
                        prevGray, frameGray, prevPtsMat, nextPts, status, errors, WIN_SIZE, MAX_LEVEL, CRITERIA);

                List<Point> nextPtsList = nextPts.toList();
                double dfx = nextPtsList.get(0).x - prevPts.get(0).x;
                double dfy = nextPtsList.get(0).y - prevPts.get(0).y;
                // Calculate Euclidean distances
                for (int j = 0; j < prevPts.size(); j++) {
                    Point prevPt = prevPts.get(j);
                    Point nextPt = nextPtsList.get(j);
                    double distance = Math.sqrt(
                            Math.pow(nextPt.x - prevPt.x - dfx, 2) +
                                    Math.pow(nextPt.y - prevPt.y - dfy, 2)
                    ) / frame.width();
                    distances.add(distance);
                }

            }

            double[] input_data = listToArray(distances);
            System.out.println(Arrays.toString(input_data));

            // Get output tensor (predicted_speed_function)
            float predictedSpeed = predictSpeed(input_data);

            int speed = (int) predictedSpeed;

            List<DetectedObject.Label> labels = sTrack.getLabels();
            if (labels == null)
                labels = new ArrayList<>();
            labels.add(new DetectedObject.Label(Float.toString(speed), -1, -1));

            detectedObjectsOut.add(
                    new DetectedObject(
                            sTrack.getBoundingBox(),
                            sTrack.getTrackingId(),
                            labels
                    )
            );

        }
        return detectedObjectsOut;
    }

    public ByteBuffer doubleToByteBuffer(double[] data) {
        int bufferSize = data.length * 4;
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bufferSize);
        byteBuffer.order(ByteOrder.nativeOrder()); // Set the byte order to native

        // Check if the buffer has enough capacity to hold all the data
        if (byteBuffer.remaining() < bufferSize) {
            throw new RuntimeException("ByteBuffer does not have enough capacity to hold all the data");
        }

        // Put the data into the ByteBuffer
        for (double value : data) {
            byteBuffer.putFloat((float) value);
        }

        byteBuffer.rewind(); // Rewind the buffer to the beginning

        return byteBuffer;
    }

    private float predictSpeed(double[] inputData) {
        //>>> from joblib import load
        //>>> sc = load('speed_prediction_model/std_scaler.bin')
        //>>> formatted_means = ", ".join("{:.6f}".format(value) for value in sc.mean_)
        //>>> formatted_scales = ", ".join("{:.6f}".format(value) for value in sc.scale_)
        //>>> formatted_means
        double[] MEANS = new double[]{
                367.753784, 294.440142, 0.033945, 0.033927,
                0.033940, 0.034647, 0.034541, 0.034742,
                0.035058, 0.035361, 0.035094, 0.034963,
                0.035104, 0.035358, 0.035206, 0.035094,
                0.035602, 0.035369, 0.035224, 0.034945,
                0.035046, 0.035158
        };

        double[] SCALES = new double[]{
                130.084451, 90.381971, 0.036939, 0.035064, 0.033901,
                0.036780, 0.036903, 0.036189, 0.037449, 0.039547,
                0.035475, 0.036059, 0.035492, 0.040612, 0.034835,
                0.036642, 0.040652, 0.038372, 0.038593, 0.035797,
                0.036812, 0.035888
        };

        double[] normalized = new double[SCALES.length];
        for (int i = 0; i < SCALES.length; i++) {
            normalized[i] = (inputData[i] - MEANS[i]) / SCALES[i];
        }

        speedInputFeature.loadBuffer(doubleToByteBuffer(normalized));

        // Runs model inference and gets result.
        SpeedPredictionModel.Outputs outputs = speedPredictionModel.process(speedInputFeature);
        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        return outputFeature0.getFloatValue(0);

    }

    public double[] listToArray(List<Double> list) {
        // Create an array with the same size as the list
        double[] array = new double[list.size()];

        // Populate the array with values from the list
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }

        return array;
    }

    private int[] sortIndices(float[] errors) {
        Integer[] indices = new Integer[errors.length];
        for (int i = 0; i < errors.length; i++) {
            indices[i] = i;
        }
        java.util.Arrays.sort(indices, (a, b) -> Float.compare(errors[a], errors[b]));
        return java.util.Arrays.stream(indices).mapToInt(Integer::intValue).toArray();
    }

    private List<Point> normalizePoints(List<Point> points, Point minV, Point maxV) {
        List<Point> normalizedPoints = new ArrayList<>();
        if (points.isEmpty()) return normalizedPoints;

        Point minVal = new Point(Double.MAX_VALUE, Double.MAX_VALUE);
        Point maxVal = new Point(Double.MIN_VALUE, Double.MIN_VALUE);

        for (Point pt : points) {
            minVal.x = Math.min(minVal.x, pt.x);
            minVal.y = Math.min(minVal.y, pt.y);
            maxVal.x = Math.max(maxVal.x, pt.x);
            maxVal.y = Math.max(maxVal.y, pt.y);
        }

        Point s;
        if (minV.x == -1) {
            s = new Point(maxVal.x - minVal.x, maxVal.y - minVal.y);
        } else
            s = new Point(maxV.x - minV.x, maxV.y - minV.y);

        for (Point pt : points) {
            double x = (pt.x - minVal.x) / s.x;
            double y = (pt.y - minVal.y) / s.y;
            normalizedPoints.add(new Point(x, y));
        }
        if (minV.x == -1) {
            minV.y = minVal.y;
            minV.x = minVal.x;
        }
        if (maxV.x == -1) {
            maxV.y = maxVal.y;
            maxV.x = maxVal.x;
        }

        return normalizedPoints;
    }

}
