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
import java.util.List;
import java.util.Random;

public class TOFSpeedDetector {

    final static int fq = 6;
    final static int ff = 4;
    final static int ptsNum = 50;
    private static final Size WIN_SIZE = new Size(10, 10);
    private static final TermCriteria CRITERIA = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
            200,
            0.001);
    private static final int MAX_CORNERS = 20;
    private final TensorBuffer speedInputFeature;
    private final List<Mat> frames;
    private final List<List<DetectedObject>> listOfObjList;
    private int frameNum = 0;
    private final SpeedPredictionModel speedPredictionModel;

    private Paint rectPaint, textPaint;

    public TOFSpeedDetector(TensorBuffer speedInputFeature, SpeedPredictionModel speedPredictionModel) {
        this.speedInputFeature = speedInputFeature;
        this.speedPredictionModel = speedPredictionModel;
        frames = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            frames.add(new Mat());
        }

        listOfObjList = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            listOfObjList.add(new ArrayList<>());
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
    Bitmap bitmap;
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
        double sx = 1, sy = 1;
        double hf = frame.height();
        double wf = frame.width();
        if (canvas != null) {
            // Render the frame onto the canvas
            int w = canvas.getWidth();
            int h = canvas.getHeight();

            sx = w /wf;
            sy = h /hf;

            System.out.println("sx = " + sx);

            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, w, h, false);
            canvas.drawBitmap(scaledBitmap, 0, 0, null);
        } else return;

        for (DetectedObject object : listOfObjList.get((frameNum - fq) % fq)) {
            Rect bBox = new Rect(object.getBoundingBox().left, object.getBoundingBox().top,
                    object.getBoundingBox().right, object.getBoundingBox().bottom);
            System.out.println("### " + bBox);

            Random random = new Random();

            List<Point> prevPts = new ArrayList<>();
            for (int i = 0; i < ptsNum; i++) {
                double x = random.nextInt(bBox.width) + bBox.tl().y;
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
                        prevGray, frameGray, prevPtsMat, nextPts, status, errors, WIN_SIZE, 5, CRITERIA);

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
                canvas.drawText(Float.toString((float) speed / 10), scaledBBox.left+16, scaledBBox.top+160, textPaint);
            }

        }

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
        //>>> formatted_means = ", ".join("{:.6f}".format(value) for value in sc.var_)
        //>>> formatted_vars = ", ".join("{:.6f}".format(value) for value in sc.mean_)
        //>>> formatted_means
        double[] MEANS = new double[]{
                367.753784, 294.440142, 0.033945, 0.033927,
                0.033940, 0.034647, 0.034541, 0.034742,
                0.035058, 0.035361, 0.035094, 0.034963,
                0.035104, 0.035358, 0.035206, 0.035094,
                0.035602, 0.035369, 0.035224, 0.034945,
                0.035046, 0.035158
        };

        double[] VARS = new double[]{
                16921.964456, 8168.900612, 0.001364, 0.001229,
                0.001149, 0.001353, 0.001362, 0.001310, 0.001402,
                0.001564, 0.001258, 0.001300, 0.001260, 0.001649,
                0.001213, 0.001343, 0.001653, 0.001472, 0.001489,
                0.001281, 0.001355, 0.001288
        };

        double[] normalized = new double[VARS.length];
        for (int i = 0; i < normalized.length; i++) {
            normalized[i] = (inputData[i] - MEANS[i]) / VARS[i];
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
