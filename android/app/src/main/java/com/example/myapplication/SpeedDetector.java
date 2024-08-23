package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.google.mlkit.vision.objects.DetectedObject;

import org.opencv.core.Mat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public abstract class SpeedDetector {

    private static final float TEXT_SIZE = 100.0f;
    private static final float STROKE_WIDTH = 15.0f;
    private static final int NUM_COLORS = 10;
    private static final int[][] COLORS =
            new int[][]{
                    // {Text color, background color}
                    {Color.BLACK, Color.WHITE},
                    {Color.WHITE, Color.MAGENTA},
                    {Color.BLACK, Color.LTGRAY},
                    {Color.WHITE, Color.RED},
                    {Color.WHITE, Color.BLUE},
                    {Color.WHITE, Color.DKGRAY},
                    {Color.BLACK, Color.CYAN},
                    {Color.BLACK, Color.YELLOW},
                    {Color.WHITE, Color.BLACK},
                    {Color.BLACK, Color.GREEN}
            };
    protected final List<Mat> frames;
    protected final List<List<DetectedObject>> listOfObjList;
    private final Paint rectPaint;
    private final Paint textPaint;
    private final Paint[] labelPaints;
    private final Paint[] boxPaints;
    private final Paint[] textPaints;
    protected final boolean showOpt = false;
    protected Bitmap bitmap;
    private int idGen = 1;

    public SpeedDetector(int fq) {
        listOfObjList = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            listOfObjList.add(new ArrayList<>());
        }
        frames = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            frames.add(new Mat());
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

        int numColors = COLORS.length;
        textPaints = new Paint[numColors];
        boxPaints = new Paint[numColors];
        labelPaints = new Paint[numColors];
        for (int i = 0; i < numColors; i++) {
            textPaints[i] = new Paint();
            textPaints[i].setColor(COLORS[i][0] /* text color */);
            textPaints[i].setTextSize(TEXT_SIZE);

            boxPaints[i] = new Paint();
            boxPaints[i].setColor(COLORS[i][1] /* background color */);
            boxPaints[i].setStyle(Paint.Style.STROKE);
            boxPaints[i].setStrokeWidth(STROKE_WIDTH);

            labelPaints[i] = new Paint();
            labelPaints[i].setColor(COLORS[i][1] /* background color */);
            labelPaints[i].setStyle(Paint.Style.FILL);
        }
    }

    /**
     * Calculates Intersection over Union (IoU) between two Rect objects.
     *
     * @param rect1 First rectangle.
     * @param rect2 Second rectangle.
     * @return The IoU value.
     */
    public static float calculateIoU(android.graphics.Rect rect1, android.graphics.Rect rect2) {
        // Compute the intersection rectangle
        android.graphics.Rect intersection = new android.graphics.Rect();
        if (!android.graphics.Rect.intersects(rect1, rect2)) {
            // No intersection
            return 0.0f;
        }
        intersection.set(
                Math.max(rect1.left, rect2.left),
                Math.max(rect1.top, rect2.top),
                Math.min(rect1.right, rect2.right),
                Math.min(rect1.bottom, rect2.bottom)
        );

        // Calculate the area of intersection
        int intersectionArea = intersection.width() * intersection.height();

        // Calculate the area of each rectangle
        int area1 = rect1.width() * rect1.height();
        int area2 = rect2.width() * rect2.height();

        // Calculate the area of union
        int unionArea = area1 + area2 - intersectionArea;

        // Calculate IoU
        return (float) intersectionArea / unionArea;
    }

    public void draw(Canvas canvas, android.graphics.Rect rect, float speed, int id) {
        // Decide color based on object tracking ID
        int colorID = id % NUM_COLORS;

        float textWidth = textPaints[colorID].measureText("ID: " + id + ", Speed: " + speed);
        float lineHeight = TEXT_SIZE + STROKE_WIDTH;
        float yLabelOffset = -lineHeight;

        float x0 = rect.left;
        float x1 = rect.right;
        rect.left = (int) Math.min(x0, x1);
        rect.right = (int) Math.max(x0, x1);

        canvas.drawRect(rect, boxPaints[colorID]);

        // Draws other object info.
        canvas.drawRect(
                rect.left - STROKE_WIDTH * 0,
                rect.top,
                rect.left + textWidth + (2 * STROKE_WIDTH),
                rect.top - yLabelOffset,
                labelPaints[colorID]);
//    yLabelOffset -= TEXT_SIZE;
        canvas.drawText(
                "ID: " + id + ", Speed: " + speed,
                rect.left,
                rect.top - .9f * yLabelOffset,
                textPaints[colorID]);
    }

    public abstract void detectSpeeds(Mat frame, List<DetectedObject> detectedObjects, Canvas canvas);

    int getId(DetectedObject object) {
        if (object.getTrackingId() != null)
            return object.getTrackingId();

        return idGen++;
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

    public double[] listToArray(List<Double> list) {
        // Create an array with the same size as the list
        double[] array = new double[list.size()];

        // Populate the array with values from the list
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }

        return array;
    }

    android.graphics.Rect setIdForFrame(int frameNum, android.graphics.Rect rect, int id) {
        double maxIOU = Double.MIN_VALUE;
        int selectedIdx = -1;
        DetectedObject selectedObj = null;
        List<DetectedObject> objects = listOfObjList.get(frameNum);
        for (int i = 0; i < objects.size(); i++) {
            DetectedObject object = objects.get(i);
            double iou = calculateIoU(object.getBoundingBox(), rect);
            System.out.println("i = " + i + ", iou = " + iou);
            if (iou > maxIOU) {
                selectedIdx = i;
                selectedObj = object;
                maxIOU = iou;
            }
        }

        System.out.println("maxIOU = " + maxIOU);
        double THRESH = 0.5;
        if (maxIOU <= THRESH)
            return null;

        objects.set(
                selectedIdx,
                new DetectedObject(
                        selectedObj.getBoundingBox(),
                        id,
                        selectedObj.getLabels()
                )
        );

        return selectedObj.getBoundingBox();
    }

    android.graphics.Rect getIdForFrame(int frameNum, android.graphics.Rect rect, int id) {
        List<DetectedObject> objects = listOfObjList.get(frameNum);
        for (DetectedObject obj : objects) {
            if (obj.getTrackingId() != null)
                if (obj.getTrackingId() == id)
                    return obj.getBoundingBox();
        }
        return rect;
    }
}
