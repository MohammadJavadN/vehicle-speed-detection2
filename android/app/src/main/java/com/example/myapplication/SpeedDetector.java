package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;

import com.google.mlkit.vision.objects.DetectedObject;

import org.opencv.core.Mat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public abstract class SpeedDetector {

    protected static final HashMap<Integer, HashMap<Integer, Float>> objectsSpeed = new HashMap<>();
    private static final float TEXT_SIZE = 20.0f;
    private static final float STROKE_WIDTH = 3.0f;
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
    protected final boolean showOpt = false;
    private final Paint rectPaint;
    private final Paint textPaint;
    private final Paint[] labelPaints;
    private final Paint[] boxPaints;
    private final Paint[] textPaints;
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

    public static HashMap<Integer, HashMap<Integer, Float>> getObjectsSpeed() {
        return objectsSpeed;
    }

    /**
     * Calculates Intersection over Union (IoU) between two Rect objects.
     *
     * @param rect1 First rectangle.
     * @param rect2 Second rectangle.
     * @return The IoU value.
     */
    public static float calculateIoU(Rect rect1, Rect rect2) {
        // Compute the intersection rectangle
        Rect intersection = new Rect();
        if (!Rect.intersects(rect1, rect2)) {
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

    protected int updateObjectsSpeed(int frameNum, int id, int speed) {
        if (speed != -1) {
            if (!objectsSpeed.containsKey(id)) {
                HashMap<Integer, Float> frameSpeed = new HashMap<>();
                frameSpeed.put(frameNum, (float) speed);
                objectsSpeed.put(id, frameSpeed);
            } else
                Objects.requireNonNull(objectsSpeed.get(id)).put(frameNum, (float) speed);

            if (!carSpeeds.containsKey(id)) {
                ArrayList<Float> speeds = new ArrayList<>();
                speeds.add((float) speed);
                carSpeeds.put(id, speeds);
            } else
                Objects.requireNonNull(carSpeeds.get(id)).add((float) speed);
        }
        if (!objectsSpeed.containsKey(id))
            return -1;
        if (objectsSpeed.get(id) == null)
            return -1;
        return (int) meanSpeed(Objects.requireNonNull(carSpeeds.get(id)));
    }

    final int N = 8;
    float[] speeds = new float[N];
    HashMap<Integer, ArrayList<Float>> carSpeeds = new HashMap<>();
    private float meanSpeed(ArrayList<Float> carSpeeds) {
        if (carSpeeds.isEmpty()) return 0; // handle empty map case

        float totalSum = 0;
        int count = 0;

        // Calculate the initial mean and gather counts for filtering
        for (Float speed : carSpeeds) {
            if (speed > 0) {
                totalSum += speed;
                count++;
                speeds[count % N] = speed;
            }
        }
        if (count == 0)
            return 0;
        if (count < N)
            return totalSum / count;

        totalSum = 0;
        for (int i = 0; i < N; i++) {
            totalSum += speeds[i];
        }
        return totalSum / N;

    }

    private float meanSpeed(HashMap<Integer, Float> frameSpeeds) {
        if (frameSpeeds.isEmpty()) return 0; // handle empty map case

        float totalSum = 0;
        int count = 0;

        // Calculate the initial mean and gather counts for filtering
        for (Float speed : frameSpeeds.values()) {
            if (speed > 0) {
                totalSum += speed;
                count++;
            }
        }
        if (count == 0)
            return 0;
        float mean1 = totalSum / count;

        float filteredSum = 0;
        int filteredCount = 0;

        // Compute filtered mean in a single pass
        for (Map.Entry<Integer, Float> entry : frameSpeeds.entrySet()) {
            Float speed = entry.getValue();
            if (speed > 0.7 * mean1 && speed < 1.3 * mean1) {
                filteredSum += speed;
                filteredCount++;
            }
        }

        // Handle case where no values fall into the filtered range
        if (filteredCount == 0) {
            return mean1; // or return another meaningful value if required
        }

        float mean2 = filteredSum / filteredCount;

        // Update the map values
        for (Map.Entry<Integer, Float> entry : frameSpeeds.entrySet()) {
            if (entry.getValue() > 0.7 * mean1 && entry.getValue() < 1.3 * mean1) {
                entry.setValue(mean2);
            }
        }

        return mean2;
    }

    public void draw(Canvas canvas, Rect rect, float speed, int id) {
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

    Rect setIdForFrame(int frameNum, Rect rect, int id) {
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

    Rect getIdForFrame(int frameNum, Rect rect, int id) {
        List<DetectedObject> objects = listOfObjList.get(frameNum);
        for (DetectedObject obj : objects) {
            if (obj.getTrackingId() != null)
                if (obj.getTrackingId() == id)
                    return obj.getBoundingBox();
        }
        return rect;
    }
}
