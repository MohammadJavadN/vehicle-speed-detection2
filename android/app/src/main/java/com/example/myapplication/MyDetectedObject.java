package com.example.myapplication;

import static java.lang.Math.log;
import static java.lang.Math.log10;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

import android.graphics.Rect;
import android.graphics.RectF;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.google.mlkit.vision.GraphicOverlay;
import com.google.mlkit.vision.objects.DetectedObject;

import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;

public class MyDetectedObject extends DetectedObject {

    private static final HashMap<Integer, HashMap<Integer, Float>> objectsSpeed = new HashMap<>();
    // size of the input image
    public static float imgHeight;
    public static float imgWidth;
    private static int nextId = 1;

    private final int SPEED_CNT = 15;
    private final Point[] speedVectors = new Point[SPEED_CNT];
    private final float[] speeds = new float[SPEED_CNT];
    protected RectF location;
    int id = -1;
    int frameNum, frameNumUpdated;
    float coef = 10000f;
    private Rect boundingBox;
    private int speedCnt = 0;
    private float speed;

    public MyDetectedObject(@NonNull Rect boundingBox, int frameNum) {
        super(boundingBox, nextId, new ArrayList<>());
//        this.boundingBox = boundingBox;
        setLocationInt(boundingBox);
        id = nextId;
        this.frameNum = frameNum;
        this.frameNumUpdated = frameNum;
        nextId++;
        objectsSpeed.put(id, new HashMap<>());
    }

    public MyDetectedObject(@NonNull Rect boundingBox, int frameNum, float speed) {
        super(boundingBox, nextId, new ArrayList<>());
//        this.boundingBox = boundingBox;
        setLocationInt(boundingBox);
        id = nextId;
        this.frameNum = frameNum;
        this.frameNumUpdated = frameNum;
        nextId++;
        objectsSpeed.put(id, new HashMap<>());
        this.speed = speed;
    }
    public static HashMap<Integer, HashMap<Integer, Float>> getObjectsSpeed() {
        return objectsSpeed;
    }

    @NonNull
    public Rect getBoundingBox() {
        return boundingBox;
    }

    public void setLocation(RectF location) {
        this.location = location;
        this.boundingBox = new Rect(Math.round(location.left * imgWidth),
                Math.round(location.top * imgHeight),
                Math.round(location.right * imgWidth),
                Math.round(location.bottom * imgHeight));
    }

    public float getSpeed() {
        return ((int) (speed));
    }

    public void updateBoxAndSpeed(Rect box, int frameNum) {
        RectF newLocation = new RectF(
                box.left / imgWidth,
                box.top / imgHeight,
                box.right / imgWidth,
                box.bottom / imgHeight
        );

        Point tmpSpeed = GraphicOverlay.getOverlayInstance().roadLine.calculateSignSpeed(
                location,
                newLocation,
                frameNum - this.frameNumUpdated
        );
        speedVectors[speedCnt % SPEED_CNT] = tmpSpeed;
        speedCnt++;
        float lastSpeed = speed;
        int cnt = min(speedCnt, SPEED_CNT);
        double speedX = 0, speedY = 0;

        for (int i = 0; i < cnt; i++) {
            Point v = speedVectors[i];
            speedX += v.x;
            speedY += v.y;
        }
        speedX /= cnt;
        speedY /= cnt;
        float speed1 = (float) sqrt(speedX * speedX + speedY * speedY) * 0.6f;

        double val1 = 90 - 40 * log10(900 / (speed1 - 10) - 12);
        double val2 = 50 * log10(speed1 / 2);
        speed = (float) min(max(0, val1), val2);
        if (speed < 10 || ((int) speed) < 20) {
            speedCnt--;
            speed = lastSpeed;
            return;
        }

        setLocation(newLocation);

        Objects.requireNonNull(objectsSpeed.get(id)).put(frameNum, speed);
        this.frameNumUpdated = frameNum;
    }


    public void updateBoxAndSpeed(Rect box, int frameNum, float speed) {
        RectF newLocation = new RectF(
                box.left / imgWidth,
                box.top / imgHeight,
                box.right / imgWidth,
                box.bottom / imgHeight
        );

        speeds[speedCnt % SPEED_CNT] = speed;
        speedCnt++;
        int cnt = min(speedCnt, SPEED_CNT);

        float sumSpeed = 0;
        for (int i = 0; i < cnt; i++) {
            sumSpeed = speeds[i];
        }
        sumSpeed /= cnt;

        this.speed = min(0, sumSpeed);
        if (this.speed < 10 || ((int) this.speed) < 20) {
            speedCnt--;
            this.speed = speed;
            return;
        }

        setLocation(newLocation);

        Objects.requireNonNull(objectsSpeed.get(id)).put(frameNum, this.speed);
        this.frameNumUpdated = frameNum;
    }

    private float distance(Point center, Point center1) {
        return (float) sqrt(
                pow(center.x - center1.x, 2) +
                        pow(center.y - center1.y, 2)
        );
    }

    public void setLocationInt(Rect locationInt) {
        this.boundingBox = locationInt;
        this.location = new RectF(locationInt.left / imgWidth,
                locationInt.top / imgHeight,
                locationInt.right / imgWidth,
                locationInt.bottom / imgHeight);
    }

    public Integer getTrackingId() {
        return this.id; // TODO: 22.04.24 int to Integer
    }

}