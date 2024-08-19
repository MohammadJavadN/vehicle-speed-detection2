package com.example.myapplication;

import android.content.Context;
import android.graphics.Rect;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.GraphicOverlay;
import com.google.mlkit.vision.VisionProcessorBase;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.ObjectDetection;
import com.google.mlkit.vision.objects.ObjectDetector;
import com.google.mlkit.vision.objects.ObjectDetectorOptionsBase;

//import org.opencv.tracking.Tracker;
//import org.opencv.tracking.TrackerKCF; // Import the desired tracking algorithm

import java.util.ArrayList;
import java.util.List;

public class ObjectTrackerProcessor extends VisionProcessorBase<List<DetectedObject>> {

    private static final String TAG = "ObjectTrackerProcessor";

    private final ObjectDetector detector;
    private final List<MyDetectedObject> prevObjects;
    double DISTANCE_TH = 1;
    private int frameNum;

    public ObjectTrackerProcessor(Context context, ObjectDetectorOptionsBase options, TOFSpeedDetector tof) {
        super(context);
        detector = ObjectDetection.getClient(options);
        frameNum = 0;
        prevObjects = new ArrayList<>();
        this.tofSpeedDetector = tof;
    }

    @Override
    public void stop() {
        super.stop();
        detector.close();
    }

    @Override
    protected Task<List<DetectedObject>> detectInImage(InputImage image) {
        return detector.process(image);
    }

    @Override
    protected void onSuccess(
            @NonNull List<DetectedObject> results, @NonNull GraphicOverlay graphicOverlay) {
        frameNum++;
//        setIDAndSpeed(results);
//        removeOutObj();
        for (DetectedObject object : results) {
            graphicOverlay.add(new ObjectGraphic(graphicOverlay, object));
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Object detection failed!", e);
    }

    private void removeOutObj() {
        prevObjects.removeIf(prevObj -> prevObj.frameNum < frameNum);

    }

    private void setIDAndSpeed(List<DetectedObject> currObjects) {
        for (DetectedObject currObj : currObjects) {

            Rect currRect = currObj.getBoundingBox();
            int id = -1;
            double minD = Double.MAX_VALUE;

            for (MyDetectedObject prevObj : prevObjects) {
                if (prevObj.frameNum < frameNum) {
                    double d = distance(currRect, prevObj.getBoundingBox());
                    if (d < minD) {
                        minD = d;
                        id = prevObj.id;
                    }
                }
            }
            if (minD > DISTANCE_TH)
                addNewObj(currObj);
            else
                updateObj(id, currObj);
        }
    }

    private double distance(Rect rect1, Rect rect2) {
        double d = Math.sqrt(
                Math.pow(rect1.centerX() - rect2.centerX(), 2) +
                        Math.pow(rect1.centerY() - rect2.centerY(), 2)
        );// * Math.max(rect1.height()/rect2.height(), rect2.height()/rect1.height()); todo

        return d;
    }

    private void updateObj(int id, DetectedObject object) {
        for (MyDetectedObject prevObj : prevObjects) {
            if (prevObj.id == id) {
//                prevObj.updateBoxAndSpeed(object.getBoundingBox(), frameNum);

                List<DetectedObject.Label> labels = object.getLabels();
                prevObj.updateBoxAndSpeed(
                        object.getBoundingBox(),
                        frameNum,
                        Float.parseFloat(labels.get(labels.size()-1).getText())
                );
                prevObj.frameNum = frameNum;
                break;
            }
        }
    }

    private void addNewObj(DetectedObject object) {
        List<DetectedObject.Label> labels = object.getLabels();

        if (labels.isEmpty())
            prevObjects.add(
                    new MyDetectedObject(
                            object.getBoundingBox(),
                            frameNum
                    )
            );
        else
            prevObjects.add(
                    new MyDetectedObject(
                            object.getBoundingBox(),
                            frameNum,
                            Float.parseFloat(labels.get(labels.size()-1).getText())
                    )
            );
    }
}