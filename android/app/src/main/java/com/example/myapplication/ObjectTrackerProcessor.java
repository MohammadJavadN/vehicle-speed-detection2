package com.example.myapplication;

import android.content.Context;
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
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions;

import org.opencv.core.Rect;
import org.opencv.core.Mat;
//import org.opencv.tracking.Tracker;
//import org.opencv.tracking.TrackerKCF; // Import the desired tracking algorithm
import org.opencv.video.Tracker;

import java.util.List;

public class ObjectTrackerProcessor extends VisionProcessorBase<List<DetectedObject>> {

    private static final String TAG = "ObjectTrackerProcessor";

    private final ObjectDetector detector;

    public ObjectTrackerProcessor(Context context, ObjectDetectorOptionsBase options) {
        super(context);
        System.out.println("*** 34 of ObjectTrackerProcessor class");
        detector = ObjectDetection.getClient(options);
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
        for (DetectedObject object : results) {
            // TODO: 18.04.24 is need to call graphicOverlay.setImageSourceInfo(...)?
            graphicOverlay.add(new ObjectGraphic(graphicOverlay, object));
        }
        graphicOverlay.postInvalidate();
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Object detection failed!", e);
    }




//    private Tracker tracker;
//    private boolean isObjectDetected = false;
//    private Rect detectedObject;
//
//    // Initialize the tracker with the first frame and the detected object(s)
//    public void initializeTracker(Mat frame, Rect objects) {
//
//
//    }
//
//    // Perform object tracking on subsequent frames
//    public void trackObject(Mat frame) {
//        InputImage image = InputImage.fromBitmap(bitmap, rotationDegree);
//        objectDetector.process(image)
//                .addOnSuccessListener(
//                        new OnSuccessListener<List<DetectedObject>>() {
//                            @Override
//                            public void onSuccess(List<DetectedObject> detectedObjects) {
//                                // Task completed successfully
//                                // ...
//                            }
//                        })
//                .addOnFailureListener(
//                        new OnFailureListener() {
//                            @Override
//                            public void onFailure(@NonNull Exception e) {
//                                // Task failed with an exception
//                                // ...
//                            }
//                        });
////        if (isObjectDetected && tracker != null) {
////            // Update the tracker with the new frame
////            isObjectDetected = tracker.update(frame, detectedObject);
////            // Visualize the tracked object(s)
////                Imgproc.rectangle(frame, detectedObject.tl(), detectedObject.br(), new Scalar(0, 255, 0), 2); // Green bounding box
////        }
//    }
//
//    public ObjectTrackerProcessor() {
//        // Live detection and tracking
//        ObjectDetectorOptions options =
//                new ObjectDetectorOptions.Builder()
//                        .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
//                        .enableClassification()  // Optional
//                        .build();
//
//        ObjectDetector objectDetector = ObjectDetection.getClient(options);
////        // Multiple object detection in static images
////        ObjectDetectorOptions options =
////                new ObjectDetectorOptions.Builder()
////                        .setDetectorMode(ObjectDetectorOptions.SINGLE_IMAGE_MODE)
////                        .enableMultipleObjects()
////                        .enableClassification()  // Optional
////                        .build();
//
//    }
//
//    // Method to handle loss of tracking or occlusions
//    public void handleLossOfTracking() {
//        // Implement your logic here to handle loss of tracking or occlusions
//        // For example, you can try to re-detect the object or reset the tracking process
//    }
}
