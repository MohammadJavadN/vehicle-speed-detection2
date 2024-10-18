package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

import com.example.myapplication.ml.SpeedPredictionModel;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions;

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
import java.util.HashMap;
import java.util.List;
import java.util.Random;

public class TOFSpeedDetector extends SpeedDetector {

    final static int fq = 38;
    final static int ff = 0;
    final static int ptsNum = 50;
    private static final Size WIN_SIZE = new Size(100, 40);
    private static final TermCriteria CRITERIA = new TermCriteria(TermCriteria.COUNT | TermCriteria.EPS,
            10,
            0.003);
    private static final int MAX_LEVEL = 10;
    private static final int MAX_CORNERS = 20;
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

    private final TensorBuffer speedInputFeature;
    private final List<Bitmap> bitmaps;
    private final List<List<STrack>> listOfSTrackList;
    private final SpeedPredictionModel speedPredictionModel;
    public Bitmap dest;
    private int frameNum = 0;

    public TOFSpeedDetector(TensorBuffer speedInputFeature, SpeedPredictionModel speedPredictionModel) {
        super(fq);
        this.speedInputFeature = speedInputFeature;
        this.speedPredictionModel = speedPredictionModel;

        bitmaps = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            bitmaps.add(null);
        }

        listOfSTrackList = new ArrayList<>(fq);
        for (int i = 0; i < fq; i++) {
            listOfSTrackList.add(new ArrayList<>());
        }
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

//            Rect bBox = new Rect(object.getBoundingBox().left, object.getBoundingBox().top,
//                    object.getBoundingBox().right, object.getBoundingBox().bottom);

            List<Double> distances = new ArrayList<>();
            List<Double> data = new ArrayList<>();
            data.add((double) rectI.width() / frame.width());
            data.add((double) rectI.height() / frame.height());
            final int STEP = 6;

            List<Point> prevPts = new ArrayList<>();
            for (double yC : yCoeffs) {
                for (double xC : xCoeffs) {
                    double x = rectI.left + xC * (rectI.width());
                    double y = rectI.top + yC * (rectI.height());
                    prevPts.add(new Point(x, y));
                }
            }

            Mat prevGray = new Mat();
            Imgproc.cvtColor(frames.get((frameNum - fq) % fq), prevGray, Imgproc.COLOR_BGR2GRAY);

            MatOfPoint2f prevPtsMat = new MatOfPoint2f();
            prevPtsMat.fromList(prevPts);

            MatOfByte status = new MatOfByte();
            MatOfFloat errors = new MatOfFloat();


            ArrayList<android.graphics.Rect> rectJs = new ArrayList<>();
            // Calculate optical flow
            for (int i = ff + 1; i < fq - 1; i += STEP) { // TODO: 19.08.24
                Mat frameGray = new Mat();
                Imgproc.cvtColor(frames.get((frameNum - fq + i) % fq), frameGray, Imgproc.COLOR_BGR2GRAY);

                MatOfPoint2f nextPts = new MatOfPoint2f();
                Video.calcOpticalFlowPyrLK(
                        prevGray, frameGray, prevPtsMat, nextPts, status, errors, WIN_SIZE, MAX_LEVEL, CRITERIA);

                List<Point> nextPtsList = nextPts.toList();
                double dfx = nextPtsList.get(0).x - prevPts.get(0).x;
                double dfy = nextPtsList.get(0).y - prevPts.get(0).y;
                double fw1 = prevPts.get(34).x - prevPts.get(0).x;
                double fw2 = nextPtsList.get(34).x - nextPtsList.get(0).x;
                double fh1 = prevPts.get(34).y - prevPts.get(0).y;
                double fh2 = nextPtsList.get(34).y - nextPtsList.get(0).y;

                double dfw = rectI.left > 20 ? fw2 / fw1 : 1;
                double dfh = rectI.top > 20 ? fh2 / fh1 : 1;

                // Calculate Euclidean distances
                for (int j = 0; j < prevPts.size(); j++) {
                    Point prevPt = prevPts.get(j);
                    Point nextPt = nextPtsList.get(j);
                    double distance = Math.sqrt(
                            Math.pow(nextPt.x - prevPt.x - dfx, 2) +
                                    Math.pow(nextPt.y - prevPt.y - dfy, 2)
                    ) / rectI.width();
                    distances.add(distance); // TODO: 19.08.24
                }

                android.graphics.Rect rectJ;
                if (MainActivity.DETECTION_MODE == ObjectDetectorOptions.SINGLE_IMAGE_MODE)
                    rectJ = setIdForFrame(
                            (frameNum - fq + i) % fq,
                            new android.graphics.Rect(
                                    (int) (rectI.left + dfx),
                                    (int) (rectI.top + dfy),
                                    rectI.left + rectI.width(),
                                    rectI.top + rectI.height()
                            ),
                            id
                    );
                else
                    rectJ = getIdForFrame(
                            (frameNum - fq + i) % fq,
                            new android.graphics.Rect(
                                    (int) (rectI.left + dfx),
                                    (int) (rectI.top + dfy),
                                    rectI.left + rectI.width(),
                                    rectI.top + rectI.height()
                            ),
                            id
                    );
                if (rectJ == null)
                    break;
//                    if (rectJs.isEmpty())
//                        break;
//                    else
//                        rectJs.add(rectJs.get(rectJs.size()-1));
                else
                    rectJs.add(rectJ);

            }
            android.graphics.Rect scaledBBox = new android.graphics.Rect(
                    (int) (rectI.left * sx),
                    (int) (rectI.top * sy),
                    (int) (rectI.right * sx),
                    (int) (rectI.bottom * sy)
            );

            if (rectJs.size() < ((fq - ff) / STEP)) { // TODO: 19.08.24
                float speed = updateObjectsSpeed(frameNum, id, -1);
                if (speed > 0)
                    draw(canvas, scaledBBox, speed, id);
                continue;
            }

//            for (android.graphics.Rect rectJ : rectJs) {
//                data.add(((double) (rectJ.width() - rectI.width()) / rectI.width()));
//                data.add(((double) (rectJ.height() - rectI.height()) / rectI.height()));
//            }

            data.addAll(distances);

            double[] input_data = listToArray(data);

            // Get output tensor (predicted_speed_function)
            float predictedSpeed = predictSpeed(input_data);

            int speed = (int) predictedSpeed;
            // Draw label text

            speed = updateObjectsSpeed(frameNum, id, speed);
            List<DetectedObject.Label> labels = object.getLabels();
            if (!labels.isEmpty()) {
                if (Float.parseFloat(labels.get(0).getText()) != -1)
                    speed = (int) Float.parseFloat(labels.get(0).getText());
            }

            draw(canvas, scaledBBox, speed, id);

        }

    }

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

    public List<DetectedObject> detectSpeeds2(
            Bitmap originalCameraImage,
            List<DetectedObject> detectedObjects
    ) {

        Mat frame = new Mat();
        Utils.bitmapToMat(originalCameraImage, frame);

        frames.set(frameNum % fq, frame.clone());
        bitmaps.set(frameNum % fq, originalCameraImage.copy(originalCameraImage.getConfig(), true));
        listOfObjList.set(frameNum % fq, new ArrayList<>(detectedObjects));
        frameNum++;
        System.out.println("*** detectedObjects.size() = " + detectedObjects.size());

        if (frameNum < fq) {
            dest = originalCameraImage;
            System.out.println("*** frameNum < fq : " + frameNum + " < " + fq);
            return new ArrayList<>();
        }
        dest = bitmaps.get((frameNum - fq) % fq);
        List<DetectedObject> detectedObjectsOut = new ArrayList<>(detectedObjects.size());

        for (DetectedObject object : listOfObjList.get((frameNum - fq) % fq)) {
            int id = getId(object);

            android.graphics.Rect rectI = object.getBoundingBox();
            System.out.println("rectI = " + rectI);
//            Rect rectI = new Rect(object.getBoundingBox().left, object.getBoundingBox().top,
//                    object.getBoundingBox().right, object.getBoundingBox().bottom);

            List<Double> distances = new ArrayList<>();
            List<Double> data = new ArrayList<>();
            data.add((double) rectI.width() / frame.width());
            data.add((double) rectI.height() / frame.height());

            final int STEP = 7;
//            for (int j = STEP; j < fq; j+= STEP) {
//                Rect rectJ = null;
//                for (STrack sTrackJ : listOfSTrackList.get((frameNum - fq + j) % fq)) {
//                    if (sTrackJ.trackId == id) {
//                        rectJ = sTrackJ.getRect();
//                    }
//                }
//                if (rectJ == null)
//                    break;
//
//                distances.add((double) ((rectJ.width - rectI.width) / rectI.width));
//                distances.add((double) ((rectJ.height - rectI.height) / rectI.height));
//            }
//
//            if (distances.size() < fq / STEP + 1)
//                continue;

            List<Point> prevPts = new ArrayList<>();
            for (double yC : yCoeffs) {
                for (double xC : xCoeffs) {
                    double x = rectI.left + xC * (rectI.width());
                    double y = rectI.top + yC * (rectI.height());
                    prevPts.add(new Point(x, y));
                }
            }

            Mat prevGray = new Mat();
            Imgproc.cvtColor(frames.get((frameNum - fq) % fq), prevGray, Imgproc.COLOR_BGR2GRAY);

            MatOfPoint2f prevPtsMat = new MatOfPoint2f();
            prevPtsMat.fromList(prevPts);

            MatOfByte status = new MatOfByte();
            MatOfFloat errors = new MatOfFloat();

            ArrayList<android.graphics.Rect> rectJs = new ArrayList<>();
            // Calculate optical flow
            for (int i = ff + 1; i < fq - 1; i += STEP) { // TODO: 19.08.24
                Mat frameGray = new Mat();
                Imgproc.cvtColor(frames.get((frameNum - fq + i) % fq), frameGray, Imgproc.COLOR_BGR2GRAY);

                MatOfPoint2f nextPts = new MatOfPoint2f();
                Video.calcOpticalFlowPyrLK(
                        prevGray, frameGray, prevPtsMat, nextPts, status, errors, WIN_SIZE, MAX_LEVEL, CRITERIA);

                List<Point> nextPtsList = nextPts.toList();
                double dfx = nextPtsList.get(0).x - prevPts.get(0).x;
                double dfy = nextPtsList.get(0).y - prevPts.get(0).y;
                double fw1 = prevPts.get(34).x - prevPts.get(0).x;
                double fw2 = nextPtsList.get(34).x - nextPtsList.get(0).x;
                double fh1 = prevPts.get(34).y - prevPts.get(0).y;
                double fh2 = nextPtsList.get(34).y - nextPtsList.get(0).y;

                double dfw = rectI.left > 20 ? fw2 / fw1 : 1;
                double dfh = rectI.top > 20 ? fh2 / fh1 : 1;

                // Calculate Euclidean distances
                for (int j = 0; j < prevPts.size(); j++) {
                    Point prevPt = prevPts.get(j);
                    Point nextPt = nextPtsList.get(j);
                    double distance = Math.sqrt(
                            Math.pow(nextPt.x - prevPt.x - dfx, 2) +
                                    Math.pow(nextPt.y - prevPt.y - dfy, 2)
                    ) / frame.width();
                    distances.add(distance); // TODO: 19.08.24
                }
//                android.graphics.Rect rectJ = setIdForFrame(
//                    (frameNum - fq + i) % fq,
//                    new android.graphics.Rect(
//                        (int) (rectI.left + dfx),
//                        (int) (rectI.top + dfy),
//                        (int) (rectI.left + rectI.width()),
//                        (int) (rectI.top + rectI.height())
//                    ),
//                    id
//                );

                android.graphics.Rect rectJ = getIdForFrame(
                        (frameNum - fq + i) % fq,
                        new android.graphics.Rect(
                                (int) (rectI.left + dfx),
                                (int) (rectI.top + dfy),
                                rectI.left + rectI.width(),
                                rectI.top + rectI.height()
                        ),
                        id
                );
                if (rectJ == null)
                    if (rectJs.isEmpty())
                        break;
                    else
                        rectJs.add(rectJs.get(rectJs.size() - 1));
                else
                    rectJs.add(rectJ);

            }

            System.out.println("rectJs.size() < (fq - ff) / STEP) : " + rectJs.size() + " < " + ((fq - ff) / STEP));
            if (rectJs.size() < ((fq - ff) / STEP)) // TODO: 19.08.24
                continue;

            for (android.graphics.Rect rectJ : rectJs) {
                data.add(((double) (rectJ.width() - rectI.width()) / rectI.width()));
                data.add(((double) (rectJ.height() - rectI.height()) / rectI.height()));
            }

            data.addAll(distances);

            System.out.println("data.size()" + data.size());
            double[] input_data = listToArray(data);
            System.out.println(Arrays.toString(input_data));

            // Get output tensor (predicted_speed_function)
            float predictedSpeed = predictSpeed(input_data);

            System.out.println("predictedSpeed = " + predictedSpeed);

            int speed = (int) predictedSpeed;

            List<DetectedObject.Label> labels = object.getLabels();
            labels.add(new DetectedObject.Label(Float.toString(speed), -1, -1));

            detectedObjectsOut.add(
                    new DetectedObject(
                            object.getBoundingBox(),
                            id,
                            labels
                    )
            );

        }
        return detectedObjectsOut;
    }

//    HashMap<Integer, Float>

    private float predictSpeed(double[] inputData) {
        //>>> from joblib import load
        //>>> sc = load('speed_prediction_model/std_scaler.bin')
        //>>> formatted_means = ", ".join("{:.6f}".format(value) for value in sc.mean_)
        //>>> formatted_scales = ", ".join("{:.6f}".format(value) for value in sc.scale_)
        //>>> formatted_means
        double[] MEANS = new double[]{
//                0.456081, 0.512430, 0.000000, 0.001280, 0.001775, 0.002110, 0.002128, 0.002154,
//                0.002289, 0.001145, 0.001586, 0.002152, 0.002330, 0.002281, 0.002172, 0.002365,
//                0.001661, 0.002140, 0.002709, 0.002808, 0.002699, 0.002421, 0.002477, 0.001906,
//                0.002409, 0.002876, 0.002961, 0.002881, 0.002593, 0.002542, 0.002032, 0.002372,
//                0.002865, 0.002951, 0.002842, 0.002540, 0.002559, 0.000000, 0.004831, 0.006962,
//                0.009130, 0.009897, 0.010999, 0.012427, 0.005413, 0.006507, 0.008128, 0.009743,
//                0.010661, 0.011985, 0.013528, 0.008360, 0.009183, 0.009693, 0.010565, 0.011656,
//                0.012918, 0.014559, 0.010133, 0.010983, 0.010649, 0.011386, 0.012463, 0.013381,
//                0.015273, 0.011199, 0.011717, 0.011569, 0.012257, 0.013436, 0.014180, 0.015799,
//                0.000000, 0.006509, 0.009475, 0.012713, 0.014314, 0.016332, 0.018402, 0.008166,
//                0.009437, 0.011266, 0.013621, 0.015353, 0.017982, 0.020437, 0.013360, 0.013338,
//                0.013826, 0.015343, 0.017591, 0.020506, 0.023018, 0.016758, 0.016522, 0.015884,
//                0.017164, 0.019536, 0.021655, 0.024618, 0.019400, 0.018642, 0.018227, 0.019327,
//                0.021875, 0.023686, 0.026288, 0.000000, 0.007712, 0.011329, 0.015259, 0.017532,
//                0.020189, 0.022866, 0.010168, 0.011620, 0.013824, 0.016520, 0.018674, 0.022063,
//                0.024964, 0.016728, 0.016161, 0.016597, 0.018417, 0.021080, 0.025403, 0.028602,
//                0.021745, 0.020280, 0.019594, 0.020879, 0.023694, 0.027498, 0.031093, 0.025739,
//                0.023464, 0.022965, 0.024081, 0.027379, 0.031200, 0.034378, 0.000000, 0.008837,
//                0.012922, 0.017242, 0.020179, 0.023301, 0.026356, 0.011716, 0.013353, 0.015855,
//                0.018721, 0.021483, 0.025468, 0.028473, 0.019593, 0.018288, 0.018857, 0.020699,
//                0.023493, 0.028818, 0.032535, 0.025794, 0.023029, 0.022540, 0.023814, 0.026820,
//                0.032076, 0.036002, 0.031149, 0.027455, 0.026947, 0.028153, 0.031388, 0.037214,
//                0.041020
//                0.45662053508204203, 0.5129855759244284, 0.004200685068271734, 0.003961696657187035, 0.0335313675272257, 0.03164933970251201, 0.06443568893006058, 0.06045059329428415, 0.0971392702564873, 0.0901858940122303, 0.13197696550992813, 0.12064982478425307, 0.0, 0.0013020294296622993, 0.0018045939728224133, 0.0021421318086614774, 0.0021607093188142057, 0.0021912130239958084, 0.0023289419423859056, 0.0011736259899018389, 0.0016175310666213855, 0.0021839617081623477, 0.0023613421870106294, 0.002311558919305588, 0.0022096422192818916, 0.0024098729982831206, 0.0017013414720533906, 0.0021777402959504213, 0.0027417289680872134, 0.0028379527460356433, 0.002732289280365324, 0.002458287428281514, 0.002518092857719602, 0.0019555192608092195, 0.0024490437830639064, 0.002920252984589667, 0.003006717870548961, 0.00292974523070884, 0.0026453255460383014, 0.00258638246455034, 0.0020842035853711896, 0.00242087443494836, 0.0029136118252959713, 0.003001164274635411, 0.0028928831984393892, 0.002593167858322405, 0.0026028701601414684, 6.077903071328397e-13, 0.004832568674054296, 0.006952341103881723, 0.009096528782073982, 0.009854262017015643, 0.010952440136525335, 0.012364276763500002, 0.005430424938898395, 0.006506747799158677, 0.008110639673531981, 0.00970036006781107, 0.0106050793518982, 0.011918949452872998, 0.013463622808913327, 0.00836572335146962, 0.009181032906229186, 0.009681169534738897, 0.010553780904291417, 0.011624690148841984, 0.012877287212502172, 0.014495808678302587, 0.010132460245037358, 0.010984764683356678, 0.010663039420385116, 0.011389404345432962, 0.012447504824788007, 0.013349507434084511, 0.015210177729068288, 0.011223588574163085, 0.011725285424699666, 0.011586693633079296, 0.012256276529228918, 0.013430457253136388, 0.014152751219038785, 0.01574597987053087, 1.0129838452213994e-12, 0.0065190799274920915, 0.009490391535916548, 0.012700808150254903, 0.014267631789214951, 0.01626099510414737, 0.01833941821351288, 0.008155420994300255, 0.009411364143301333, 0.01125899172604204, 0.013591399196389627, 0.01529784201011547, 0.01789088335630208, 0.02032568957004149, 0.013279777264442813, 0.013289119382193883, 0.013753514689083528, 0.015270446157220325, 0.017469548321146278, 0.020358751855949084, 0.022863877707286474, 0.016676279811785825, 0.016408899472047107, 0.015806147894223008, 0.017067595054258606, 0.01937234804699401, 0.02149761391928749, 0.02443613968267448, 0.019295965648025305, 0.01851238468777338, 0.018101640734390887, 0.019193618298982352, 0.021713561552729844, 0.023545724242385636, 0.02608111772167629, 1.7220725368763793e-12, 0.007733916876647489, 0.011347133096614422, 0.015226332641139987, 0.017474951613963938, 0.02011394416607628, 0.022765092135377012, 0.010174324243884536, 0.011599121570863033, 0.013788283894140023, 0.016472553046388325, 0.018648155163674546, 0.022001905665796302, 0.02487501057191728, 0.016707023498225462, 0.01614612340924021, 0.016557518044606787, 0.0183570676009755, 0.02098293360643489, 0.025248636891721108, 0.028435008583151202, 0.021651409638552797, 0.02016534026418425, 0.019509370734327444, 0.020800642297204002, 0.023582070110738703, 0.02731376266909483, 0.03088319854003172, 0.025578409996073796, 0.02333185655474018, 0.022832902614320305, 0.0239869742737501, 0.027202516427944697, 0.030950701374740774, 0.03414013837224901, 1.4519435114840061e-12, 0.008875638735770625, 0.012971169837223535, 0.01727231885134336, 0.02013326764622392, 0.023207426227512436, 0.02627424701800623, 0.0117384013764935, 0.013348626213263168, 0.015850519020390204, 0.01870958126638498, 0.021427091901780975, 0.025384959243800662, 0.028404968672757606, 0.019585035050286557, 0.018330186305392095, 0.01886508743812413, 0.020698703336535142, 0.0234836681328965, 0.028767585451171324, 0.03245264788243024, 0.025738764055831684, 0.023016907798220104, 0.02249344252411994, 0.023782405132563033, 0.026748390254230894, 0.03191816442623523, 0.03583697300356369, 0.031015475476590823, 0.027368616851925792, 0.026842050776194174, 0.028037975913404616, 0.031278667709476425, 0.03693505828977387, 0.04077710409363207
                0.358371, 0.417553, 0.000000, 0.002177, 0.004731, 0.005495, 0.005876, 0.005984, 0.005418, 0.001689, 0.003380, 0.006208, 0.006867, 0.007072, 0.006923, 0.005697, 0.002940, 0.005038, 0.007450, 0.007937, 0.008074, 0.007958, 0.006464, 0.003788, 0.005850, 0.007853, 0.008232, 0.008332, 0.008212, 0.006672, 0.003937, 0.005485, 0.007508, 0.007856, 0.007923, 0.007630, 0.006341, 0.000000, 0.005778, 0.012187, 0.016531, 0.020209, 0.022783, 0.025313, 0.004667, 0.008346, 0.013305, 0.016529, 0.019625, 0.022180, 0.024629, 0.008186, 0.012116, 0.014938, 0.017213, 0.020281, 0.022677, 0.024890, 0.011339, 0.015285, 0.016394, 0.018005, 0.021229, 0.023507, 0.025681, 0.012583, 0.015251, 0.017492, 0.019207, 0.022981, 0.024697, 0.026600, 0.000000, 0.010069, 0.018316, 0.025304, 0.032667, 0.037405, 0.043397, 0.008474, 0.013198, 0.019093, 0.024650, 0.031460, 0.036194, 0.041935, 0.013951, 0.018139, 0.021388, 0.025667, 0.033054, 0.038106, 0.043011, 0.019134, 0.022687, 0.023934, 0.027164, 0.035159, 0.040205, 0.045337, 0.022075, 0.024297, 0.026538, 0.030194, 0.039161, 0.043317, 0.047483, 0.000000, 0.013537, 0.023985, 0.033397, 0.043603, 0.050062, 0.059325, 0.011304, 0.016905, 0.024585, 0.032188, 0.041738, 0.048335, 0.057829, 0.018790, 0.023002, 0.027764, 0.033634, 0.043680, 0.051095, 0.059954, 0.025814, 0.028788, 0.031241, 0.036112, 0.047317, 0.054892, 0.063959, 0.030911, 0.032208, 0.035110, 0.040594, 0.053416, 0.059961, 0.067698, 0.000000, 0.016740, 0.029347, 0.040036, 0.052631, 0.061185, 0.073960, 0.014210, 0.020361, 0.029797, 0.038689, 0.050046, 0.058519, 0.072791, 0.023973, 0.027944, 0.034348, 0.041012, 0.052674, 0.062032, 0.075875, 0.032971, 0.034831, 0.038645, 0.044738, 0.057881, 0.067575, 0.080892, 0.040114, 0.039993, 0.043642, 0.050551, 0.065815, 0.074493, 0.086504, 0.000000, 0.019648, 0.034215, 0.045979, 0.060069, 0.070591, 0.086946, 0.016844, 0.023791, 0.034799, 0.044532, 0.056866, 0.067032, 0.086145, 0.029277, 0.032635, 0.040515, 0.047720, 0.060422, 0.071453, 0.090276, 0.040433, 0.040861, 0.046090, 0.052823, 0.066867, 0.078121, 0.095667, 0.049635, 0.047823, 0.052316, 0.059812, 0.076253, 0.086788, 0.103488
        };

        double[] SCALES = new double[]{
//                0.07150405645756418, 0.07852032136857703, 1.0, 0.0026713866001129996, 0.002765407592945686,
//                0.003294773189560498, 0.0032376951752429717, 0.0032004108277218194, 0.003114110908490867,
//                0.0029644971206852135, 0.002827693461239547, 0.0028349287643886895, 0.002881193114044747,
//                0.0029904933332003196, 0.0030379228992480026, 0.0031920397096160515, 0.0033234837849475795,
//                0.0031695947388117327, 0.0030652185084103197, 0.0030990288458292207, 0.003164303323922235,
//                0.0031859738785023494, 0.003067292163498995, 0.003526500105748415, 0.003448200186994151,
//                0.003334120743230051, 0.003431681327136023, 0.003422764171210631, 0.0034274434518230972,
//                0.0031600126260905026, 0.0036875315267819535, 0.0035111887801100665, 0.003573948775763457,
//                0.0036931158522223733, 0.0035753676529249913, 0.003392968762577973, 0.003099066382218837,
//                6.658966347107841e-11, 0.007333444469394485, 0.00856770846990472, 0.009477656697112365,
//                0.00948005589460316, 0.009713636942557547, 0.009421539055024747, 0.009507552600914402,
//                0.008845251953262473, 0.009292345558457456, 0.009548611892839882, 0.009668385955297778,
//                0.00976352734924456, 0.010265794596223591, 0.014036310253529844, 0.01247781063417716,
//                0.01037435276415391, 0.009554284758522845, 0.009678577927275818, 0.010469884996673234,
//                0.01069106448929563, 0.016619999876183088, 0.015268550029943337, 0.010437403183552602,
//                0.010128584002482833, 0.010280514044774187, 0.010877576950200097, 0.010671098202534786,
//                0.01508832030198106, 0.01341371605743066, 0.012085810436291729, 0.011212574329505443,
//                0.011274412702278901, 0.0117456593766726, 0.011324533707986055, 1.0590484800148546e-10,
//                0.008581627597171916, 0.010407664940073537, 0.01143539628110913, 0.015333648477849864,
//                0.016964626628018342, 0.013020348006539104, 0.012533212561855828, 0.011556749164224495,
//                0.01131357864975695, 0.011752033850222575, 0.01156753727346471, 0.01241142029002406,
//                0.013226310578718299, 0.019236329297506318, 0.01559599651894563, 0.012947403326056891,
//                0.012045978506924682, 0.012274260375717006, 0.013711638595570345, 0.014653168101044917,
//                0.02295477833645227, 0.01939568034134556, 0.013014972420815398, 0.012750578702590616,
//                0.013318903787769345, 0.01502494016934932, 0.015601206025227094, 0.02221446943260526,
//                0.017319959997896497, 0.015253169662465973, 0.014376948949929453, 0.015094943317163009,
//                0.017184535563568023, 0.017938281520989537, 1.1617832190691374e-10, 0.00874982518140243,
//                0.011716887832473611, 0.01293245249585382, 0.016835418116369568, 0.02044099237538998,
//                0.016261037633302996, 0.015437129904143816, 0.01362587260253229, 0.012962281470537051,
//                0.013387344026042617, 0.013562012142538316, 0.014661142893782451, 0.016010501242881135,
//                0.020801298672886003, 0.01765412496762067, 0.014333796686344982, 0.013559501368465213,
//                0.014137342807996224, 0.016008594341412472, 0.017426328507861886, 0.02526122593642604,
//                0.020901095770481583, 0.014439005226721853, 0.014154218488937201, 0.015553434392470498,
//                0.01930006211527514, 0.018968599859510625, 0.024092016181345186, 0.01860289868453954,
//                0.016987809039781577, 0.01614253382597919, 0.017542897038955913, 0.023140184915533442,
//                0.02271692315891307, 1.049749964865616e-10, 0.010033685100180958, 0.013227938701790615,
//                0.014495668537708312, 0.018467125183361004, 0.020888167396307302, 0.01873556145623834,
//                0.017139358390052227, 0.015306166435359437, 0.014572050416006765, 0.014834698206058157,
//                0.01552028528339428, 0.017175943217602473, 0.018748669277902443, 0.02198208663297282,
//                0.01811759177940468, 0.015483028164837494, 0.015010696259624399, 0.0161118002020852,
//                0.018709913101597538, 0.020963951601793842, 0.02565323381162335, 0.020475777091855638,
//                0.015735110849451634, 0.015384612146975292, 0.017895909815255663, 0.023271939874909137,
//                0.023084493824492916, 0.025591328184605747, 0.01990680769060079, 0.018537579577560467,
//                0.01816978929751124, 0.019633813448652497, 0.028479248370817205, 0.026434781142076737
//                0.07207906528688167, 0.07908329505738777, 0.0017261915238996482, 0.0020089153907468763, 0.0108910292555509, 0.012612049484630199, 0.02007458181539711, 0.022584087025828162, 0.030266732931364136, 0.03326487747820175, 0.04179357358431981, 0.04483372137906949, 1.0, 0.0027930765351512894, 0.002955092836739496, 0.0034812946430783126, 0.003448326624173479, 0.0034806013402656605, 0.0033835298182538632, 0.0031729911159218073, 0.0030628394099541183, 0.003063760155279834, 0.003090921833697997, 0.0031892951834709958, 0.0033053062989091557, 0.003540967786294727, 0.0036110429313378997, 0.0034593199390929876, 0.003300500655172291, 0.0033174730954216814, 0.0033950461273508075, 0.0034578655510560485, 0.0033464849931689076, 0.0038638369839162343, 0.00372815728934514, 0.003648537253281922, 0.0037708113522724445, 0.003774333280365884, 0.0037628179496597427, 0.003444510149257968, 0.004020081236054476, 0.003833591913506697, 0.003912623234044951, 0.004066550103509305, 0.003935581473268437, 0.0037535933270157946, 0.003383442736078856, 8.271959892803924e-11, 0.007838846102017927, 0.009111820515567921, 0.009930846184974198, 0.009902301217914529, 0.010270035548213053, 0.009755764876366183, 0.010122728823432906, 0.009326881242806059, 0.009696166352497473, 0.00983089050554815, 0.009909306867096785, 0.010111375791190936, 0.010571310376383154, 0.014340842025195673, 0.012909266353417296, 0.010600784468952877, 0.009913024231035857, 0.010064206076131278, 0.010970541882340545, 0.011072672374631616, 0.016801816795483072, 0.015327327466144795, 0.010853441825708096, 0.010555008389644153, 0.010779324202897253, 0.011335990883067583, 0.011193244025094794, 0.01588770066799033, 0.01400214465159892, 0.012541247500880344, 0.011683398328959628, 0.011812882554594926, 0.01223918228965722, 0.01195194664640773, 1.315579313873174e-10, 0.009182354778313485, 0.011043529226310245, 0.012052839298858434, 0.015393218487723237, 0.01721137457730701, 0.013540849660849317, 0.013069353998319549, 0.011993500285764097, 0.01185579080190061, 0.01228526210833766, 0.012141187824635536, 0.013054182985488987, 0.013701697566813192, 0.019347893697022747, 0.016101002416933692, 0.013284993076565001, 0.01252067033861497, 0.012758159929063893, 0.014320129815221074, 0.015159854242557077, 0.023156172998846265, 0.01947376631078621, 0.01351923816902607, 0.01323306697178499, 0.013802491308972318, 0.01549215066093982, 0.01602789747662768, 0.022647595543826383, 0.017832658020165884, 0.015726972107734966, 0.014898339751455726, 0.015625999696059906, 0.017866564305204405, 0.018295898390558694, 1.4431781437133266e-10, 0.009321209434898267, 0.012317758103202124, 0.013452890975191985, 0.017106747749428435, 0.02049391711666733, 0.01657296820967672, 0.01588878098731748, 0.01406308982419555, 0.01338878757949388, 0.013826891447574558, 0.014005018071524593, 0.015150163052064331, 0.016398257141238545, 0.02127270304925345, 0.01821451378571497, 0.0146852198874902, 0.013977618451629132, 0.014536558065487254, 0.016542035990854995, 0.01799127536183581, 0.02556890548173298, 0.02096361732093082, 0.014833543840203953, 0.014513718804024588, 0.015886274586283156, 0.019497724409678428, 0.019375695399182862, 0.024622388653916846, 0.01907593074361489, 0.017329854471742186, 0.016534476775947007, 0.01788226646346084, 0.023268305012263405, 0.022916832190128737, 1.3040135964018837e-10, 0.010751960050971126, 0.01401703168997774, 0.015241821718072236, 0.019069622400586735, 0.021530659867532567, 0.019252265300807946, 0.01764338798894498, 0.015905906460410905, 0.015345379846458352, 0.015523259880601808, 0.016142813903401894, 0.017753339383530575, 0.019224781924069453, 0.022597310538791115, 0.019053144946680298, 0.0161313192056525, 0.01569114024532506, 0.01675760758522294, 0.019350520942565803, 0.02156458588867818, 0.026321305314703152, 0.02119260698480223, 0.016359177648395222, 0.016048564511552156, 0.01835923247398378, 0.023383693888019016, 0.02348937973795243, 0.02634530654439362, 0.020660749949036838, 0.01921330823703201, 0.01879667753674796, 0.020365500957826937, 0.028635936955473805, 0.027350060700624863
                0.056256, 0.068182, 1.000000, 0.003349, 0.004535, 0.004759, 0.004906, 0.004970, 0.004499, 0.002980, 0.004416, 0.005192, 0.005242, 0.005318, 0.005422, 0.004452, 0.003456, 0.004863, 0.005348, 0.005446, 0.005522, 0.005545, 0.004755, 0.003819, 0.005116, 0.005436, 0.005454, 0.005505, 0.005493, 0.004748, 0.003504, 0.004912, 0.005457, 0.005467, 0.005509, 0.005405, 0.004489, 1.000000, 0.011067, 0.015128, 0.016616, 0.017590, 0.019202, 0.020888, 0.012041, 0.013358, 0.014782, 0.015337, 0.016092, 0.019060, 0.019296, 0.017340, 0.017876, 0.017440, 0.017866, 0.018227, 0.020490, 0.020486, 0.021228, 0.021578, 0.020704, 0.020772, 0.020789, 0.021573, 0.021008, 0.020074, 0.020808, 0.020680, 0.021039, 0.021020, 0.021832, 0.021053, 1.000000, 0.017142, 0.020374, 0.021771, 0.022672, 0.023871, 0.027616, 0.020495, 0.020412, 0.021020, 0.021292, 0.022220, 0.024902, 0.025455, 0.025901, 0.025998, 0.025534, 0.026004, 0.026924, 0.029488, 0.028672, 0.030320, 0.030644, 0.029862, 0.029440, 0.030131, 0.030337, 0.029542, 0.028822, 0.029496, 0.029163, 0.029372, 0.029607, 0.029677, 0.029154, 1.000000, 0.019588, 0.023375, 0.025186, 0.026549, 0.027690, 0.032825, 0.022608, 0.022368, 0.023751, 0.024789, 0.026206, 0.028828, 0.031204, 0.029521, 0.028946, 0.029048, 0.029811, 0.031332, 0.034216, 0.034425, 0.033819, 0.033175, 0.032916, 0.032621, 0.034345, 0.035183, 0.035350, 0.032366, 0.031808, 0.031760, 0.032334, 0.033957, 0.034238, 0.035511, 1.000000, 0.021031, 0.024839, 0.026900, 0.029234, 0.031482, 0.038156, 0.023394, 0.023217, 0.024990, 0.026703, 0.028751, 0.032215, 0.036847, 0.031946, 0.030262, 0.030784, 0.031717, 0.034047, 0.037944, 0.039734, 0.036559, 0.034139, 0.034255, 0.034313, 0.037041, 0.039349, 0.040625, 0.035422, 0.033201, 0.033212, 0.034391, 0.037147, 0.038418, 0.041555, 1.000000, 0.021834, 0.026325, 0.028887, 0.032243, 0.035705, 0.043189, 0.023642, 0.024034, 0.025885, 0.028204, 0.031043, 0.035222, 0.042172, 0.033732, 0.030973, 0.031837, 0.033075, 0.036076, 0.041298, 0.045349, 0.039051, 0.034779, 0.035368, 0.035758, 0.038866, 0.042746, 0.046198, 0.038333, 0.034583, 0.034958, 0.036245, 0.039226, 0.042107, 0.047798
        };

        double[] normalized = new double[SCALES.length];
        for (int i = 0; i < SCALES.length; i++) {
            normalized[i] = (inputData[i] - MEANS[i]) / SCALES[i];
        }

        System.out.println(Arrays.toString(normalized));
        speedInputFeature.loadBuffer(doubleToByteBuffer(normalized));

        // Runs model inference and gets result.
        SpeedPredictionModel.Outputs outputs = speedPredictionModel.process(speedInputFeature);
        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        return outputFeature0.getFloatValue(0);

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
