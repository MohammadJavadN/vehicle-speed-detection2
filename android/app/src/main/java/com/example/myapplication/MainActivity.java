package com.example.myapplication;

import static com.google.mlkit.vision.BitmapUtils.matToBitmap;
import static java.lang.Math.abs;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.myapplication.ml.LicensePlateDetectorFloat32;
import com.example.myapplication.ml.SpeedPredictionModel;
import com.example.myapplication.ml.SpeedPredictionModelSideView;
import com.example.myapplication.ml.SpeedPredictionTopViewNoPlateModel;
import com.google.mlkit.vision.GraphicOverlay;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.ObjectDetection;
import com.google.mlkit.vision.objects.ObjectDetector;
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
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.Request;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity implements View.OnTouchListener {

    public static final String TAG = "ObjectDetector";
    static final int STATE = 3;
    private static final Scalar speedColor = new Scalar(255, 0, 0);
    private static final int PERMISSION_REQUEST_CODE = 100;
    private static final String outCSVFileName = "out.csv";
    private static final String outVideoFileName = "out.mp4";
    private static final Size WIN_SIZE = new Size(15, 15);
    private static final TermCriteria CRITERIA = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
            10,
            0.03);
    private static final int MAX_LEVEL = 2;
    public static int DETECTION_MODE = ObjectDetectorOptions.SINGLE_IMAGE_MODE;
    public static boolean isBusy = false;
    static int state = 1;
    static Paint pointPaint, linePaint;
    static Paint textPaint;
    private static String inVideoPath = "/sdcard/Download/FILE0030.mp4"; // tof (ASE)
    //    private static String inVideoPath = "/sdcard/Download/side_vid.mp4"; // side
//    private static String inVideoPath = "/sdcard/Download/Video(1).mp4"; // top
    private static String outVideoPath = "/sdcard/Download/ou_.mp4";
    private static int maxFrames = 500;
    private static VideoCapture cap;
    private static MyVideoEncoder out;
    private static LicensePlateDetectorFloat32 plateDetectorModel;
    private static SpeedPredictionTopViewNoPlateModel speedPredictionTopViewNoPlateModel;
    private static SpeedPredictionModel speedPredictionModel;
    private static SpeedPredictionModelSideView speedPredictionModelSideView;
    private static SpeedPredictionModel OptFlowSpeedPredictionModel;
    private static TensorBuffer speedInputFeature;
    private static TensorBuffer plateInputFeature;
    private static TensorBuffer topSpeedInputFeature1;
    private static TensorBuffer topSpeedInputFeature2;
    private static TensorBuffer sideSpeedInputFeature;
    private static int height, width;
    private static Mat prevGray;
    private static List<Point> prevPts = new ArrayList<>();
    private static int frameNum;
    private static Size winSize;
    private static TermCriteria criteria;
    private static int imW;
    private static int imH;
    private static String outCSVPath = "/sdcard/Download/out.csv";

    static {
        pointPaint = new Paint();
        pointPaint.setColor(Color.RED);
        pointPaint.setStyle(Paint.Style.FILL);

        linePaint = new Paint();
        linePaint.setColor(Color.RED);
        linePaint.setStrokeWidth(5);
    }

    static {
        textPaint = new Paint();
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(40.0f);
    }

    SurfaceView surfaceView;
    String serverUrl = "http://192.168.43.226:5000/";
    Yolov8ObjectDetector yolov8ObjectDetector;
    Interpreter interpreter;
    boolean finished = false, flag = true;
    TensorImage image;
    //    InputImage image;
    Bitmap bitmap;
    Mat frame;
    int stopTime = 100;
    MainActivity context;
    Mat frame2;
    InputImage image2;
    Bitmap bitmap2;
    int FRAME_STEP = 1;
    int REQUEST_VIDEO_CODE = 1;
    MatOfPoint2f prevPtsMat;
    double sx = 1, sy = 1;
    int frameNum1, frameNum2;
    double x1, y1, x2, y2;
    double x1c, y1c, x2c, y2c;
    boolean p1Selected = false;
    boolean p2Selected = false;
    boolean pause = false;
    private ScheduledExecutorService scheduledExecutorService;
    private ObjectTrackerProcessor trackerProcessor;
    private ObjectDetector objectDetector;
    //    private ObjectDetector objectDetector;
    private TOFSpeedDetector tofSpeedDetector;
    private OptSpeedDetector topSpeedDetector;
    private OptSpeedDetector sideSpeedDetector;
    private CPSpeedDetector CPSpeedDetector;
    private ServerSpeedDetector serverSpeedDetector;
    private GraphicOverlay graphicOverlay;
    private View circle1, circle2, circle3, circle4;
    private boolean isServer, isTOF, isSide, isCP, serverWithSpeed;

    public static int getId(float x, float y, int imW) {
        int gs = (int) (imW / 4.8);  // grid_width
        return (int) ((y / gs) * (imW / gs) + (x / gs));
    }

    public static int getLane(int x, int y) {
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

    public static Mat predictWithoutPlate(boolean sideView) {
        Mat frame = new Mat();
        if (MainActivity.frameNum < MainActivity.maxFrames) {
            boolean ret = cap.read(frame);
            if (!ret) {
                return null;
            }

            MainActivity.frameNum++;
            imH = frame.rows();
            imW = frame.cols();

            Mat imageRgb = new Mat();
            Imgproc.cvtColor(frame, imageRgb, Imgproc.COLOR_BGR2RGB);

            Mat frameGray = new Mat();
            Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_BGR2GRAY);


            try {
                if (MainActivity.frameNum > 2) {
                    float[] laneSpeeds = {0, 0, 0, 0};
                    int[] cntOfLaneSpeeds = {0, 0, 0, 0};

                    MatOfPoint prevPtsMat_ = new MatOfPoint();

                    Imgproc.goodFeaturesToTrack(MainActivity.prevGray, prevPtsMat_, 150, 0.1, 5, new Mat(), 7, false, 0.04);

                    MatOfPoint2f prevPtsMat = new MatOfPoint2f(prevPtsMat_.toArray());

                    MatOfPoint2f nextPts = new MatOfPoint2f();
                    // Calculate optical flow using Lucas-Kanade method
                    MatOfByte status = new MatOfByte();
                    MatOfFloat err = new MatOfFloat();
                    Video.calcOpticalFlowPyrLK(
                            MainActivity.prevGray, frameGray, prevPtsMat,
                            nextPts, status, err, MainActivity.winSize,
                            0, MainActivity.criteria);

                    byte[] StatusArr = status.toArray();
                    Point[] p0Arr = prevPtsMat.toArray();
                    Point[] p1Arr = nextPts.toArray();
                    Mat mask = Mat.zeros(frame.size(), CvType.CV_8UC3);

                    for (int i = 0; i < StatusArr.length; i++) {
                        if (StatusArr[i] == 1) {
                            Point newPt = p0Arr[i];

                            Point oldPt = p1Arr[i];

                            double a = newPt.x;
                            double b = newPt.y;
                            double c = oldPt.x;
                            double d = oldPt.y;

                            if (b < 0.25 * imH || d < b || abs(a - c) > (double) imW / 100 || abs(b - d) > imH * 0.14)
                                continue;

                            int lane = getLane((int) newPt.x, (int) newPt.y);

                            double pixelSpeed = Math.sqrt(Math.pow(a - c, 2) + Math.pow(b - d, 2));

                            double[] input_data = {a, b, pixelSpeed};

                            // Get output tensor (predicted_speed_function)
                            float predictedSpeed = predictSpeedNoPlate(input_data, sideView);
                            if (pixelSpeed > 15) {
                                if (!sideView) {
                                    predictedSpeed = (float) (predictedSpeed * Math.pow(1920 / imW, 0.08) * 1.2);
                                    laneSpeeds[lane - 1] += predictedSpeed;
                                    cntOfLaneSpeeds[lane - 1]++;
                                } else {
                                    laneSpeeds[0] += predictedSpeed;
                                    cntOfLaneSpeeds[0]++;
                                }
                                Imgproc.line(mask, newPt, oldPt, new Scalar(255, 0, 0), 4 * imW / 1920);
                                Imgproc.circle(mask, newPt, (5 * imW) / 1920, new Scalar(255, 0, 0), -1);
                            }
                        }
                    }
                    if (sideView) {
                        // Draw lines and circles
                        int speed = (int) (laneSpeeds[0] / cntOfLaneSpeeds[0] * 10);

                        if (speed < 20)
                            speed = 0;
                        Imgproc.putText(frame,
                                Float.toString((float) speed / 10),
                                new Point(0.45 * imW, 55),
                                Imgproc.FONT_HERSHEY_PLAIN,
                                5.5 * imW / 1920,
                                speedColor,
                                6 * imW / 1920);
                    } else
                        for (int lane = 0; lane < 4; lane++) {
                            // Draw lines and circles
                            int speed = (int) (laneSpeeds[lane] / cntOfLaneSpeeds[lane] * 10);

                            if (speed < 5)
                                speed = 0;
                            Imgproc.putText(frame,
                                    Float.toString((float) speed / 10),
                                    new Point((lane + 1) * 0.208 * imW, 55),
                                    Imgproc.FONT_HERSHEY_PLAIN,
                                    4.5 * imW / 1920,
                                    speedColor,
                                    6 * imW / 1920);

                        }
                    // Add mask to frame
                    Core.add(frame, mask, frame);
                }
            } catch (Exception e) {
                System.out.println(e);
            }

            MainActivity.prevGray = frameGray.clone();

            // Return the processed frame
            return frame;

        }
        return null;

    }

    public static Mat predictAndVisualize() {
        Mat frame = new Mat();
        if (MainActivity.frameNum < MainActivity.maxFrames) {
            boolean ret = cap.read(frame);
            if (!ret) {
                return null;
            }

            MainActivity.frameNum++;
            imH = frame.rows();
            imW = frame.cols();

            Mat imageRgb = new Mat();
            Imgproc.cvtColor(frame, imageRgb, Imgproc.COLOR_BGR2RGB);


            Mat imageResized = new Mat();
            Imgproc.resize(imageRgb, imageResized, new Size(MainActivity.width, MainActivity.height));

            System.out.println(imageResized.get(0, 0)[0]);
            Core.normalize(imageResized, //todo: imageResized to input
                    imageResized, 0.0, 1.0, Core.NORM_MINMAX, CvType.CV_32FC3);
            System.out.println(imageResized.get(0, 0)[0]);


            Map<Integer, List<Float>> plates = predictPlates(imageResized);

            Mat frameGray = new Mat();
            Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_BGR2GRAY);


            try {
                if (MainActivity.frameNum > 2 && !MainActivity.prevPts.isEmpty()) {

                    // Convert prevPts to float32
                    MatOfPoint2f prevPtsMat = new MatOfPoint2f();
                    prevPtsMat.fromList(prevPts);

                    MatOfPoint2f nextPts = new MatOfPoint2f();
                    // Calculate optical flow using Lucas-Kanade method
                    Video.calcOpticalFlowPyrLK(
                            MainActivity.prevGray, frameGray, prevPtsMat,
                            nextPts, new MatOfByte(), new MatOfFloat(), MainActivity.winSize,
                            5, MainActivity.criteria);
                    Point[] p0Arr = prevPtsMat.toArray();
                    Point[] p1Arr = nextPts.toArray();
                    // Draw the tracks
                    //    private MatOfPoint2f nextPts = new MatOfPoint2f();
                    Mat mask = Mat.zeros(frame.size(), CvType.CV_8UC3);

                    for (int i = 0; i < prevPts.size(); i++) {
                        Point newPt = p0Arr[i];
                        Point oldPt = p1Arr[i];

                        double a = newPt.x;
                        double b = newPt.y;
                        double c = oldPt.x;
                        double d = oldPt.y;

                        if (b < 0.18 * imH) {
                            continue;
                        }

                        double pixelSpeed = Math.sqrt(Math.pow(a - c, 2) + Math.pow(b - d, 2));

                        double[] input_data = {a, b, pixelSpeed};

                        // Get output tensor (predicted_speed_function)
                        float predictedSpeed = predictSpeedWithPlate(input_data, false) * 10;
                        int speed = (int) predictedSpeed;
                        // Draw lines and circles
                        Imgproc.line(mask, newPt, oldPt, new Scalar(255, 0, 0), 2);
                        Imgproc.circle(mask, newPt, 5, new Scalar(255, 0, 0), -1);
                        // Draw label text
                        Imgproc.putText(frame,
                                Float.toString((float) speed / 10),
                                newPt,
                                Imgproc.FONT_HERSHEY_PLAIN,
                                4.5 * imW / 1920,
                                speedColor,
                                6 * imW / 1920);

                        int lane = getLane((int) newPt.x, (int) newPt.y);
                        Imgproc.putText(frame,
                                Float.toString((float) speed / 10),
                                new Point(lane * 0.208 * imW, 55),
                                Imgproc.FONT_HERSHEY_PLAIN,
                                4.5 * imW / 1920,
                                speedColor,
                                6 * imW / 1920);

                    }
                    // Add mask to frame
                    Core.add(frame, mask, frame);
                }
            } catch (Exception e) {
                System.out.println(e);
            }

            MainActivity.prevPts = new ArrayList<>();
            for (List<Float> sumBoxCnt : plates.values()) {
                float sumX = sumBoxCnt.get(0);
                float sumY = sumBoxCnt.get(1);
                float sumW = sumBoxCnt.get(2);
                float sumH = sumBoxCnt.get(3);
                float cnt = sumBoxCnt.get(4);

                float xCenter = sumX / cnt;
                float yCenter = sumY / cnt;
                float w = sumW / cnt;
                float h = sumH / cnt;

                MainActivity.prevPts.add(new Point((int) (xCenter * imW), (int) (yCenter * imH)));

                int x1 = (int) ((xCenter - w / 2) * imW);
                int y1 = (int) ((yCenter - h / 2) * imH);
                int x2 = (int) ((xCenter + w / 2) * imW);
                int y2 = (int) ((yCenter + h / 2) * imH);

                Imgproc.rectangle(frame, new Point(x1, y1), new Point(x2, y2), speedColor, 2);
            }

            MainActivity.prevGray = frameGray.clone();

            // Return the processed frame
            return frame;

        }
        return null;

    }

    public static ByteBuffer doubleToByteBuffer(double[] data) {
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

    private static float predictSpeedNoPlate(double[] inputData, boolean sideView) {
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

    private static float predictSpeedWithPlate(double[] inputData, boolean sideView) {
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
            double MEAN_A = 936.88328756;
            double MEAN_B = 617.87426442;
            double MEAN_P = 42.33951691;
            double VAR_A = 550.843393004;
            double VAR_B = 322.851840306;
            double VAR_P = 26.414411479;
            double[] normalized;
            normalized = new double[]{(inputData[0] - MEAN_A) / VAR_A,
                    (inputData[1] - MEAN_B) / VAR_B,
                    (inputData[2] - MEAN_P) / VAR_P};
            topSpeedInputFeature1.loadBuffer(doubleToByteBuffer(normalized));

            // Runs model inference and gets result.
            SpeedPredictionModel.Outputs outputs = speedPredictionModel.process(topSpeedInputFeature1);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            return outputFeature0.getFloatValue(0);
        }
    }

    public static ByteBuffer matToByteBuffer(Mat mat) {
        // Ensure the Mat is of type CV_32FC3
        if (mat.type() != CvType.CV_32FC3) {
            throw new IllegalArgumentException("Input Mat must be of type CV_32FC3");
        }

        // Get the number of bytes needed for the Mat data
        int numChannels = mat.channels();
        int bufferSize = (int) (mat.total() * numChannels * 4);
        System.out.println(mat.total() + ", " + numChannels + ", " + 4);
        // Create a ByteBuffer with the appropriate size
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bufferSize);
        byteBuffer.order(ByteOrder.nativeOrder()); // Set the byte order to native

        // Get the data elements from the Mat and put them into the ByteBuffer
        float[] data = new float[(int) mat.total() * numChannels];
        mat.get(0, 0, data);

        // Check if the buffer has enough capacity to hold all the data
        if (byteBuffer.remaining() < bufferSize) {
            throw new RuntimeException("ByteBuffer does not have enough capacity to hold all the data");
        }

        // Put the data into the ByteBuffer
        for (float value : data) {
            byteBuffer.putFloat(value);
        }

        byteBuffer.rewind(); // Rewind the buffer to the beginning

        return byteBuffer;
    }

    public static Map<Integer, List<Float>> predictPlates(Mat frame) {
        try {
            plateInputFeature.loadBuffer(matToByteBuffer(frame));
        } catch (Exception e) {
            System.out.println(e);
        }

        // Runs model inference and gets result.
        LicensePlateDetectorFloat32.Outputs outputs = plateDetectorModel.process(plateInputFeature);
        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

        return computeBoxesAndScores(outputFeature0);
    }

    private static Map<Integer, List<Float>> computeBoxesAndScores(TensorBuffer outFeature) {
        System.out.println(Arrays.toString(outFeature.getShape()));

        float[] floats = outFeature.getFloatArray();
        final int N = outFeature.getShape()[2];
        final float threshold = 0.09f;
        Map<Integer, List<Float>> plates = new HashMap<>();
        for (int i = 0; i < N; i++) {
            if (floats[4 * N + i] > threshold) {
                System.out.println("it is bigger than threshold!");
                float xCenter = floats[i];
                float yCenter = floats[i + N];
                float w = floats[i + 2 * N];
                float h = floats[i + 3 * N];

                if ((yCenter > 0.104) && (w < 0.093) && (h < 0.052)) {
                    int carId = getId(xCenter * imW, yCenter * imH, imW);

                    List<Float> sumBoxCnt = plates.getOrDefault(carId, Arrays.asList(0f, 0f, 0f, 0f, 0f));
                    assert sumBoxCnt != null;
                    float sumX = sumBoxCnt.get(0);
                    float sumY = sumBoxCnt.get(1);
                    float sumW = sumBoxCnt.get(2);
                    float sumH = sumBoxCnt.get(3);
                    float cnt = sumBoxCnt.get(4);

                    sumBoxCnt.set(0, sumX + xCenter);
                    sumBoxCnt.set(1, sumY + yCenter);
                    sumBoxCnt.set(2, sumW + w);
                    sumBoxCnt.set(3, sumH + h);
                    sumBoxCnt.set(4, cnt + 1);

                    plates.put(carId, sumBoxCnt);
                }
                System.out.println("go for next...");

            }
        }
        System.out.println("all plates of this frame are detected!");
        return plates;

    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        getPermission();
        init();

        graphicOverlay = findViewById(R.id.overlayView);

        surfaceView = findViewById(R.id.surfaceView);
        surfaceView.setOnTouchListener(this);
        initializeCircles();

        setupObjectDetector();
    }

    private void setupObjectDetector() {
        Log.d(TAG, "setupObjectDetector");
//***************************** ml kit model *********************************
//        ObjectDetectorOptions options = new ObjectDetectorOptions.Builder()
//                .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
//                .enableClassification()  // Optional: Enable classification
//                .build();
//        objectDetector = ObjectDetection.getClient(options);


//*********************** costume classifier in ml-kit model *****************
//        LocalModel localModel =
//                new LocalModel.Builder()
//                        .setAssetFilePath("efficientnet.tflite")
//                        // or .setAbsoluteFilePath(absolute file path to model file)
//                        // or .setUri(URI to model file)
//                        .build();
//
//        // Multiple object detection in static images
//        CustomObjectDetectorOptions customObjectDetectorOptions =
//                new CustomObjectDetectorOptions.Builder(localModel)
//                        .setDetectorMode(CustomObjectDetectorOptions.SINGLE_IMAGE_MODE)
//                        .enableMultipleObjects()
//                        .enableClassification()
//                        .setClassificationConfidenceThreshold(0.5f)
//                        .setMaxPerObjectLabelCount(3)
//                        .build();
//        objectDetector = ObjectDetection.getClient(customObjectDetectorOptions);

//***************************** tflite model *********************************
//        yolov8ObjectDetector = new Yolov8ObjectDetector();
//        yolov8ObjectDetector.setModelFile("yolov8n_float32.tflite");
//        yolov8ObjectDetector.initialModel(this);

//***************************** ml kit model *********************************
        // Multiple object detection in static images
        ObjectDetectorOptions options =
                new ObjectDetectorOptions.Builder()
                        .setDetectorMode(DETECTION_MODE)
                        .enableMultipleObjects()
//                        .enableClassification()  // Optional
                        .build();

        objectDetector = ObjectDetection.getClient(options);
//        trackerProcessor = new ObjectTrackerProcessor(this, options, tofSpeedDetector);
    }

    private void initializeSurface() {
        graphicOverlay.roadLine.initializeCircles(circle1, circle2, circle3, circle4);
//        findViewById(R.id.changBtn).setVisibility(View.VISIBLE);
    }

    public void changeCircles(android.view.View view) {
        switch (state) {
            case 0:
                RoadLine.setCirclesTop(circle1, circle2, circle3, circle4);
                break;
            case 1:
                RoadLine.setCirclesSide1(circle1, circle2, circle3, circle4);
                break;
            case 2:
                RoadLine.setCirclesSide2(circle1, circle2, circle3, circle4);
                break;
        }
        state = (state + 1) % STATE;
    }

    private void initializeCircles() {
        circle1 = findViewById(R.id.circle1);
        circle2 = findViewById(R.id.circle2);
        circle3 = findViewById(R.id.circle3);
        circle4 = findViewById(R.id.circle4);

        circle1.setOnTouchListener(this);
        circle2.setOnTouchListener(this);
        circle3.setOnTouchListener(this);
        circle4.setOnTouchListener(this);
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouch(View v, MotionEvent event) {
        System.out.println("*** onTouch: " + event.getAction());
        if (v.getId() == R.id.surfaceView && event.getAction() == MotionEvent.ACTION_UP) {
            Utils.matToBitmap(frame2, bitmap2);
            Canvas canvas = surfaceView.getHolder().lockCanvas();
            if (canvas != null) {
                // Render the frame onto the canvas
                int w = canvas.getWidth();
                int h = canvas.getHeight();
                Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap2, w, h, false);
                canvas.drawBitmap(scaledBitmap, 0, 0, null);
            }
            if (!p1Selected) {
                double sw = surfaceView.getWidth();
                double sh = surfaceView.getHeight();
                sx = imW / sw;
                sy = imH / sh;

                x1c = event.getRawX() - surfaceView.getX();
                y1c = event.getRawY() - surfaceView.getY();
                x1 = (x1c) * sx;
                y1 = (y1c) * sy;

                Point userSelectedP = new Point(x1, y1);
                Point selectedP = new Point(x1, y1);
                prevGray = new Mat();
                Imgproc.cvtColor(frame2, prevGray, Imgproc.COLOR_BGR2GRAY);
//                MatOfPoint p0MatofPoint = new MatOfPoint();
//                Imgproc.goodFeaturesToTrack(prevGray, p0MatofPoint,1000,0.3,7, new Mat(),7,false,0.04);
//
//                Point[] points = p0MatofPoint.toArray();
//                double minD = 300;
//                for (Point p:points) {
//                    double d = d2(p, userSelectedP);
//                    if (d < minD) {
//                        minD = d;
//                        selectedP = p.clone();
//                    }
//                }
//                x1c = selectedP.x/sx;
//                y1c = selectedP.y/sy;
//                x1 = (x1c) * sx;
//                y1 = (y1c) * sy;

                prevPts = new ArrayList<>();
                prevPts.add(selectedP);
                prevPtsMat = new MatOfPoint2f();
                prevPtsMat.fromList(prevPts);

                p1Selected = true;
                Toast.makeText(
                        getApplicationContext(),
                        "x1= " + x1 + ", y1= " + y1,
                        Toast.LENGTH_LONG
                ).show();
                frameNum1 = frameNum;

                if (canvas != null) {
                    canvas.drawCircle(
                            (float) x1c,
                            (float) y1c,
                            20,
                            pointPaint);
                }
            } else if (!p2Selected) {
                x2c = event.getRawX() - surfaceView.getX();
                y2c = event.getRawY() - surfaceView.getY();
                x2 = (x2c) * sx;
                y2 = (y2c) * sy;
                prevPts.clear();
                prevPts.add(new Point(x2, y2));
                prevPtsMat.fromList(prevPts);
                Imgproc.cvtColor(frame2, prevGray, Imgproc.COLOR_BGR2GRAY);

                p2Selected = true;
                Toast.makeText(
                        getApplicationContext(),
                        "x2= " + x2 + ", y2= " + y2,
                        Toast.LENGTH_LONG
                ).show();
                frameNum2 = frameNum;

                if (canvas != null) {
                    canvas.drawLine((float) x1c, (float) y1c, (float) (x2c), (float) (y2c), linePaint);
                    canvas.drawCircle(
                            (float) x2c,
                            (float) y2c,
                            20,
                            pointPaint);
                    int speed = CPSpeedDetector.predict(x1, y1, x2, y2) / (frameNum2 - frameNum1);
                    canvas.drawText(
                            "Speed: " + speed,
                            50,
                            150,
                            textPaint);
                }
            } else {
                p1Selected = false;
                p2Selected = false;
            }
            if (canvas != null) {
                surfaceView.getHolder().unlockCanvasAndPost(canvas);
            }

        } else if (event.getAction() == MotionEvent.ACTION_MOVE && v.getId() != R.id.surfaceView)
            graphicOverlay.roadLine.movePoint(v, event);
//        if (graphicOverlay.show(true))
//            findViewById(R.id.floatingActionButton).setVisibility(View.VISIBLE);
        return true;
    }

    private double d2(Point p1, Point p2) {
        return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
    }

    void getPermission() {
        if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    PERMISSION_REQUEST_CODE);
        } else if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.READ_MEDIA_VIDEO)
                != PackageManager.PERMISSION_GRANTED) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                ActivityCompat.requestPermissions(MainActivity.this,
                        new String[]{Manifest.permission.READ_MEDIA_VIDEO},
                        PERMISSION_REQUEST_CODE);
            }
        } else if (ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.INTERNET)
                != PackageManager.PERMISSION_GRANTED) {
            System.out.println("*** has not internet access..");
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.INTERNET},
                    PERMISSION_REQUEST_CODE);

        }
    }

    public String getRealPathFromURI(Context context, Uri contentUri) {
        String filePath = null;
        Cursor cursor = null;
        try {
            String[] projection = {MediaStore.Video.Media.DATA};
            cursor = context.getContentResolver().query(contentUri, projection, null, null, null);
            if (cursor != null && cursor.moveToFirst()) {
                int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DATA);
                filePath = cursor.getString(columnIndex);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (cursor != null) {
                cursor.close();
            }
        }
        return filePath;
    }

    private void processImage() {
        frame = new Mat();
        if (cap.read(frame)) {
            if (!frame.empty()) {
                Utils.matToBitmap(frame, bitmap);

//                // Resize the frame to the expected input size (640x640)
//                Mat resizedFrame = new Mat();
//                Size size = new Size(640, 640);
//                Imgproc.resize(frame, resizedFrame, size);
//
//                // Convert the resized frame to a bitmap
//                Bitmap resizedBitmap = Bitmap.createBitmap(resizedFrame.cols(), resizedFrame.rows(), Bitmap.Config.ARGB_8888);
//                Utils.matToBitmap(resizedFrame, resizedBitmap);
//
//                // Creates inputs for reference
//                TensorImage image = TensorImage.fromBitmap(resizedBitmap);
//
//                // Debug: Check input shape
//                int[] inputShape = image.getTensorBuffer().getShape();
//                Log.d("InputShape", "Input shape: " + Arrays.toString(inputShape));
//
//                // Runs model inference and gets result
//                Yolov8nFloat32.Outputs outputs = objectDetector.process(image);
//
//                ArrayList<DetectedObject> detectedObjects = extractObjects(outputs, frame.width(),frame.height());

                ArrayList<Recognition> recognitions = yolov8ObjectDetector.detect(bitmap);
                Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);

                Canvas canvas = surfaceView.getHolder().lockCanvas();
//                tofSpeedDetector.detectSpeeds(frame, detectedObjects, canvas); // todo

                Paint boxPaint = new Paint();
                boxPaint.setStrokeWidth(5);
                boxPaint.setStyle(Paint.Style.STROKE);
                boxPaint.setColor(Color.RED);

                Paint textPain = new Paint();
                textPain.setTextSize(50);
                textPain.setColor(Color.GREEN);
                textPain.setStyle(Paint.Style.FILL);

//                runOnUiThread(() -> {
                if (canvas != null) {
                    // Render the frame onto the canvas
                    int w = canvas.getWidth();
                    int h = canvas.getHeight();

//                        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, w, h, false);
                    canvas.drawBitmap(bitmap, 0, 0, null);

                    for (Recognition recognition : recognitions) {
                        if (recognition.getConfidence() > 0.4) {
                            RectF location = recognition.getLocation();
                            canvas.drawRect(location, boxPaint);
                            canvas.drawText(recognition.getLabelName() + ":" + recognition.getConfidence(), location.left, location.top, textPain);
                        }
                    }
                    surfaceView.getHolder().unlockCanvasAndPost(canvas);
                }
//                });
                flag = true;
            }
        } else
            finished = true;
    }

    private void startProcess() {
        releaseResources();

        initialInOutVideo();

        scheduledExecutorService = Executors.newSingleThreadScheduledExecutor();
        context = this;
        frame2 = new Mat();
        Runnable updateFrameTask;
        if (isCP) {
            updateFrameTask = () -> {
                if (!isBusy) {
                    boolean ret = false;
                    if (!pause)
                        ret = cap.read(frame2);
                    if (ret) {
                        if (frameNum == 0) {
                            Bitmap bitmap = matToBitmap(frame2);
                            initialParameters(bitmap);
                        }
                        frameNum++;
                        if (frameNum < 2)
                            return;

                        isBusy = true;
                        bitmap2 = Bitmap.createBitmap(frame2.cols(), frame2.rows(), Bitmap.Config.ARGB_8888);
                        processImageCP();
                        isBusy = false;

                    } else if (!pause) {
                        onDestroy();
                        System.out.println("*** destroy!!!");
                    }
                }
            };
        } else {
            updateFrameTask = () -> {
                if (!isBusy) {
//                if (isOutAvailable && graphicOverlay.isValidBitmap)
//                    out.encodeFrame(graphicOverlay.getBitmap());

                    Mat frame = new Mat();
                    boolean ret = cap.read(frame);
                    if (ret) {
                        if (frameNum == 0) {
                            Bitmap bitmap = matToBitmap(frame);
                            initialParameters(bitmap);
                        }
                        frameNum++;
//                    if (frameNum < 2469)
                        if (frameNum < 2)
                            return;
//                    stopTime = 160;
                        try {
                            isBusy = true;
//                        trackerProcessor.processBitmap(bitmap, graphicOverlay);
                            scheduledExecutorService.shutdown();
                            bitmap2 = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
                            processImage2();
                        } catch (Exception e) {
                            Log.e(TAG, "Failed to process image. Error: " + e.getLocalizedMessage());
                            Toast.makeText(getApplicationContext(), e.getLocalizedMessage(), Toast.LENGTH_SHORT)
                                    .show();
                        }
                    } else
                        onDestroy();
                }
            };

        }

        // Schedule the task to run every 33 milliseconds (30 frames per second)
        scheduledExecutorService.scheduleAtFixedRate(
                updateFrameTask,
                0, // Initial delay
                stopTime, // Period (milliseconds)
                TimeUnit.MILLISECONDS);
    }

    public void pause(View v) {
        findViewById(R.id.pause).setVisibility(View.INVISIBLE);
        findViewById(R.id.resume).setVisibility(View.VISIBLE);
        pause = true;
    }

    public void resume(View v) {
        findViewById(R.id.resume).setVisibility(View.INVISIBLE);
        findViewById(R.id.pause).setVisibility(View.VISIBLE);
        pause = false;
    }

    public void processImageCP() {
        System.out.println("*** processImageCP");

        Utils.matToBitmap(frame2, bitmap2);
        Canvas canvas = surfaceView.getHolder().lockCanvas();
        if (canvas != null) {
            // Render the frame onto the canvas
            int w = canvas.getWidth();
            int h = canvas.getHeight();
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap2, w, h, false);
            canvas.drawBitmap(scaledBitmap, 0, 0, null);
        }
        if (p1Selected) {
            if (canvas != null) {

                MatOfByte status = new MatOfByte();
                MatOfFloat errors = new MatOfFloat();

                // Calculate optical flow
                Mat frameGray = new Mat();
                Imgproc.cvtColor(frame2, frameGray, Imgproc.COLOR_BGR2GRAY);

                MatOfPoint2f nextPts = new MatOfPoint2f();
                Video.calcOpticalFlowPyrLK(
                        prevGray, frameGray, prevPtsMat, nextPts, status, errors, WIN_SIZE, MAX_LEVEL, CRITERIA);

                List<Point> nextPtsList = nextPts.toList();
                prevPtsMat = new MatOfPoint2f();
                prevPtsMat.fromList(nextPtsList);
                prevGray = frameGray.clone();

                double x = nextPtsList.get(0).x;
                double y = nextPtsList.get(0).y;
                canvas.drawCircle(
                        (float) (x / sx),
                        (float) (y / sy),
                        20,
                        pointPaint);
                canvas.drawLine((float) x1c, (float) y1c, (float) (x / sx), (float) (y / sy), linePaint);
                int speed = CPSpeedDetector.predict(x1, y1, x, y) / (frameNum - frameNum1);
                canvas.drawText(
                        "Speed: " + speed,
                        50,
                        50,
                        textPaint);
            }
        }
        if (p2Selected) {
            if (canvas != null) {
                int speed = CPSpeedDetector.predict(x1, y1, x2, y2) / (frameNum2 - frameNum1);
                canvas.drawText(
                        "Speed: " + speed,
                        50,
                        150,
                        textPaint);
            }
        }
//        runOnUiThread(() -> {
        if (canvas != null) {
            surfaceView.getHolder().unlockCanvasAndPost(canvas);
        }
//        });
    }

    public void processImage2() {
        System.out.println("public static void processImage2()");
        boolean ret;
        frame2 = new Mat();
        do {
            ret = cap.read(frame2);
            frameNum++;
        }
        while (frameNum % FRAME_STEP != 0);
        if (ret) {
            Utils.matToBitmap(frame2, bitmap2);
            image2 = InputImage.fromBitmap(bitmap2, 0);

            if (isServer) {

                Request request = serverSpeedDetector.sendFrameToServer(bitmap2);
                serverSpeedDetector.client.newCall(request).enqueue(new Callback() {
                    @Override
                    public void onFailure(Call call, IOException e) {
                        Log.e("ServerSpeedDetector", "Failed to send frame", e);
                    }

                    @Override
                    public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                        if (response.isSuccessful()) {
                            String responseData = response.body().string();

                            // Parse the response into List<DetectedObject>
                            List<DetectedObject> detectedObjects = ServerSpeedDetector.parseDetectedObjects(responseData);

                            // Log or use the detected objects
                            for (DetectedObject obj : detectedObjects) {
                                Log.d("ServerSpeedDetector", "Detected object: " + obj.getTrackingId() + " Bbox: " + obj.getBoundingBox());
                            }

                            Mat frame2 = new Mat();
                            Utils.bitmapToMat(image2.getBitmapInternal(), frame2);
                            Canvas canvas = surfaceView.getHolder().lockCanvas();
                            if (isCP)
                                CPSpeedDetector.detectSpeeds(frame2, detectedObjects, canvas);
                            else if (isTOF)
                                tofSpeedDetector.detectSpeeds(frame2, detectedObjects, canvas);
                            else if (isSide)
                                sideSpeedDetector.detectSpeeds(frame2, detectedObjects, canvas);
                            else
                                topSpeedDetector.detectSpeeds(frame2, detectedObjects, canvas);
//                            runOnUiThread(() -> {
                            if (canvas != null) {
                                surfaceView.getHolder().unlockCanvasAndPost(canvas);
                                try {
                                    Bitmap bitmap1 = Bitmap.createBitmap(canvas.getWidth(), canvas.getHeight(), Bitmap.Config.ARGB_8888);
                                    canvas.drawBitmap(bitmap1, 0, 0, null);
                                    out.encodeFrame(bitmap1);
                                } catch (Exception e) {
                                    System.out.println("*** save video failed: " + e);
                                }
                            }
//                            });
                            processImage2();


                        } else {
                            String responseBody = response.body() != null ? response.body().string() : "No response body";
                            Log.e(TAG, "Server returned an error: " + response.code() + " - " + response.message());
                            Log.e(TAG, "Response body: " + responseBody);
                        }
                    }
                });
            } else {
                objectDetector.process(image2)
                        .addOnSuccessListener(detectedObjects -> {

                            Mat frame2 = new Mat();
                            Utils.bitmapToMat(image2.getBitmapInternal(), frame2);
                            Canvas canvas = surfaceView.getHolder().lockCanvas();
                            if (isCP)
                                CPSpeedDetector.detectSpeeds(frame2, detectedObjects, canvas);
                            else if (isTOF)
                                tofSpeedDetector.detectSpeeds(frame2, detectedObjects, canvas);
                            else if (isSide)
                                sideSpeedDetector.detectSpeeds(frame2, detectedObjects, canvas);
                            else
                                topSpeedDetector.detectSpeeds(frame2, detectedObjects, canvas);
                            runOnUiThread(() -> {
                                if (canvas != null) {
                                    surfaceView.getHolder().unlockCanvasAndPost(canvas);
                                }
                            });
                            processImage2();
                        })
                        .addOnFailureListener(e -> {
                            Log.e(TAG, "Object detection failed", e);
                        });
            }
        }

    }

    private void initialParameters(Bitmap bitmap) {
        graphicOverlay.setImageSourceInfo(bitmap.getWidth(), bitmap.getHeight(), false);
//        trackerProcessor.DISTANCE_TH = (double) bitmap.getWidth() / 10;
        MyDetectedObject.imgWidth = bitmap.getWidth();
        MyDetectedObject.imgHeight = bitmap.getHeight();
        imW = bitmap.getWidth();
        imH = bitmap.getHeight();
    }

    public void saveCsv(View view) {
        HashMap<Integer, HashMap<Integer, Float>> ObjectsSpeed = SpeedDetector.getObjectsSpeed();
        Set<Integer> unnecessaryKey = new HashSet<>();
        for (Integer key : ObjectsSpeed.keySet()) {
            if (Objects.requireNonNull(ObjectsSpeed.get(key)).isEmpty())
                unnecessaryKey.add(key);
        }

        for (Integer key : unnecessaryKey) {
            ObjectsSpeed.remove(key);
        }
        try {
            File file = new File(this.getExternalFilesDir(null), outCSVFileName);
            CsvWriter.saveHashMapToCsv(ObjectsSpeed, file.getPath());
            Toast.makeText(getApplicationContext(),
                    "out_path: " + file.getPath(),
                    Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            Toast.makeText(getApplicationContext(), e.getLocalizedMessage(), Toast.LENGTH_LONG)
                    .show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Release resources
        if (MainActivity.cap != null && MainActivity.cap.isOpened())
            MainActivity.cap.release();
        if (out != null && out.isMuxerStarted())
            out.stopEncoder();

        if (scheduledExecutorService != null) {
            scheduledExecutorService.shutdown();
        }
    }

    private void releaseResources() {
        // Release resources
        if (cap != null && cap.isOpened())
            cap.release();
        if (out != null && out.isMuxerStarted())
            out.stopEncoder();
        if (scheduledExecutorService != null)
            scheduledExecutorService.shutdown();
    }

    private void initialInOutVideo() {
        cap = new VideoCapture();
        cap.open(inVideoPath);

        double fps = cap.get(Videoio.CAP_PROP_FPS);
        int width = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
        int height = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);

        try {
            File file = new File(this.getExternalFilesDir(null), outVideoFileName);
            out = new MyVideoEncoder(width, height, (int) fps, file.getPath());
            out.startEncoder();
//            isOutAvailable = true;
        } catch (IOException e) {
            Toast.makeText(getApplicationContext(), e.getLocalizedMessage(), Toast.LENGTH_SHORT)
                    .show();
        }
        frameNum = 0;

//        RoadLine.globalCoeff *= (float) (fps / 30);
    }

    public void browseVideo(android.view.View view) {
//        fileChooser.launch("video/*");
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("video/*"); // Filter to show only videos
        startActivityForResult(intent, REQUEST_VIDEO_CODE);

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_VIDEO_CODE && resultCode == Activity.RESULT_OK) {
            if (data != null) {


                isServer = ((RadioButton) findViewById(R.id.radioServer)).isChecked();
                serverWithSpeed = ((RadioButton) findViewById(R.id.radioServer_speed)).isChecked();
                if (serverWithSpeed)
                    isServer = true;
                if (isServer) {
                    DETECTION_MODE = ObjectDetectorOptions.STREAM_MODE;
                    String ip1, ip2, ip3, ip4, port;
                    ip1 = ((EditText) findViewById(R.id.ip1)).getText().toString();
                    ip2 = ((EditText) findViewById(R.id.ip2)).getText().toString();
                    ip3 = ((EditText) findViewById(R.id.ip3)).getText().toString();
                    ip4 = ((EditText) findViewById(R.id.ip4)).getText().toString();
                    port = ((EditText) findViewById(R.id.port)).getText().toString();
                    serverUrl = "http://" + ip1 + "." + ip2 + "." + ip3 + "." + ip4 + ":" + port + "/";
                    serverUrl = ((EditText) findViewById(R.id.url)).getText().toString();
                    serverSpeedDetector = new ServerSpeedDetector(serverUrl, serverWithSpeed);
                }
                isTOF = ((RadioButton) findViewById(R.id.radioASE)).isChecked();
                isSide = ((RadioButton) findViewById(R.id.radioSide)).isChecked();
                isCP = ((RadioButton) findViewById(R.id.radioIPSA)).isChecked();

                if (isCP) {

                    float f = Float.parseFloat(((EditText) findViewById(R.id.f)).getText().toString());
                    float XR = Float.parseFloat(((EditText) findViewById(R.id.XR)).getText().toString());
                    float YR = Float.parseFloat(((EditText) findViewById(R.id.YR)).getText().toString());
                    float HR = Float.parseFloat(((EditText) findViewById(R.id.HR)).getText().toString());
                    CPSpeedDetector.init(f, XR, YR, HR);
                    findViewById(R.id.pause).setVisibility(View.VISIBLE);
                    findViewById(R.id.resume).setVisibility(View.INVISIBLE);
                    pause = false;
                } else {
                    findViewById(R.id.pause).setVisibility(View.GONE);
                    findViewById(R.id.resume).setVisibility(View.GONE);
                }

                if (isTOF) {
                    inVideoPath = "/sdcard/Download/FILE0010.mp4"; // tof (ASE)
                    FRAME_STEP = 4;
                } else if (isSide | isCP) {
                    inVideoPath = "/sdcard/Download/side_vid.mp4"; // side
                    inVideoPath = "/sdcard/Download/side_bus.mp4"; // side
                    inVideoPath = "/sdcard/Download/side_van.mp4"; // side
//                    inVideoPath = "/sdcard/Download/side_L90.mp4"; // side
                    FRAME_STEP = 1;
                } else {
                    FRAME_STEP = 1;
                    inVideoPath = "/sdcard/Download/Video(1).mp4"; // top
                }

                System.out.println("*** " + inVideoPath);
                findViewById(R.id.radioGroupServer).setVisibility(View.GONE);
                findViewById(R.id.radioGroup).setVisibility(View.GONE);
                findViewById(R.id.IP_layout).setVisibility(View.GONE);
                findViewById(R.id.url_layout).setVisibility(View.GONE);
                findViewById(R.id.surfaceView).setVisibility(View.VISIBLE);


                Uri selectedVideoUri = data.getData();
                // Now you have the selected video URI to use in your app
                String path = null;
                try {
                    path = FilePathHelper.getPathFromUri(this, selectedVideoUri);
                } catch (Exception e) {
                    System.out.println(e);
                }
                if (path == null)
                    path = inVideoPath;
                if (new File(path).exists())
                    inVideoPath = path;

                String[] paths = inVideoPath.split("/");
                paths[paths.length - 1] = "output.mp4";
                outVideoPath = String.join("/", paths);

                paths[paths.length - 1] = "output.csv";
                outCSVPath = String.join("/", paths);


                Toast.makeText(getApplicationContext(),
                        "out_path: " + outVideoPath,
                        Toast.LENGTH_LONG).show();
                System.out.println(MainActivity.inVideoPath);
                System.out.println(outCSVPath);

                // Start updating frames periodically
                findViewById(R.id.saveBtn).setVisibility(View.VISIBLE);
                findViewById(R.id.browseBtn).setVisibility(View.GONE);
                initializeSurface();
                try {
                    startProcess();
                } catch (Exception e) {
                    System.out.println(e);
                }
            }
        }
    }

    public void init() {
        System.loadLibrary("opencv_java4");
        if (MainActivity.maxFrames == 0) {
            MainActivity.maxFrames = (int) MainActivity.cap.get(Videoio.CAP_PROP_FRAME_COUNT);
        }

        try {
            plateDetectorModel = LicensePlateDetectorFloat32.newInstance(MainActivity.this);
            plateInputFeature = TensorBuffer.createFixedSize(new int[]{1, 640, 640, 3},
                    DataType.FLOAT32);
            MainActivity.height = 640;
            MainActivity.width = 640;

            speedPredictionModel = SpeedPredictionModel.newInstance(MainActivity.this);
            topSpeedInputFeature1 = TensorBuffer.createFixedSize(new int[]{1, 3}, DataType.FLOAT32);

            speedPredictionTopViewNoPlateModel = SpeedPredictionTopViewNoPlateModel.newInstance(MainActivity.this);
            topSpeedInputFeature2 = TensorBuffer.createFixedSize(new int[]{1, 3}, DataType.FLOAT32);

            speedPredictionModelSideView = SpeedPredictionModelSideView.newInstance(MainActivity.this);
            sideSpeedInputFeature = TensorBuffer.createFixedSize(new int[]{1, 3}, DataType.FLOAT32);

            OptFlowSpeedPredictionModel = SpeedPredictionModel.newInstance(this);
            speedInputFeature = TensorBuffer.createFixedSize(new int[]{1, 212}, DataType.FLOAT32);

            tofSpeedDetector = new TOFSpeedDetector(speedInputFeature, OptFlowSpeedPredictionModel);

            topSpeedDetector = new OptSpeedDetector(topSpeedInputFeature2, speedPredictionTopViewNoPlateModel);
            sideSpeedDetector = new OptSpeedDetector(sideSpeedInputFeature, speedPredictionModelSideView);

            CPSpeedDetector = new CPSpeedDetector();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        MainActivity.winSize = new Size(10, 10);
        MainActivity.criteria = new TermCriteria(TermCriteria.COUNT + TermCriteria.EPS,
                200,
                0.001);
        MainActivity.prevGray = null;
        MainActivity.prevPts = null;
        MainActivity.frameNum = 0;
    }
}


