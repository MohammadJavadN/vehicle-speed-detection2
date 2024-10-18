package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.objects.DetectedObject;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class ServerSpeedDetector {
    private static final String TAG = "ServerSpeedDetector";
    public final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(0, TimeUnit.SECONDS)
            .writeTimeout(0, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.SECONDS)
            .build();
    private final String SERVER_URL; // = "http://192.168.43.226:5000/";
    private final String PROCESS_URL;
    protected Bitmap bitmap;
    public ServerSpeedDetector(String url, boolean withSpeed) {
        SERVER_URL = url;
        if (withSpeed)
            PROCESS_URL = SERVER_URL + "process_frame3";
        else
            PROCESS_URL = SERVER_URL + "process_frame";

        initializeServerModel();
    }

    // Parse JSON response manually and convert it to a List of DetectedObjects
    public static List<DetectedObject> parseDetectedObjects(String jsonResponse) {
        Gson gson = new Gson();
        Type listType = new TypeToken<List<ServerDetectedObject>>() {
        }.getType();
        List<ServerDetectedObject> serverObjects = gson.fromJson(jsonResponse, listType);

        List<DetectedObject> detectedObjects = new ArrayList<>();

        for (ServerDetectedObject obj : serverObjects) {
            // Map the bbox array to Rect object
            Rect bbox = new Rect(obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3]);

            // Tracking id from server `id` or you can leave null
            Integer trackingId = obj.id;

            // Labels are empty for now, you can adjust if the server starts sending them
            List<DetectedObject.Label> labels = new ArrayList<>();
            labels.add(new DetectedObject.Label(Float.toString(obj.speed), 1, 0));

            // Construct the DetectedObject
            DetectedObject detectedObject = new DetectedObject(bbox, trackingId, labels);
            detectedObjects.add(detectedObject);
        }

        return detectedObjects;
    }

    // Function to send a POST request to initialize the server model
    private void initializeServerModel() {
        OkHttpClient client = new OkHttpClient();

        // Create an empty request body as we don't need to send any data to the server in this case
        RequestBody requestBody = RequestBody.create(
                "", MediaType.parse("application/json"));

        // Build the request
        Request request = new Request.Builder()
                .url(SERVER_URL + "init")
                .post(requestBody)  // Sending POST request
                .build();

        // Make the asynchronous request
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                // Log or handle the error
                Log.e(TAG, "Failed to initialize the server", e);
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    // Server initialization was successful
                    Log.d(TAG, "Server model initialized successfully");
                } else {
                    // Handle the case where initialization failed
                    Log.e(TAG, "Server initialization failed: " + response.message());
                    Log.e(TAG, "Server initialization failed: " + response.code() + " " + response.message());
                }
            }
        });
    }

    public Request sendFrameToServer(Bitmap frameBitmap) {
        // Convert Bitmap to byte array (JPEG format)
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        frameBitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
        byte[] frameData = stream.toByteArray();

        // Prepare the request to send the frame to the server
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("frame", "frame.jpg",
                        RequestBody.create(MediaType.parse("image/jpeg"), frameData))
                .build();

        // Create request object
        Request request = new Request.Builder()
                .url(PROCESS_URL)
                .post(requestBody)
                .build();

        // Send the request
        return request;
    }

    private byte[] matToByteArray(Mat frame) {
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", frame, matOfByte);
        return matOfByte.toArray();
    }

    private byte[] sendFrameToServer(byte[] frameData) throws IOException {
        // Set up the HTTP request
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("frame", "frame.jpg",
                        RequestBody.create(frameData, MediaType.parse("image/jpeg")))
                .build();

        System.out.println("*** request body created");
        Request request = new Request.Builder()
//                .url("http://192.168.43.226:5000/process_video")
                .url("http://10.42.0.191:5000/process_video")
                .post(requestBody)
                .build();
        System.out.println("*** request builder built");

        // Send request and get response
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("Unexpected code " + response);
            }

            assert response.body() != null;
            return response.body().bytes();
        }
    }

    public class ServerDetectedObject {
        public int id;
        public int[] bbox;  // bbox[0]: left, bbox[1]: top, bbox[2]: right, bbox[3]: bottom
        public float speed;
    }
}
