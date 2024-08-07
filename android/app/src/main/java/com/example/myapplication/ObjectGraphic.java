/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
//import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.GraphicOverlay.Graphic;
import com.google.mlkit.vision.GraphicOverlay;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.DetectedObject.Label;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.util.Locale;

/**
 * Draw the detected object info in preview.
 */
public class ObjectGraphic extends Graphic {

  private static final float TEXT_SIZE = 30.0f;
  private static final float STROKE_WIDTH = 4.0f;
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
  private static final String LABEL_FORMAT = "%.2f%% confidence (index: %d)";

  private final MyDetectedObject object;
  private final Paint[] boxPaints;
  private final Paint[] textPaints;
  private final Paint[] labelPaints;

  public ObjectGraphic(GraphicOverlay overlay, MyDetectedObject object) {
    super(overlay);

    this.object = object;

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

  @Override
  public void draw(Canvas canvas) {
    // Decide color based on object tracking ID
    int colorID =
            object.getTrackingId() == null ? 0 : Math.abs(object.getTrackingId() % NUM_COLORS);
    float textWidth = textPaints[colorID].measureText("ID: " + object.getTrackingId() + ", Speed: " + ((int) object.getSpeed()));
    float lineHeight = TEXT_SIZE + STROKE_WIDTH;
    float yLabelOffset = lineHeight;

    // Calculate width and height of label box
    for (Label label : object.getLabels()) {
      textWidth = Math.max(textWidth, textPaints[colorID].measureText(label.getText()));
      textWidth =
              Math.max(
                      textWidth,
                      textPaints[colorID].measureText(
                              String.format(
                                      Locale.US, LABEL_FORMAT, label.getConfidence() * 100, label.getIndex())));
      yLabelOffset += 2 * lineHeight;
    }

//    yLabelOffset *= -1;

    RectF rect = new RectF(object.getBoundingBox());

    float x0 = rect.left;
    float x1 = rect.right;
    rect.left = Math.min(x0, x1);
    rect.right = Math.max(x0, x1);

    canvas.drawRect(rect, boxPaints[colorID]);

    // Draws other object info.
    canvas.drawRect(
            rect.left - STROKE_WIDTH*0,
            rect.top,
            rect.left + textWidth + (2 * STROKE_WIDTH),
            rect.top + yLabelOffset*1.4f,
            labelPaints[colorID]);
//    yLabelOffset += TEXT_SIZE;
    canvas.drawText(
            "ID: " + object.getTrackingId() + ", Speed: " + ((int) object.getSpeed()),
            rect.left,
            rect.top + yLabelOffset,
            textPaints[colorID]);
    yLabelOffset -= lineHeight;

//    for (Label label : object.getLabels()) {
//      canvas.drawText(label.getText(), rect.left, rect.top + yLabelOffset, textPaints[colorID]);
//      yLabelOffset -= lineHeight;
//      canvas.drawText(
//              String.format(Locale.US, LABEL_FORMAT, label.getConfidence() * 100, label.getIndex()),
//              rect.left,
//              rect.top + yLabelOffset,
//              textPaints[colorID]);
//
//      yLabelOffset -= lineHeight;
//    }
  }
}