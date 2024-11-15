package com.example.myapplication;

import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.MotionEvent;
import android.view.View;

import com.google.mlkit.vision.GraphicOverlay;

import org.opencv.core.Point;

public class RoadLine {
    public static float globalCoeff = 9000f;
    private static float viewW, viewH;
    private final Paint linePaint = new Paint();
    private final RectF inside = new RectF(0.1f, 0.1f, 0.9f, 0.9f);
    GraphicOverlay overlay;
    private Point point1, point2, point3, point4;
    private Point P1, P2, normalLineVector;
    private double W1;
    private double W2;
    private View circle1, circle2, circle3, circle4;
    private Matrix normToViewTransform;

    public RoadLine(GraphicOverlay overlay) {
        this.overlay = overlay;
    }

    public static void setCirclesTop(View circle1, View circle2, View circle3, View circle4) {
        circle1.setX(652f / 2102 * viewW);
        circle1.setY(745f / 1266 * viewH);

        circle2.setX(1171f / 2102 * viewW);
        circle2.setY(712f / 1266 * viewH);

        circle3.setX(598f / 2102 * viewW);
        circle3.setY(1544f / 1266 * viewH);

        circle4.setX(1486f / 2102 * viewW);
        circle4.setY(1580f / 1266 * viewH);

    }

    public void setCirclesPH(View circle1, View circle2, View circle3, View circle4) {
        int offsetX = (int) overlay.getX();
        int offsetY = (int) overlay.getY();

        circle1.setX(0f / 200 * viewW + offsetX);
        circle1.setY(0f / 200 * viewH + offsetY);

        circle2.setX(100f / 200 * viewW + offsetX);
        circle2.setY(0f / 200 * viewH + offsetY);

        circle3.setX(100f / 200 * viewW + offsetX);
        circle3.setY(100f / 200 * viewH + offsetY);

        circle4.setX(0f / 200 * viewW + offsetX);
        circle4.setY(100f / 200 * viewH + offsetY);

    }

    public static void setCirclesSide1(View circle1, View circle2, View circle3, View circle4) {
        circle1.setX(1653f / 2102 * viewW);
        circle1.setY(1218f / 1266 * viewH);

        circle2.setX(1764f / 2102 * viewW);
        circle2.setY(1331f / 1266 * viewH);

        circle3.setX(105f / 2102 * viewW);
        circle3.setY(1416f / 1266 * viewH);

        circle4.setX(380f / 2102 * viewW);
        circle4.setY(1616f / 1266 * viewH);

    }

    public static void setCirclesSide2(View circle1, View circle2, View circle3, View circle4) {
        circle1.setX(1417f / 2102 * viewW);
        circle1.setY(1074f / 1266 * viewH);

        circle2.setX(1820f / 2102 * viewW);
        circle2.setY(1100f / 1266 * viewH);

        circle3.setX(203f / 2102 * viewW);
        circle3.setY(1282f / 1266 * viewH);

        circle4.setX(747f / 2102 * viewW);
        circle4.setY(1552f / 1266 * viewH);

    }

    public void initializeCircles(View circle1, View circle2, View circle3, View circle4) {
        linePaint.setColor(Color.GREEN);
        linePaint.setStrokeWidth(5);

        this.circle1 = circle1;
        this.circle2 = circle2;
        this.circle3 = circle3;
        this.circle4 = circle4;

        viewW = overlay.getWidth();
        viewH = overlay.getHeight();

        setCirclesTop(circle1, circle2, circle3, circle4);
//        setCirclesSide1(circle1, circle2, circle3, circle4);
//        setCirclesSide2(circle1, circle2, circle3, circle4);

        updateParameters();

//        setVisible(View.VISIBLE); // todo: uncomment
        setVisible(View.GONE);
    }

    public void updateParameters() {
        setPoints();
    }

    void setVisible(int visible) {
        circle1.setVisibility(visible);
        circle2.setVisibility(visible);
        circle3.setVisibility(visible);
        circle4.setVisibility(visible);
    }

    private void setPoints() {
        // TODO: 16.04.24
        float w = overlay.getWidth();
        float h = overlay.getHeight();
        normToViewTransform = ImageUtils.getTransformationMatrix(
                1,
                1,
                (int) w,
                (int) h,
                0,
                false
        );

        int offsetX = (int) overlay.getX() - circle1.getWidth() / 2;
        int offsetY = (int) overlay.getY() - circle1.getHeight() / 2;
        float x1 = circle1.getX() - offsetX;
        float x2 = circle2.getX() - offsetX;
        float x3 = circle3.getX() - offsetX;
        float x4 = circle4.getX() - offsetX;

        float y1 = circle1.getY() - offsetY;
        float y2 = circle2.getY() - offsetY;
        float y3 = circle3.getY() - offsetY;
        float y4 = circle4.getY() - offsetY;

        float dx1 = x1 - x3;
        float dx2 = x2 - x4;
        float dy1 = y3 - y1;
        float dy2 = y4 - y2;

        float Y1 = dy1 > 0 ? h - y1 : y1;
        float Y2 = dy2 > 0 ? h - y2 : y2;
        float Y3 = dy1 > 0 ? y3 : h - y3;
        float Y4 = dy2 > 0 ? y4 : h - y4;
        float X1 = dx1 > 0 ? x1 : w - x1;
        float X2 = dx2 > 0 ? x2 : w - x2;
        float X3 = dx1 > 0 ? w - x3 : x3;
        float X4 = dx2 > 0 ? w - x4 : x4;

        point1 = new Point(
                max(min(x3 + dx1 * Y3 / abs(dy1), w - 1), 0),
                min(max(y3 - dy1 * (X3) / abs(dx1), 0), h)
        );
        point2 = new Point(
                max(min(x4 + dx2 * Y4 / abs(dy2), w - 1), 0),
                min(max(y4 - (dy2 * (X4) / abs(dx2)), 0), h)
        );
        point3 = new Point(
                min(max(x1 - dx1 * (Y1) / abs(dy1), 0), w),
                max(min(y1 + dy1 * X1 / abs(dx1), h - 1), 0)
        );
        point4 = new Point(
                min(max(x2 - dx2 * (Y2) / abs(dy2), 0), w),
                max(min(y2 + dy2 * X2 / abs(dx2), h - 1), 0)
        );
        P1 = new Point((point1.x + point2.x) / 2, (point1.y + point2.y) / 2);
        P2 = new Point((point3.x + point4.x) / 2, (point3.y + point4.y) / 2);
        W1 = pow(d(point1, point2), 1);
        W2 = pow(d(point3, point4), 1);
        Point lineVector = new Point(P1.x - P2.x, P1.y - P2.y);
        normalLineVector = new Point(lineVector.x / norm(lineVector), lineVector.y / norm(lineVector));
    }

    public void movePoint(View circle, MotionEvent event) {
        if (event.getRawX() < overlay.getX()
                || event.getRawY() < overlay.getY()
                || event.getRawX() > overlay.getX() + overlay.getWidth()
                || event.getRawY() > overlay.getY() + overlay.getHeight()
        )
            return;
        circle.setX((int) event.getRawX() - (float) circle.getWidth() / 2);
        circle.setY((int) event.getRawY() - (float) circle.getHeight() / 2);
    }

    private double calculateLocalCoefficient(Point point) {
        double d1 = pow(d(point, P1), 4.5);
        double d2 = pow(d(point, P2), 4.5);
        double tmp = ((W1 * d2 + W2 * d1) / (d1 + d2) / max(W1, W2));
//        tmp /= overlay.getHeight() / abs(P1.y - P2.y);
        return 1 / (tmp);
    }
    public Point calculateSignSpeed(Point pN1, Point pN2, float frames) {
        float[] pts = {(float) pN1.x, (float) pN1.y};

        double dx = (pN2.x - pN1.x);
        double dy = (pN2.y - pN1.y);

        normToViewTransform.mapPoints(pts);

        double coef = calculateLocalCoefficient(new Point(pts[0], pts[1]));

        Point speed = new Point((coef * globalCoeff * dx / frames), (coef * globalCoeff * (dy / frames)));
        double cosine = abs(speed.dot(normalLineVector) / sqrt(norm2(speed)));

        return new Point(speed.x * cosine, speed.y * cosine);
    }

    public Point calculateSignSpeed(Point pN1, Point pN2) {
        float[] pts = {(float) (pN1.x*0.4 + pN2.x*0.6)/2, (float) (pN1.y*0.6 + pN2.y*0.4)/2};
//        float[] pts = {(float) pN1.x, (float) pN1.y};

        double dx = (pN2.x - pN1.x);
        double dy = (pN2.y - pN1.y);

        normToViewTransform.mapPoints(pts);

        double coef = calculateLocalCoefficient(new Point(pts[0], pts[1]));

        Point speed = new Point((coef * globalCoeff * dx), (coef * globalCoeff * (dy)));
        double cosine = abs(speed.dot(normalLineVector) / sqrt(norm2(speed)));

        return new Point(speed.x * cosine, speed.y * cosine);
    }

    public Point calculateSignSpeed(RectF rect1, RectF rect2, float frames) {
        RectF unionR = new RectF(
                min(rect1.left, rect2.left),
                min(rect1.top, rect2.top),
                max(rect1.right, rect2.right),
                max(rect1.bottom, rect2.bottom)
        );

        if (inside.contains(unionR))
            return calculateSignSpeed(
                    new Point(rect1.centerX(), rect1.centerY()),
                    new Point(rect2.centerX(), rect2.centerY()),
                    frames
            );
        else if (inside.contains(unionR.left, unionR.top))
            return calculateSignSpeed(
                    new Point(rect1.left, rect1.top),
                    new Point(rect2.left, rect2.top),
                    frames
            );
        else if (inside.contains(unionR.right, unionR.bottom))
            return calculateSignSpeed(
                    new Point(rect1.right, rect1.bottom),
                    new Point(rect2.right, rect2.bottom),
                    frames
            );
        else if (inside.contains(unionR.left, unionR.bottom))
            return calculateSignSpeed(
                    new Point(rect1.left, rect1.bottom),
                    new Point(rect2.left, rect2.bottom),
                    frames
            );
        else if (inside.contains(unionR.right, unionR.top))
            return calculateSignSpeed(
                    new Point(rect1.right, rect1.top),
                    new Point(rect2.right, rect2.top),
                    frames
            );
        return new Point(0, 0);
    }

    private double norm(Point p) {
        return sqrt(pow(p.x, 2) + pow(p.y, 2));
    }

    private double norm2(Point p) {
        return pow(p.x, 2) + pow(p.y, 2);
    }

    private double d(Point p1, Point p2) {
        return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
    }


    void drawLine(Canvas canvas, float x1, float y1, float x2, float y2, int offsetX, int offsetY, float sx, float sy) {
        x1 = (x1 - offsetX) * sx;
        y1 = (y1 - offsetY) * sy;
        x2 = (x2 - offsetX) * sx;
        y2 = (y2 - offsetY) * sy;
        float dx = x2 - x1;
        float dy = y2 - y1;

        int w = canvas.getWidth();
        int h = canvas.getHeight();

        float o, s, X1, X2, Y1, Y2;

        o = x1;
        if (y1 < y2)
            s = (h-y1)/dy*dx;
        else
            s = -y1/dy*dx;
        X1 = min(max(o + s, 0), w);

        o = x2;
        if (y1 > y2)
            s = (h-y2)/dy*dx;
        else
            s = -y2/dy*dx;
        X2 = min(max(o + s, 0), w);

        o = y1;
        if (x1 < x2)
            s = (w-x1)/dx*dy;
        else
            s = -x1/dx*dy;
        Y1 = min(max(o + s, 0), h);

        o = y2;
        if (x1 > x2)
            s = (w-x2)/dx*dy;
        else
            s = -x2/dx*dy;
        Y2 = min(max(o + s, 0), h);

        canvas.drawLine(X1, Y1, X2, Y2, linePaint);
    }
    public void drawLines(Canvas canvas, int offsetX, int offsetY, float sx, float sy) {
        drawLine(canvas, circle1.getX(), circle1.getY(), circle2.getX(), circle2.getY(), offsetX, offsetY, sx, sy);
        drawLine(canvas, circle3.getX(), circle3.getY(), circle2.getX(), circle2.getY(), offsetX, offsetY, sx, sy);
        drawLine(canvas, circle3.getX(), circle3.getY(), circle4.getX(), circle4.getY(), offsetX, offsetY, sx, sy);
        drawLine(canvas, circle1.getX(), circle1.getY(), circle4.getX(), circle4.getY(), offsetX, offsetY, sx, sy);
    }
    public void drawLines(Canvas canvas) {
        if (circle1 == null)
            return;
        updateParameters();

        double sx = (double) overlay.getImageWidth() / overlay.getWidth();
        double sy = (double) overlay.getImageHeight() / overlay.getHeight();

        canvas.drawLine((int) (point3.x * sx), (int) (point3.y * sy), (int) (point1.x * sx), (int) (point1.y * sy), linePaint);
        canvas.drawLine((int) (point2.x * sx), (int) (point2.y * sy), (int) (point4.x * sx), (int) (point4.y * sy), linePaint);
        canvas.drawCircle((float) (P1.x * sx), (float) (P1.y * sy), 15, linePaint);
        canvas.drawCircle((float) (P2.x * sx), (float) (P2.y * sy), 15, linePaint);
    }
    public void drawRoadLines(Canvas canvas, int offsetX, int offsetY) {
        if (circle1 == null)
            return;
        updateParameters();

        float sx = 1;
        float sy = 1;
        drawLine(canvas, circle1.getX(), circle1.getY(), circle3.getX(), circle3.getY(), offsetX, offsetY, sx, sy);
        drawLine(canvas, circle4.getX(), circle4.getY(), circle2.getX(), circle2.getY(), offsetX, offsetY, sx, sy);
//        canvas.drawLine((float) (point3.x * sx), (float) (point3.y * sy), (float) (point1.x * sx), (float) (point1.y * sy), linePaint);
//        canvas.drawLine((float) (point2.x * sx), (float) (point2.y * sy), (float) (point4.x * sx), (float) (point4.y * sy), linePaint);
        canvas.drawCircle((float) (P1.x * sx), (float) (P1.y * sy), 15, linePaint);
        canvas.drawCircle((float) (P2.x * sx), (float) (P2.y * sy), 15, linePaint);
    }
}