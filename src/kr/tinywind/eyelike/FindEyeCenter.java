package kr.tinywind.eyelike;

import org.opencv.core.*;

import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.SynchronousQueue;

import static java.lang.Math.*;
import static kr.tinywind.eyelike.Constants.*;
import static kr.tinywind.eyelike.Global.BLACK;
import static kr.tinywind.eyelike.Global.YELLOW;
import static kr.tinywind.eyelike.Helper.*;
import static org.opencv.core.Core.minMaxLoc;
import static org.opencv.core.Core.rectangle;
import static org.opencv.core.CvType.*;
import static org.opencv.highgui.Highgui.imwrite;
import static org.opencv.imgproc.Imgproc.*;

/**
 * Created by tinywind on 2016-06-04.
 */
public class FindEyeCenter {

    private Point unscalePoint(Point p, Rect origSize) {
        float ratio = (((float) kFastEyeWidth) / origSize.width);
        int x = (int) round(p.x / ratio);
        int y = (int) round(p.y / ratio);
        return new Point(x, y);
    }

    private void scaleToFastSize(Mat src, Mat dst) {
        resize(src, dst, new Size(kFastEyeWidth, (((float) kFastEyeWidth) / src.cols()) * src.rows()));
    }

    private Mat computeMatXGradient(Mat mat) {
        Mat output = new Mat(mat.rows(), mat.cols(), CvType.CV_64F);

        for (byte y = 0; y < mat.rows(); ++y) {
            Mat mr = mat.row(y);
            output.put(y, 0, mr.get(0, 1)[0] - mr.get(0, 0)[0]);
            for (byte x = 1; x < mat.cols() - 1; ++x) {
                output.put(y, x, (mr.get(0, x + 1)[0] - mr.get(0, x - 1)[0]) / 2.0);
            }
        }

        return output;
    }

    private void testPossibleCentersFormula(int x, int y, Mat weight, double gx, double gy, Mat out) {
        for (int cy = 0; cy < out.rows(); ++cy) {
            for (int cx = 0; cx < out.cols(); ++cx) {
                if (x == cx && y == cy) {
                    continue;
                }

                double dx = x - cx;
                double dy = y - cy;

                double magnitude = sqrt((dx * dx) + (dy * dy));
                dx = dx / magnitude;
                dy = dy / magnitude;
                double dotProduct = dx * gx + dy * gy;
                dotProduct = max(0.0, dotProduct);
                out.put(cy, cx, out.get(cy, cx)[0] + dotProduct * dotProduct * (kEnableWeight ? (byte) weight.get(cy, cx)[0] / kWeightDivisor : 1));
            }
        }
    }


    public Point findEyeCenter(Mat face, Rect eye, String debugWindow) {
        Mat eyeROIUnscaled = new Mat(face, eye);
        imwrite("debug_eyeROIUnscaled.png", eyeROIUnscaled);
        Mat eyeROI = new Mat();
        scaleToFastSize(eyeROIUnscaled, eyeROI);
        imwrite("debug_eyeROI.png", eyeROI);
        rectangle(face, eye.tl(), eye.br(), BLACK);
        imwrite("debug_face_findEyeCenter.png", face);
        Mat gradientX = computeMatXGradient(eyeROI);
        imwrite("debug_gradientX_init.png", gradientX);
        Mat gradientY = computeMatXGradient(eyeROI.t()).t();
        imwrite("debug_gradientY_init.png", gradientY);
        Mat mags = matrixMagnitude(gradientX, gradientY);
        imwrite("debug_mags_init.png", mags);
        double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
        for (int y = 0; y < eyeROI.rows(); ++y) {
            for (int x = 0; x < eyeROI.cols(); ++x) {
                double magnitude = mags.get(y, x)[0];
                if (magnitude > gradientThresh) {
                    gradientX.put(y, x, gradientX.get(y, x)[0] / magnitude);
                    gradientY.put(y, x, gradientY.get(y, x)[0] / magnitude);
                } else {
                    gradientX.put(y, x, 0.0);
                    gradientY.put(y, x, 0.0);
                }
            }
        }
        imwrite("debug_gradientX.png", gradientX);
        imwrite("debug_gradientX.png", gradientX);
//        new Imshow(debugWindow).showImage(gradientX);

        Mat weight = new Mat();
        GaussianBlur(eyeROI, weight, new Size(kWeightBlurSize, kWeightBlurSize), 0, 0);
        imwrite("debug_weight_init.png", weight);
        for (int y = 0; y < weight.rows(); ++y) {
            for (int x = 0; x < weight.cols(); ++x) {
                weight.put(y, x, 255 - /*(byte)*/ weight.get(y, x)[0]);
            }
        }
        imwrite("debug_weight.png", weight);

        Mat outSum = Mat.zeros(eyeROI.rows(), eyeROI.cols(), CV_64F);
        System.out.println("Eye Size: " + outSum.cols() + ", " + outSum.rows());
        for (int y = 0; y < weight.rows(); ++y) {
            for (int x = 0; x < weight.cols(); ++x) {
                double gX = gradientX.get(y, x)[0], gY = gradientY.get(y, x)[0];
                if (gX == 0.0 && gY == 0.0) {
                    continue;
                }
                testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
            }
        }

        double numGradients = (weight.rows() * weight.cols());
        Mat out = new Mat();
        outSum.convertTo(out, CV_32F, 1.0 / numGradients);
        imwrite("debug_out_findeyecenter_init.png", out);

        Core.MinMaxLocResult minMaxLocResult = minMaxLoc(out);
        double maxVal = minMaxLocResult.maxVal;
        Point maxP = minMaxLocResult.maxLoc;
        //-- Flood fill the edges
        if (kEnablePostProcess) {
            Mat floodClone = new Mat();
            double floodThresh = maxVal * kPostProcessThreshold;
            threshold(out, floodClone, floodThresh, 0.0f, THRESH_TOZERO);
            // if (kPlotVectorField)  plotVecField(gradientX, gradientY, floodClone);
            Mat mask = floodKillEdges(floodClone);
            imwrite("debug_mask.png", mask);
            maxP = minMaxLoc(out, mask).maxLoc;
        }
        return unscalePoint(maxP, eye);
    }


    private boolean floodShouldPushPoint(Point np, Mat mat) {
        return inMat(np, mat.rows(), mat.cols());
    }


    private Mat floodKillEdges(Mat floodClone) {
        Rect rect = new Rect(0, 0, floodClone.cols(), floodClone.rows());
        rectangle(floodClone, rect.tl(), rect.br(), YELLOW);
        imwrite("debug_floodClone.png", floodClone);

        Mat mask = new Mat(floodClone.rows(), floodClone.cols(), CV_8U, YELLOW);
        Queue<Point> toDo = new LinkedList<>();
        toDo.add(new Point(0, 0));
        while (!toDo.isEmpty()) {
            Point p = toDo.remove();
            if (floodClone.get((int) p.y, (int) p.x)[0] == 0.0) {
                continue;
            }
            Point np = new Point(p.x + 1, p.y);
            if (floodShouldPushPoint(np, floodClone)) toDo.add(np);
            np.x = p.x - 1;
            np.y = p.y;
            if (floodShouldPushPoint(np, floodClone)) toDo.add(np);
            np.x = p.x;
            np.y = p.y + 1;
            if (floodShouldPushPoint(np, floodClone)) toDo.add(np);
            np.x = p.x;
            np.y = p.y - 1;
            if (floodShouldPushPoint(np, floodClone)) toDo.add(np);
            floodClone.put((int) p.y, (int) p.x, 0);
            mask.put((int) p.y, (int) p.x, 0);
        }
        return mask;
    }
}
