package kr.tinywind.opencv;

import org.opencv.core.*;
import org.opencv.objdetect.CascadeClassifier;

import java.io.IOException;
import java.util.*;

import static java.lang.Math.max;
import static java.lang.Math.pow;
import static java.lang.Math.*;
import static java.lang.Math.sqrt;
import static kr.tinywind.opencv.Color.*;
import static org.opencv.core.Core.*;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.highgui.Highgui.imread;
import static org.opencv.highgui.Highgui.imwrite;
import static org.opencv.imgproc.Imgproc.*;

public class EyeFinder {
    private final static CascadeClassifier faceCascade;
    private final static Boolean usable;

    static {
        System.load(System.getProperty("opencv.lib"));
        faceCascade = new CascadeClassifier();
        usable = faceCascade.load(System.getProperty("opencv.cascade.face"));
    }

    private final Options options = new Options();
    private Mat image;

    public EyeFinder(String imagePath) {
        if (!usable)
            throw new IllegalArgumentException("Error: loading face cascade, please change face_cascade_name in source code.");
        image = imread(imagePath);
        if (image.empty())
            throw new IllegalArgumentException("Can't read image: " + imagePath);
    }

    private static Mat matrixMagnitude(Mat matX, Mat matY) {
        Mat mags = new Mat(matX.rows(), matX.cols(), CV_64F);
        for (int y = 0; y < matX.rows(); ++y) {
            for (int x = 0; x < matX.cols(); ++x) {
                double gX = matX.get(y, x)[0];
                double gY = matY.get(y, x)[0];
                double magnitude = sqrt((gX * gX) + (gY * gY));
                mags.put(y, x, magnitude);
            }
        }
        return mags;
    }

    private static boolean inMat(Point p, int rows, int cols) {
        return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
    }

    private static double computeDynamicThreshold(Mat mat, int channel, double stdDevFactor) {
        long numElements = 0;
        double meanValue = 0;
        for (int y = 0; y < mat.rows(); y++) {
            for (int x = 0; x < mat.cols(); x++) {
                double v = mat.get(y, x)[channel];
                if (0.0 <= v && v <= 255.0) {
                    meanValue += v;
                    numElements++;
                }
            }
        }
        meanValue /= numElements;
        double standardDeviation = 0;
        for (int y = 0; y < mat.rows(); y++) {
            for (int x = 0; x < mat.cols(); x++) {
                double v = mat.get(y, x)[channel];
                if (0.0 <= v && v <= 255.0) {
                    standardDeviation += pow(meanValue - v, 2) / (numElements - 1);
                }
            }
        }
        standardDeviation = sqrt(standardDeviation) / sqrt(numElements);
        return stdDevFactor * standardDeviation + meanValue;
        /** incorrect work: meanStdDev(mat, mean, stddev);
         MatOfDouble mean = new MatOfDouble();
         MatOfDouble stddev = new MatOfDouble();
         meanStdDev(mat, mean, stddev);
         double stdDev = stddev.get(0, 0)[0] / sqrt(mat.rows() * mat.cols());
         return stdDevFactor * stdDev + mean.get(0, 0)[0];
         */
    }

    private static Mat computeMatXGradient(Mat mat) {
        final Mat output = new Mat(mat.rows(), mat.cols(), CV_64F);

        for (byte y = 0; y < mat.rows(); ++y) {
            final Mat mr = mat.row(y);
            output.put(y, 0, mr.get(0, 1)[0] - mr.get(0, 0)[0]);
            for (byte x = 1; x < mat.cols() - 1; ++x) {
                output.put(y, x, (mr.get(0, x + 1)[0] - mr.get(0, x - 1)[0]) / 2.0);
            }
        }

        return output;
    }

    private static boolean floodShouldPushPoint(Point np, Mat mat) {
        return inMat(np, mat.rows(), mat.cols());
    }

    private static Mat floodKillEdges(Mat floodClone) {
        final Rect rect = new Rect(0, 0, floodClone.cols(), floodClone.rows());
        rectangle(floodClone, rect.tl(), rect.br(), YELLOW);

        final Mat mask = new Mat(floodClone.rows(), floodClone.cols(), CV_8U, YELLOW);
        final Queue<Point> toDo = new LinkedList<>();
        toDo.add(new Point(0, 0));
        while (!toDo.isEmpty()) {
            final Point p = toDo.remove();
            if (floodClone.get((int) p.y, (int) p.x)[0] == 0.0) {
                continue;
            }
            final Point np = new Point(p.x + 1, p.y);
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

    private static Mat eyeCornerMap(Mat region, boolean left, boolean left2) {
        final Mat rightCornerKernel = new Mat(4, 6, CV_32F, YELLOW);
        final Mat leftCornerKernel = new Mat(4, 6, CV_32F);
        flip(rightCornerKernel, leftCornerKernel, 1);

        final Mat cornerMap = new Mat();
        final Size sizeRegion = region.size();
        final Range colRange = new Range((int) (sizeRegion.width / 4), (int) (sizeRegion.width * 3 / 4));
        final Range rowRange = new Range((int) (sizeRegion.height / 4), (int) (sizeRegion.height * 3 / 4));
        final Mat miRegion = new Mat(region, rowRange, colRange);
        filter2D(miRegion, cornerMap, CV_32F, (left && !left2) || (!left && !left2) ? leftCornerKernel : rightCornerKernel);
        return cornerMap;
    }

    private static Point findEyeCorner(Mat region, boolean left, boolean left2) {
        return findSubPixelEyeCorner(eyeCornerMap(region, left, left2));
    }

    private static Point findSubPixelEyeCorner(Mat region) {
        final Size sizeRegion = region.size();
        final Mat cornerMap = new Mat((int) sizeRegion.height * 10, (int) sizeRegion.width * 10, CV_32F);
        resize(region, cornerMap, cornerMap.size(), 0, 0, INTER_CUBIC);
        final Point maxP2 = minMaxLoc(cornerMap).maxLoc;
        return new Point(sizeRegion.width / 2 + maxP2.x / 10, sizeRegion.height / 2 + maxP2.y / 10);
    }

    private static Mat getDebugImage(Mat image, Result result) {
        final Mat debugImage = new Mat();
        image.copyTo(debugImage);

        result.faces.forEach(e -> {
            rectangle(debugImage, e.faceArea.tl(), e.faceArea.br(), GREEN);
            for (Eye eye : (new Eye[]{e.leftEye, e.rightEye})) {
                circle(debugImage, new Point(e.faceArea.x + eye.pupil.x, e.faceArea.y + eye.pupil.y), 2, GREEN);
                circle(debugImage, new Point(e.faceArea.x + eye.leftCorner.x, e.faceArea.y + eye.leftCorner.y), 2, BLUE);
                circle(debugImage, new Point(e.faceArea.x + eye.rightCorner.x, e.faceArea.y + eye.rightCorner.y), 2, BLUE);
                rectangle(debugImage,
                        new Point(e.faceArea.x + eye.leftCornerRegion.x, e.faceArea.y + eye.leftCornerRegion.y),
                        new Point(e.faceArea.x + eye.leftCornerRegion.x + eye.leftCornerRegion.width,
                                e.faceArea.y + eye.leftCornerRegion.y + eye.leftCornerRegion.height),
                        RED);
                rectangle(debugImage,
                        new Point(e.faceArea.x + eye.rightCornerRegion.x, e.faceArea.y + eye.rightCornerRegion.y),
                        new Point(e.faceArea.x + eye.rightCornerRegion.x + eye.rightCornerRegion.width,
                                e.faceArea.y + eye.rightCornerRegion.y + eye.rightCornerRegion.height),
                        RED);
            }
        });
        return debugImage;
    }

    public Mat getDebugImage() {
        return getDebugImage(image, detectAndDisplay(image));
    }

    public Result result() throws IOException {
        return detectAndDisplay(image);
    }

    private Result detectAndDisplay(Mat image) {
        final Result result = new Result();
        final Vector<Mat> rgbChannels = new Vector<>(image.channels());
        split(image, rgbChannels);
        final Mat grayChannel = new Mat();
        rgbChannels.lastElement().copyTo(grayChannel);

        final int width = image.cols();
        final int height = image.rows();
        final int faceMinSideLength = Integer.min(width, height) / 3;
        final int faceMaxSideLength = (int) (Integer.max(width, height) * 1.5);

        final MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(grayChannel, faces, 1.1, 2, 0,
                new Size(faceMinSideLength, faceMinSideLength),
                new Size(faceMaxSideLength, faceMaxSideLength));

        faces.toList().forEach(e -> result.add(findEyes(grayChannel, e)));

        return result;
    }

    private Face findEyes(Mat grayChannel, Rect faceArea) {
        final Mat faceROI = new Mat(grayChannel, faceArea);
        if (options.kSmoothFaceImage) {
            double sigma = options.kSmoothFaceFactor * faceArea.width;
            GaussianBlur(faceROI, faceROI, new Size(0, 0), sigma);
        }

        final int eyeRegionWidth = (int) (faceArea.width * (options.kEyePercentWidth / 100.0));
        final int eyeRegionHeight = (int) (faceArea.width * (options.kEyePercentHeight / 100.0));
        final int eyeRegionTop = (int) (faceArea.height * (options.kEyePercentTop / 100.0));
        final Rect leftEyeRegion = new Rect((int) (faceArea.width * (options.kEyePercentSide / 100.0)), eyeRegionTop, eyeRegionWidth, eyeRegionHeight);
        final Rect rightEyeRegion = new Rect((int) (faceArea.width - eyeRegionWidth - faceArea.width * (options.kEyePercentSide / 100.0)), eyeRegionTop, eyeRegionWidth, eyeRegionHeight);

        imwrite("debugImage_faceROI.png", faceROI);
        final Point leftPupil = findEyeCenter(faceROI, leftEyeRegion);
        final Point rightPupil = findEyeCenter(faceROI, rightEyeRegion);

        final Rect leftRightCornerRegion = leftEyeRegion.clone();
        leftRightCornerRegion.width -= leftPupil.x;
        leftRightCornerRegion.x += leftPupil.x;
        leftRightCornerRegion.height /= 2;
        leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
        final Rect leftLeftCornerRegion = leftEyeRegion.clone();
        leftLeftCornerRegion.width = (int) leftPupil.x;
        leftLeftCornerRegion.height /= 2;
        leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
        final Rect rightLeftCornerRegion = rightEyeRegion.clone();
        rightLeftCornerRegion.width = (int) rightPupil.x;
        rightLeftCornerRegion.height /= 2;
        rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
        final Rect rightRightCornerRegion = rightEyeRegion.clone();
        rightRightCornerRegion.width -= rightPupil.x;
        rightRightCornerRegion.x += rightPupil.x;
        rightRightCornerRegion.height /= 2;
        rightRightCornerRegion.y += rightRightCornerRegion.height / 2;

        // change eye centers to face coordinates
        rightPupil.x += rightEyeRegion.x;
        rightPupil.y += rightEyeRegion.y;
        leftPupil.x += leftEyeRegion.x;
        leftPupil.y += leftEyeRegion.y;

        final Point leftRightCorner = findEyeCorner(new Mat(faceROI, leftRightCornerRegion), true, false);
        leftRightCorner.x += leftRightCornerRegion.x;
        leftRightCorner.y += leftRightCornerRegion.y;
        final Point leftLeftCorner = findEyeCorner(new Mat(faceROI, leftLeftCornerRegion), true, true);
        leftLeftCorner.x += leftLeftCornerRegion.x;
        leftLeftCorner.y += leftLeftCornerRegion.y;
        final Point rightLeftCorner = findEyeCorner(new Mat(faceROI, rightLeftCornerRegion), false, true);
        rightLeftCorner.x += rightLeftCornerRegion.x;
        rightLeftCorner.y += rightLeftCornerRegion.y;
        final Point rightRightCorner = findEyeCorner(new Mat(faceROI, rightRightCornerRegion), false, false);
        rightRightCorner.x += rightRightCornerRegion.x;
        rightRightCorner.y += rightRightCornerRegion.y;

        final Eye leftEye = new Eye(leftPupil, leftLeftCornerRegion, leftRightCornerRegion, leftLeftCorner, leftRightCorner);
        final Eye rightEye = new Eye(rightPupil, rightLeftCornerRegion, rightRightCornerRegion, rightLeftCorner, rightRightCorner);
        return new Face(faceArea, leftEye, rightEye);
    }

    private Point findEyeCenter(Mat face, Rect eye) {
        final Mat eyeROIUnscaled = new Mat(face, eye);
        final Mat eyeROI = new Mat();
        scaleToFastSize(eyeROIUnscaled, eyeROI);
        imwrite("eyeROIUnscaled.png", eyeROIUnscaled);
        imwrite("eyeROI.png", eyeROI);
        rectangle(face, eye.tl(), eye.br(), BLACK);
        final Mat gradientX = computeMatXGradient(eyeROI);
        final Mat gradientY = computeMatXGradient(eyeROI.t()).t();
        final Mat mags = matrixMagnitude(gradientX, gradientY);
        imwrite("mags.png", mags);
        final double gradientThresh = computeDynamicThreshold(mags, 0, options.kGradientThreshold);
        for (int y = 0; y < eyeROI.rows(); ++y) {
            for (int x = 0; x < eyeROI.cols(); ++x) {
                final double magnitude = mags.get(y, x)[0];
                if (magnitude > gradientThresh) {
                    gradientX.put(y, x, gradientX.get(y, x)[0] / magnitude);
                    gradientY.put(y, x, gradientY.get(y, x)[0] / magnitude);
                } else {
                    gradientX.put(y, x, 0.0);
                    gradientY.put(y, x, 0.0);
                }
            }
        }

        imwrite("gradientX.png", gradientX);
        imwrite("gradientY.png", gradientY);
        final Mat weight = new Mat();
        GaussianBlur(eyeROI, weight, new Size(options.kWeightBlurSize, options.kWeightBlurSize), 0, 0);
        imwrite("eyeROI.png", eyeROI);
        imwrite("weight.png", weight);
        for (int y = 0; y < weight.rows(); ++y) {
            for (int x = 0; x < weight.cols(); ++x) {
                weight.put(y, x, 255 - /*(byte)*/ weight.get(y, x)[0]);
            }
        }
        imwrite("weight2.png", weight);

        final Mat outSum = Mat.zeros(eyeROI.rows(), eyeROI.cols(), CV_64F);
        System.out.println("Eye Size: " + outSum.cols() + ", " + outSum.rows());
        for (int y = 0; y < weight.rows(); ++y) {
            for (int x = 0; x < weight.cols(); ++x) {
                final double gX = gradientX.get(y, x)[0];
                final double gY = gradientY.get(y, x)[0];
                if (gX == 0.0 && gY == 0.0)
                    continue;
                testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
            }
        }

        final double numGradients = (weight.rows() * weight.cols());
        final Mat out = new Mat();
        outSum.convertTo(out, CV_32F, 1.0 / numGradients);

        final Core.MinMaxLocResult minMaxLocResult = minMaxLoc(out);
        final double maxVal = minMaxLocResult.maxVal;
        Point maxP = minMaxLocResult.maxLoc;
        //-- Flood fill the edges
        if (options.kEnablePostProcess) {
            final Mat floodClone = new Mat();
            final double floodThresh = maxVal * options.kPostProcessThreshold;
            threshold(out, floodClone, floodThresh, 0.0f, THRESH_TOZERO);
            maxP = minMaxLoc(out, floodKillEdges(floodClone)).maxLoc;
        }
        return unscalePoint(maxP, eye);
    }

    private Point unscalePoint(Point p, Rect origSize) {
        final float ratio = (((float) options.kFastEyeWidth) / origSize.width);
        return new Point(round(p.x / ratio), round(p.y / ratio));
    }

    private void scaleToFastSize(Mat src, Mat dst) {
        resize(src, dst, new Size(options.kFastEyeWidth, (((float) options.kFastEyeWidth) / src.cols()) * src.rows()));
    }

    private void testPossibleCentersFormula(int x, int y, Mat weight, double gx, double gy, Mat out) {
        for (int cy = 0; cy < out.rows(); ++cy) {
            for (int cx = 0; cx < out.cols(); ++cx) {
                if (x == cx && y == cy)
                    continue;

                double dx = x - cx;
                double dy = y - cy;

                final double magnitude = sqrt((dx * dx) + (dy * dy));
                dx = dx / magnitude;
                dy = dy / magnitude;
                final double dotProduct = max(0.0, dx * gx + dy * gy);
                out.put(cy, cx, out.get(cy, cx)[0]
                        + dotProduct * dotProduct
                        * (options.kEnableWeight ? (byte) weight.get(cy, cx)[0] / options.kWeightDivisor : 1));
            }
        }
    }

    private class Options {
        // Size public static finalants
        int kEyePercentTop = 25;
        int kEyePercentSide = 13;
        int kEyePercentHeight = 30;
        int kEyePercentWidth = 35;

        // Preprocessing
        boolean kSmoothFaceImage = false;
        float kSmoothFaceFactor = 0.005f;

        // Algorithm Parameters
        int kFastEyeWidth = 50;
        int kWeightBlurSize = 5;
        boolean kEnableWeight = true;
        float kWeightDivisor = 1.0f;
        double kGradientThreshold = 50.0;

        // Postprocessing
        boolean kEnablePostProcess = true;
        float kPostProcessThreshold = 0.97f;
    }

    public class Result {
        private List<Face> faces = new ArrayList<>();

        void add(Face face) {
            faces.add(face);
        }
    }

    private class Face {
        private Rect faceArea;
        private Eye leftEye;
        private Eye rightEye;

        Face(Rect faceArea, Eye leftEye, Eye rightEye) {
            this.faceArea = faceArea;
            this.leftEye = leftEye;
            this.rightEye = rightEye;
        }
    }

    private class Eye {
        Point pupil;
        Rect leftCornerRegion;
        Rect rightCornerRegion;
        Point leftCorner;
        Point rightCorner;

        Eye(Point pupil, Rect leftCornerRegion, Rect rightCornerRegion, Point leftCorner, Point rightCorner) {
            this.pupil = pupil;
            this.leftCornerRegion = leftCornerRegion;
            this.rightCornerRegion = rightCornerRegion;
            this.leftCorner = leftCorner;
            this.rightCorner = rightCorner;
        }
    }
}