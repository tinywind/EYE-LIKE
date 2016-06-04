package kr.tinywind.eyelike;

import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Point;

import static java.lang.Math.sqrt;
import static org.opencv.core.Core.meanStdDev;
import static org.opencv.core.CvType.CV_64F;

/**
 * Created by tinywind on 2016-06-04.
 */
public class Helper {
    public static Mat matrixMagnitude(Mat matX, Mat matY) {
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

    public static boolean inMat(Point p, int rows, int cols) {
        return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
    }

    public static double computeDynamicThreshold(Mat mat, double stdDevFactor) {
        MatOfDouble mean = new MatOfDouble();
        MatOfDouble stddev = new MatOfDouble();
        meanStdDev(mat, mean, stddev); // Calculates a mean and standard deviation of array elements.
        double stdDev = stddev.get(0, 0)[0] / sqrt(mat.rows() * mat.cols());
        return stdDevFactor * stdDev + mean.get(0, 0)[0];
    }
}
