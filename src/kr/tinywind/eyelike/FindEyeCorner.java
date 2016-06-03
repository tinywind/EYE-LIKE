package kr.tinywind.eyelike;

import org.opencv.core.*;

import static org.opencv.core.Core.flip;
import static org.opencv.core.Core.minMaxLoc;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.imgproc.Imgproc.*;

/**
 * Created by tinywind on 2016-06-04.
 */
public class FindEyeCorner {
    private float[][] kEyeCornerKernel = {
            {-1, -1, -1, 1, 1, 1},
            {-1, -1, -1, -1, 1, 1},
            {-1, -1, -1, -1, 0, 3},
            {1, 1, 1, 1, 1, 1},
    };
    private Mat rightCornerKernel = new Mat(4, 6, CV_32F, new Scalar(kEyeCornerKernel[0][0], kEyeCornerKernel[1][0], kEyeCornerKernel[2][0], kEyeCornerKernel[3][0]));
    private Mat leftCornerKernel = new Mat(4, 6, CV_32F);

    public FindEyeCorner() {
        flip(rightCornerKernel, leftCornerKernel, 1);
    }

    private Mat eyeCornerMap(Mat region, boolean left, boolean left2) {
        Mat cornerMap = new Mat();
        Size sizeRegion = region.size();
        Range colRange = new Range((int) (sizeRegion.width / 4), (int) (sizeRegion.width * 3 / 4));
        Range rowRange = new Range((int) (sizeRegion.height / 4), (int) (sizeRegion.height * 3 / 4));
        Mat miRegion = new Mat(region, rowRange, colRange);
        filter2D(miRegion, cornerMap, CV_32F, (left && !left2) || (!left && !left2) ? leftCornerKernel : rightCornerKernel);
        return cornerMap;
    }

    public Point findEyeCorner(Mat region, boolean left, boolean left2) {
        return findSubpixelEyeCorner(eyeCornerMap(region, left, left2));
    }

    public Point findSubpixelEyeCorner(Mat region) {
        Size sizeRegion = region.size();
        Mat cornerMap = new Mat((int) sizeRegion.height * 10, (int) sizeRegion.width * 10, CV_32F);
        resize(region, cornerMap, cornerMap.size(), 0, 0, INTER_CUBIC);
        Point maxP2 = minMaxLoc(cornerMap).maxLoc;
        return new Point(sizeRegion.width / 2 + maxP2.x / 10, sizeRegion.height / 2 + maxP2.y / 10);
    }
}
