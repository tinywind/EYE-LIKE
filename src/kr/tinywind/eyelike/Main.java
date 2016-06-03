package kr.tinywind.eyelike;

import org.opencv.core.*;
import org.opencv.highgui.Highgui;
import org.opencv.objdetect.CascadeClassifier;

import java.io.IOException;
import java.util.List;
import java.util.Vector;

import static kr.tinywind.eyelike.Constants.*;
import static org.opencv.core.Core.*;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.highgui.Highgui.imwrite;
import static org.opencv.imgproc.Imgproc.GaussianBlur;

public class Main {
    private static String ANALYSIS_FILE_PATH = "c:/20131231_0045263.jpg";

    static {
//        System.load(Core.NATIVE_LIBRARY_NAME);
//        System.load("opencv_java249.dll");
        System.load("C:/Users/tinywind/IdeaProjects/eyeLike/lib/x64/opencv_java249.dll");
    }

    Imshow mainImshow = new Imshow("Capture - Face detection");
    private CascadeClassifier face_cascade = new CascadeClassifier();
    private String face_window_name = "Capture - Face";
    private Mat debugImage = new Mat();
    private Mat skinCrCbHist = Mat.zeros(new Size(256, 256), CV_8UC1);
    private FindEyeCenter eyeCenter = new FindEyeCenter();
    private FindEyeCorner eyeCorner = new FindEyeCorner();
//    private static RNG rng(12345);

    public static void main(String[] args) throws IOException {
        new Main().play();
    }

    private void play() throws IOException {
        Mat frame = new Mat();
        String face_cascade_name = "C:/Users/tinywind/IdeaProjects/eyeLike/resources/haarcascade_frontalface_alt.xml";
        if (!face_cascade.load(face_cascade_name)) {
            System.out.println("--(!)Error loading face cascade, please change face_cascade_name in source code.");
            return;
        }

        ellipse(skinCrCbHist, new Point(113, 155.6), new Size(23.4, 15.2), 43.0, 0.0, 360.0, new Scalar(255, 255, 255), -1);

        frame = Highgui.imread(ANALYSIS_FILE_PATH);

        flip(frame, frame, 1);
        frame.copyTo(debugImage);

        if (!frame.empty()) {
            detectAndDisplay(frame, 1);
        } else {
            System.out.println(" --(!) No captured frame -- Break!");
        }
    }

    private void findEyes(Mat frame_gray, Rect face) {
        Mat faceROI = new Mat(frame_gray, face);
        Mat debugFace = faceROI.clone();

        if (kSmoothFaceImage) {
            double sigma = kSmoothFaceFactor * face.width;
            GaussianBlur(faceROI, faceROI, new Size(0, 0), sigma);
        }
        //-- Find eye regions and draw them
        int eye_region_width = (int) (face.width * (kEyePercentWidth / 100.0));
        int eye_region_height = (int) (face.width * (kEyePercentHeight / 100.0));
        int eye_region_top = (int) (face.height * (kEyePercentTop / 100.0));
        Rect leftEyeRegion = new Rect((int) (face.width * (kEyePercentSide / 100.0)), eye_region_top, eye_region_width, eye_region_height);
        Rect rightEyeRegion = new Rect((int) (face.width - eye_region_width - face.width * (kEyePercentSide / 100.0)), eye_region_top, eye_region_width, eye_region_height);

        //-- Find Eye Centers
        Point leftPupil = eyeCenter.findEyeCenter(faceROI, leftEyeRegion, "Left Eye");
        Point rightPupil = eyeCenter.findEyeCenter(faceROI, rightEyeRegion, "Right Eye");
        // get corner regions
        Rect leftRightCornerRegion = leftEyeRegion.clone();
        leftRightCornerRegion.width -= leftPupil.x;
        leftRightCornerRegion.x += leftPupil.x;
        leftRightCornerRegion.height /= 2;
        leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
        Rect leftLeftCornerRegion = leftEyeRegion.clone();
        leftLeftCornerRegion.width = (int) leftPupil.x;
        leftLeftCornerRegion.height /= 2;
        leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
        Rect rightLeftCornerRegion = rightEyeRegion.clone();
        rightLeftCornerRegion.width = (int) rightPupil.x;
        rightLeftCornerRegion.height /= 2;
        rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
        Rect rightRightCornerRegion = rightEyeRegion.clone();
        rightRightCornerRegion.width -= rightPupil.x;
        rightRightCornerRegion.x += rightPupil.x;
        rightRightCornerRegion.height /= 2;
        rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
        rectangle(debugFace, leftRightCornerRegion.tl(), leftRightCornerRegion.br(), new Scalar(200));
        rectangle(debugFace, leftLeftCornerRegion.tl(), leftLeftCornerRegion.br(), new Scalar(200));
        rectangle(debugFace, rightLeftCornerRegion.tl(), rightLeftCornerRegion.br(), new Scalar(200));
        rectangle(debugFace, rightRightCornerRegion.tl(), rightRightCornerRegion.br(), new Scalar(200));
        // change eye centers to face coordinates
        rightPupil.x += rightEyeRegion.x;
        rightPupil.y += rightEyeRegion.y;
        leftPupil.x += leftEyeRegion.x;
        leftPupil.y += leftEyeRegion.y;
        // draw eye centers
        circle(debugFace, rightPupil, 3, new Scalar(1234));
        circle(debugFace, leftPupil, 3, new Scalar(1234));

        //-- Find Eye Corners
        if (kEnableEyeCorner) {
            Point leftRightCorner = eyeCorner.findEyeCorner(new Mat(faceROI, leftRightCornerRegion), true, false);
            leftRightCorner.x += leftRightCornerRegion.x;
            leftRightCorner.y += leftRightCornerRegion.y;
            Point leftLeftCorner = eyeCorner.findEyeCorner(new Mat(faceROI, leftLeftCornerRegion), true, true);
            leftLeftCorner.x += leftLeftCornerRegion.x;
            leftLeftCorner.y += leftLeftCornerRegion.y;
            Point rightLeftCorner = eyeCorner.findEyeCorner(new Mat(faceROI, rightLeftCornerRegion), false, true);
            rightLeftCorner.x += rightLeftCornerRegion.x;
            rightLeftCorner.y += rightLeftCornerRegion.y;
            Point rightRightCorner = eyeCorner.findEyeCorner(new Mat(faceROI, rightRightCornerRegion), false, false);
            rightRightCorner.x += rightRightCornerRegion.x;
            rightRightCorner.y += rightRightCornerRegion.y;
            circle(faceROI, leftRightCorner, 3, new Scalar(200));
            circle(faceROI, leftLeftCorner, 3, new Scalar(200));
            circle(faceROI, rightLeftCorner, 3, new Scalar(200));
            circle(faceROI, rightRightCorner, 3, new Scalar(200));
        }

        new Imshow(face_window_name).showImage(faceROI);
    }

    private void detectAndDisplay(Mat frame, int index) {
        Vector<Mat> rgbChannels = new Vector<>(3);
        split(frame, rgbChannels);
        Mat frame_gray = rgbChannels.get(2).clone();

        int width = frame.cols();
        int height = frame.rows();
        int min = Integer.min(width, height);
        int max = Integer.max(width, height);
        int faceMinSideLength = min / 3;
        int faceMaxSideLength = (int) (max * 1.5);

        MatOfRect faces = new MatOfRect();
        face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0,
                new Size(faceMinSideLength, faceMinSideLength),
                new Size(faceMaxSideLength, faceMaxSideLength));
        List<Rect> rectList = faces.toList();

        for (int i = 0; i < rectList.size(); i++) {
            Rect e = rectList.get(i);
            rectangle(debugImage, e.tl(), e.br(), new Scalar(0, 255, 0), 5);
            imwrite("debug_face.png", debugImage);
            findEyes(frame_gray, e);
            // mainImshow.showImage(debugImage);
        }
    }
}