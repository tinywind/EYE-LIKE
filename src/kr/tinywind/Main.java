package kr.tinywind;

import kr.tinywind.opencv.EyeFinder;

import java.io.IOException;

import static org.opencv.highgui.Highgui.imwrite;

/**
 * Created by tinywind on 2016-06-06.
 */
public class Main {
    public static void main(String[] args) throws IOException {
        EyeFinder eyeFinder = new EyeFinder("C:/20131231_0045263.jpg");
        EyeFinder.Result result = eyeFinder.result();
        System.out.println(result);
        imwrite("debugImage.png", eyeFinder.getDebugImage());
    }
}
