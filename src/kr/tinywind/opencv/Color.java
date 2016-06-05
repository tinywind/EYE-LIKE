package kr.tinywind.opencv;

import org.opencv.core.Scalar;

/**
 * Created by tinywind on 2016-06-06.
 */
public class Color {
    public static final Scalar GREEN = new Scalar(java.awt.Color.GREEN.getRed(), java.awt.Color.GREEN.getGreen(), java.awt.Color.GREEN.getBlue());
    public static final Scalar RED = new Scalar(java.awt.Color.RED.getRed(), java.awt.Color.RED.getGreen(), java.awt.Color.RED.getBlue());
    public static final Scalar BLUE = new Scalar(java.awt.Color.BLUE.getRed(), java.awt.Color.BLUE.getGreen(), java.awt.Color.BLUE.getBlue());
    public static final Scalar YELLOW = new Scalar(java.awt.Color.YELLOW.getRed(), java.awt.Color.YELLOW.getGreen(), java.awt.Color.YELLOW.getBlue());
    public static final Scalar BLACK = new Scalar(java.awt.Color.BLACK.getRed(), java.awt.Color.BLACK.getGreen(), java.awt.Color.BLACK.getBlue());
    public static final Scalar WHITE = new Scalar(java.awt.Color.WHITE.getRed(), java.awt.Color.WHITE.getGreen(), java.awt.Color.WHITE.getBlue());

    private Color() {
    }
}
