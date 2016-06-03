package kr.tinywind.eyelike;

/**
 * Created by tinywind on 2016-06-04.
 */
public class Constants {
    // Debugging
    public static final boolean kPlotVectorField = false;

    // Size public static finalants
    public static final int kEyePercentTop = 25;
    public static final int kEyePercentSide = 13;
    public static final int kEyePercentHeight = 30;
    public static final int kEyePercentWidth = 35;

    // Preprocessing
    public static final boolean kSmoothFaceImage = false;
    public static final float kSmoothFaceFactor = 0.005f;

    // Algorithm Parameters
    public static final int kFastEyeWidth = 50;
    public static final int kWeightBlurSize = 5;
    public static final boolean kEnableWeight = true;
    public static final float kWeightDivisor = 1.0f;
    public static final double kGradientThreshold = 50.0;

    // Postprocessing
    public static final boolean kEnablePostProcess = true;
    public static final float kPostProcessThreshold = 0.97f;

    // Eye Corner
    public static final boolean kEnableEyeCorner = false;
}
