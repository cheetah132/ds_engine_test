package com.example.beomseok.opencv_face_detect;

import android.content.res.AssetManager;

import org.opencv.core.Mat;

/**
 * Created by Beomseok on 2017. 12. 19..
 */

public class Caffe2 {

    public Caffe2(AssetManager mgr) {
        init(mgr);
    }

    public void faceSwap(Mat input, Mat output, int w, int h) {
        nativeFaceSwap(input.getNativeObjAddr(), output.getNativeObjAddr(), w, h);
    }

    public void seamlessClone(Mat content, Mat style, Mat output, float[] landmarks) {
        nativeSeamlessClone(
                content.getNativeObjAddr(),
                style.getNativeObjAddr(),
                output.getNativeObjAddr(),
                landmarks);
    }

    public void init(AssetManager mgr) {
        initCaffe2(mgr);
    }

    private native void nativeFaceSwap(long input, long output, int w, int h);
    private native void nativeSeamlessClone(long content, long style, long output, float[] landmarks);
    private native void initCaffe2(AssetManager mgr);
}
