package com.example.beomseok.opencv_face_detect;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import com.opencsv.CSVReader;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = "OCVSample::Activity";
    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;

    private MenuItem               mItemFace50;
    private MenuItem               mItemFace40;
    private MenuItem               mItemFace30;
    private MenuItem               mItemFace20;
    private MenuItem               mItemType;

    private Mat                    mRgba;
    private Mat                    mGray;
    private Mat                    mCrop;
    private Mat                    mResize;
    private byte[]                 mRgbByte;
    private File                   mCascadeFile;
    private File                   mCascadeEyeFile;

    private CascadeClassifier      mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;

    private int                    mDetectorType       = NATIVE_DETECTOR; //JAVA_DETECTOR;
    private String[]               mDetectorName;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private Caffe2 caffe2;
    private int netNum = 0;
    private Button maButton;

    private Mat contentMat;
    private Mat styleMat;
    private float [] landmarks;

    private Mat loadImage(String imageName) {
        InputStream stream = null;
        Uri uri = Uri.parse("android.resource://com.example.beomseok.opencv_face_detect/drawable/" + imageName);
        try {
            stream = getContentResolver().openInputStream(uri);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
        bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;

        Bitmap bmp = BitmapFactory.decodeStream(stream, null, bmpFactoryOptions);
        Mat mat = new Mat();
        Utils.bitmapToMat(bmp, mat);

        return mat;
    }

    private File copyFile(int res, String fileName) {
        try {
            // load cascade file from application resources
            InputStream is = getResources().openRawResource(res);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File file = new File(cascadeDir, fileName);
            FileOutputStream os = new FileOutputStream(file);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            cascadeDir.delete();

            return file;

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }

        return null;
    }

    private float[] getLandmark() {
        File csvFile = copyFile(R.raw.seamless, "seamless.csv");

        CSVReader reader = null;
        try {
            reader = new CSVReader(new FileReader(csvFile.getAbsolutePath()));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        float [] landmarks = new float[272];
        int index = 0;
        String [] nextLine;
        try {
            for (int i = 0; i < 3; ++i) {
                nextLine = reader.readNext();

                for (int j = 0; j < nextLine.length; ++j) {
                    if(i > 0 && j > 0) {
                        landmarks[index++] = Float.valueOf(nextLine[j]);
                    }
                }
            }

//            for (int i = 0; i < landmarks.length; ++i) {
//                Log.d(TAG, "csv_" + i + " : " + landmarks[i]);
//            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return landmarks;
    }

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("native-lib");

                    mCascadeFile = copyFile(R.raw.haarcascade_frontalface_alt, "lbpcascade_frontalface.xml");
                    mCascadeEyeFile = copyFile(R.raw.haarcascade_eye, "lbpcascade_eye.xml");

                    mNativeDetector = new DetectionBasedTracker(
                            mCascadeFile.getAbsolutePath(),
                            mCascadeEyeFile.getAbsolutePath(), 0);

                    caffe2 = new Caffe2(getResources().getAssets());

                    landmarks = getLandmark();
                    contentMat = loadImage("content");
                    styleMat = loadImage("style");

                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public FdActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        mDetectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        maButton = (Button) findViewById(R.id.maButton);
        maButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                netNum++;
            }
        });

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
//        mOpenCvCameraView.setCameraIndex(0); // front-camera(1),  back-camera(0)
        mOpenCvCameraView.setCameraIndex(1); // front-camera(1),  back-camera(0)

    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
        mCrop = new Mat();
        mResize = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        mCrop.release();
        mResize.release();
    }

    private Rect padRect(Rect r) {
        double p = 0.1;
        int x_p = (int) (r.width * p);
        int y_p = (int) (r.height * p * 0.25);

        r.x -= x_p;
        r.y -= y_p;
        r.width += 2 * x_p;
        r.height += 2 * x_p;

        return r;
    }

    private boolean checkRectRange(Rect roi, int frameWidth, int frameHeight) {
      return 0 <= roi.x &&
              0 <= roi.width &&
              roi.x + roi.width <= frameWidth &&
              0 <= roi.y && 0 <= roi.height &&
              roi.y + roi.height <= frameHeight;
    };

    private Rect prevRect = null;

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        long startTime = System.currentTimeMillis();

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        int frameWidth = mRgba.width();
        int frameHeight = mRgba.height();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
            mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == NATIVE_DETECTOR) {
            if (mNativeDetector != null)
                mNativeDetector.detect(mGray, faces);
        }

        Rect[] facesArray = faces.toArray();

        if (facesArray.length > 0) {
            Rect rp = padRect(facesArray[0]);

            if (checkRectRange(rp, frameWidth, frameHeight)) {

                Mat crop = mRgba.submat(rp);

                Mat input = new Mat();
                Mat output = mRgba.submat(0,256,0,256);

                Imgproc.resize(crop, input, new Size(256,256));
                input.copyTo(output);

                Mat process = new Mat(new Size(128, 128), crop.type());
                caffe2.faceSwap(crop, process, frameWidth, frameHeight);

                Imgproc.resize(process, process, crop.size());
                process.copyTo(crop);
            }
        }

//        Mat output = mRgba.submat(0,256,0,256);
//        Mat temp = new Mat();
//
//        caffe2.seamlessClone(contentMat, styleMat, temp, landmarks);
//
//        Imgproc.resize(temp, temp, new Size(256,256));
//        temp.copyTo(output);

        long endTime = System.currentTimeMillis();

        Log.d("timecheck", "onCreate:" + (endTime - startTime));
        return mRgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType   = menu.add(mDetectorName[mDetectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);
        else if (item == mItemType) {
            int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
            item.setTitle(mDetectorName[tmpDetectorType]);
            setDetectorType(tmpDetectorType);
        }
        return true;
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    private void setDetectorType(int type) {
        if (mDetectorType != type) {
            mDetectorType = type;

            if (type == NATIVE_DETECTOR) {
                Log.i(TAG, "Detection Based Tracker enabled");
                mNativeDetector.start();
            } else {
                Log.i(TAG, "Cascade detector enabled");
                mNativeDetector.stop();
            }
        }
    }
}