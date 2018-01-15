#include <jni.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <android/log.h>

#include <string>
#include <vector>
#include <ctime>

#include <android/log.h>

#define LOG_TAG "FaceTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace cv;
using namespace std;

inline void vector_Rect_to_Mat(vector<Rect>& v_rect, Mat& mat)
{
    mat = Mat(v_rect, true);
}

class CascadeDetectorAdapter: public DetectionBasedTracker::IDetector
{
public:
    CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector):
            IDetector(),
            Detector(detector)
    {
        LOGD("CascadeDetectorAdapter::Detect::Detect");
        CV_Assert(detector);
    }

    void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
    {
        LOGD("CascadeDetectorAdapter::Detect: begin");
        LOGD("CascadeDetectorAdapter::Detect: scaleFactor=%.2f, minNeighbours=%d, minObjSize=(%dx%d), maxObjSize=(%dx%d)", scaleFactor, minNeighbours, minObjSize.width, minObjSize.height, maxObjSize.width, maxObjSize.height);
        Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
        LOGD("CascadeDetectorAdapter::Detect: end");
    }

    virtual ~CascadeDetectorAdapter()
    {
        LOGD("CascadeDetectorAdapter::Detect::~Detect");
    }

private:
    CascadeDetectorAdapter();
    cv::Ptr<cv::CascadeClassifier> Detector;
};

struct DetectorAgregator
{
    cv::Ptr<CascadeDetectorAdapter> mainDetector;
    cv::Ptr<CascadeDetectorAdapter> trackingDetector;

    cv::Ptr<DetectionBasedTracker> tracker;
    DetectorAgregator(cv::Ptr<CascadeDetectorAdapter>& _mainDetector, cv::Ptr<CascadeDetectorAdapter>& _trackingDetector):
            mainDetector(_mainDetector),
            trackingDetector(_trackingDetector)
    {
        CV_Assert(_mainDetector);
        CV_Assert(_trackingDetector);

        DetectionBasedTracker::Parameters DetectorParams;
        tracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
    }
};

static CascadeClassifier * eyeDetector;

extern "C" {

JNIEXPORT jlong JNICALL Java_com_example_beomseok_opencv_1face_1detect_DetectionBasedTracker_nativeCreateObject
        (JNIEnv * jenv, jclass, jstring jFileName, jstring jEyeFileName, jint faceSize)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCreateObject enter");
    const char* jnamestr = jenv->GetStringUTFChars(jFileName, NULL);
    string stdFileName(jnamestr);

    const char* jeyenamestr = jenv->GetStringUTFChars(jEyeFileName, NULL);
    string stdEyeFileName(jeyenamestr);

    jlong result = 0;

    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCreateObject");

    try
    {
        cv::Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(
                makePtr<CascadeClassifier>(stdFileName));
        cv::Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(
                makePtr<CascadeClassifier>(stdFileName));

        result = (jlong)new DetectorAgregator(mainDetector, trackingDetector);

        eyeDetector = new CascadeClassifier(jeyenamestr);

        if (faceSize > 0)
        {
            mainDetector->setMinObjectSize(Size(faceSize, faceSize));
            //trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
        }
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeCreateObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeCreateObject()");
        return 0;
    }

    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeCreateObject exit");
    return result;
}

JNIEXPORT void JNICALL Java_com_example_beomseok_opencv_1face_1detect_DetectionBasedTracker_nativeDestroyObject
        (JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDestroyObject");

    try
    {
        if(thiz != 0)
        {
            ((DetectorAgregator*)thiz)->tracker->stop();
            delete (DetectorAgregator*)thiz;
        }
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeestroyObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeDestroyObject caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeDestroyObject()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDestroyObject exit");
}

JNIEXPORT void JNICALL Java_com_example_beomseok_opencv_1face_1detect_DetectionBasedTracker_nativeStart
        (JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStart");

    try
    {
        ((DetectorAgregator*)thiz)->tracker->run();
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStart caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeStart caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStart()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStart exit");
}

JNIEXPORT void JNICALL Java_com_example_beomseok_opencv_1face_1detect_DetectionBasedTracker_nativeStop
        (JNIEnv * jenv, jclass, jlong thiz)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStop");

    try
    {
        ((DetectorAgregator*)thiz)->tracker->stop();
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeStop caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeStop()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeStop exit");
}

JNIEXPORT void JNICALL Java_com_example_beomseok_opencv_1face_1detect_DetectionBasedTracker_nativeSetFaceSize
        (JNIEnv * jenv, jclass, jlong thiz, jint faceSize)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeSetFaceSize -- BEGIN");

    try
    {
        if (faceSize > 0)
        {
            ((DetectorAgregator*)thiz)->mainDetector->setMinObjectSize(Size(faceSize, faceSize));
            //((DetectorAgregator*)thiz)->trackingDetector->setMinObjectSize(Size(faceSize, faceSize));
        }
    }
    catch(cv::Exception& e)
    {
        LOGD("nativeStop caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeSetFaceSize caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code of DetectionBasedTracker.nativeSetFaceSize()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeSetFaceSize -- END");
}


JNIEXPORT void JNICALL Java_com_example_beomseok_opencv_1face_1detect_DetectionBasedTracker_nativeDetect
        (JNIEnv * jenv, jclass, jlong thiz, jlong imageGray, jlong faces)
{
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect");

    try
    {
        vector<Rect> RectFaces;
        ((DetectorAgregator*)thiz)->tracker->process(*((Mat*)imageGray));
        ((DetectorAgregator*)thiz)->tracker->getObjects(RectFaces);
        *((Mat*)faces) = Mat(RectFaces, true);

    }
    catch(cv::Exception& e)
    {
        LOGD("nativeCreateObject caught cv::Exception: %s", e.what());
        jclass je = jenv->FindClass("org/opencv/core/CvException");
        if(!je)
            je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, e.what());
    }
    catch (...)
    {
        LOGD("nativeDetect caught unknown exception");
        jclass je = jenv->FindClass("java/lang/Exception");
        jenv->ThrowNew(je, "Unknown exception in JNI code DetectionBasedTracker.nativeDetect()");
    }
    LOGD("Java_org_opencv_samples_facedetect_DetectionBasedTracker_nativeDetect END");
}

JNIEXPORT void JNICALL Java_com_example_beomseok_opencv_1face_1detect_DetectionBasedTracker_nativeEyeDetect
        (JNIEnv * jenv, jlong matAddrInput, jlong matAddrResult)
{
    Mat &img_input = *(Mat *) matAddrInput;
    Mat &img_result = *(Mat *) matAddrResult;
    img_result = img_input.clone();

    std::vector<Rect> eyes;

    eyeDetector->detectMultiScale(img_input, eyes, 1.1, 3, 0 |CASCADE_SCALE_IMAGE, Size(20, 20), Size(40, 40));

    if (eyes.size() > 2) {
        LOGD("eye size %d", eyes.size());
        for ( size_t j = 0; j < 2; j++ )
        {
            LOGD("eye size %d %d", eyes[j].width,eyes[j].height);
            Point eye_center(eyes[j].x + eyes[j].width/2, eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( img_result, eye_center, radius, Scalar( 255, 0, 0 ), 30, 8, 0 );
        }
    }
}
}

#include <caffe2/core/predictor.h>
#include <caffe2/core/timer.h>
#include <cpu-features.h>

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#define IMG_H 128
#define IMG_W 128
#define IMG_O_H 128
#define IMG_O_W 128

#define IMG_C 3
#define MAX_O_DATA_SIZE IMG_O_H * IMG_O_W * IMG_C
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C

#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "OpenGL", __VA_ARGS__);

#include <caffe2/mobile/contrib/opengl/android/AndroidGLContext.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/operators/conv_pool_op_base.h>
#include <unordered_map>
#include <unordered_set>

using namespace caffe2;

static bool opengl = true;

static NetDef _initNet, _predictNet;
static NetDef _resizeInitNet, _resizePredictNet;

static NetDef _glPreictNet;

static Workspace workspace;
static bool isLoad = false;
static bool isInit = false;

static Predictor * _predictor;

struct Analysis {
    struct SSA {
        using BlobVersions = std::unordered_map<std::string, size_t>;
        BlobVersions inVersions;
        BlobVersions outVersions;
    };
    std::vector<SSA> ssa;
    std::unordered_map<std::string, std::unordered_map<size_t, std::vector<size_t>>> inUsages;
};

static Analysis analyzeNet(const NetDef& net) {
    Analysis::SSA::BlobVersions frontier;
    Analysis analysis;

    auto play = [&](size_t i, const OperatorDef& op) {
        Analysis::SSA::BlobVersions inVersions;
        for (const auto& s : op.input()) {
            inVersions[s] = frontier[s];
            analysis.inUsages[s][frontier[s]].push_back(i);
        }
        Analysis::SSA::BlobVersions outVersions;
        for (const auto& s : op.output()) {
            if (frontier.find(s) != frontier.end()) {
                frontier[s] += 1;
            }
            outVersions[s] = frontier[s];
        }
        analysis.ssa.push_back(Analysis::SSA{inVersions, outVersions});
    };

    for (auto i = 0; i < net.op_size(); ++i) {
        play(i, net.op(i));
    }
    return analysis;
}

static void insertCopyToGPUOp(NetDef& predictNet, const std::string& cpu_blob) {
    auto* op = predictNet.add_op();
    op->set_name("CopyToOpenGL");
    op->set_type("CopyToOpenGL");
    op->add_input(cpu_blob);
    op->add_output(cpu_blob + "_M");
}

static void insertCopyFromGPUOp(NetDef& predictNet, const std::string& cpu_blob) {
    // add argument "is_last" to the last op to signal this is the last operator before the
    // CopyFromOpenGL op
    auto* last_op = predictNet.mutable_op(predictNet.op_size() - 1);
    auto* arg = last_op->add_arg();
    arg->set_name("is_last");
    arg->set_i(1);

    auto* op = predictNet.add_op();
    op->set_name("CopyFromOpenGL");
    op->set_type("CopyFromOpenGL");
    op->add_input(cpu_blob + "_M");
    op->add_output(cpu_blob);
}

static NetDef insertInputOutputCopyOps(const NetDef& def, std::unordered_set<std::string>& glOps) {
    // Do some validation of the outputs. For this version, we require:
    // - a single input (first element of external_input()) is consumed by the NetDef
    // - a single output (first element of external_output()) is produced by the NetDef.
    // - the input is consumed by def.op(0), and this is the only consumer.
    // - the output is produced by def.op(-1).
    alog("external input %d", def.external_input_size());

//    for (auto i = 0; i < def.external_input_size(); ++i)
//        alog("%d : %s", i, def.external_input(i).c_str());

    CAFFE_ENFORCE_GE(def.external_input_size(), 1);
    CAFFE_ENFORCE_GE(def.external_output_size(), 1);
    auto analysis = analyzeNet(def);
    // enforce a single use of the input blob.
    CAFFE_ENFORCE_GE(def.op_size(), 1);

    const auto& inputBlob = def.external_input(0);
    // Enforce that the input blob has a single usage - in the first operator.

    alog("%s %d", inputBlob.c_str(), analysis.inUsages[inputBlob][0][0]);

    //CAFFE_ENFORCE(analysis.inUsages[inputBlob][0] == (std::vector<size_t>{0}));

    // Enforce that the external_output(0) blob is produced by the last operator in this sequence.
    const auto& outputBlob = def.external_output(0);
    CAFFE_ENFORCE(analysis.ssa.back().outVersions.find(outputBlob) !=
                  analysis.ssa.back().outVersions.end());
    const auto& outputBlobVersion = analysis.ssa.back().outVersions[outputBlob];

    // This should hold true by definition of the SSA analysis.
    CAFFE_ENFORCE(analysis.inUsages[outputBlob].find(outputBlobVersion) ==
                  analysis.inUsages[outputBlob].end());

    NetDef mdef;
    mdef.CopyFrom(def);
    mdef.clear_op();

    std::unordered_map<std::string, std::set<size_t>> cpu_blobs, gpu_blobs;

    for (auto i = 0; i < 5; ++i) {
        alog("%d %s", i, def.external_input(i).c_str());
        cpu_blobs[def.external_input(i)].insert(0);
    }

//    alog("opsize : %d", def.op_size());
    for (auto i = 0; i < def.op_size(); i++) {
        const auto& currentOp = def.op(i);
//        alog("%d, name : %s, input_size : %d ", i, currentOp.type().c_str(), currentOp.input_size());
        if (glOps.count(currentOp.type()) > 0) {
            // OpenGL Op
            // insert copyToOpenGLOp
            for (auto j = 0; j < currentOp.input_size(); j++) {
                auto& input = currentOp.input(j);
//                alog("input :  %d", j);
//                alog("input :  %s", input.c_str());
//                insertCopyToGPUOp(mdef, input);
                auto version = analysis.ssa[i].inVersions[input];
                if (cpu_blobs[input].count(version) > 0) {
                    alog("insert %s", input.c_str());
                    insertCopyToGPUOp(mdef, input);
                    gpu_blobs[input].insert(version);
                    cpu_blobs[input].erase(version);
                }
                // Only the first input should be OpenGL texture
                // Otherwise, copyToOpenGLOp will be inserted for the weights,
                // which are outputs of QuantDecode
                if (currentOp.type().find("OpenGLConv") == 0) {
                    if (j == 0) {
                        break;
                    }
                }
            }

            auto* op = mdef.add_op();
            op->CopyFrom(currentOp);

            // swap input blob
            for (auto j = 0; j < currentOp.input_size(); j++) {
                auto& input = currentOp.input(j);
                auto version = analysis.ssa[i].inVersions[input];
                if (gpu_blobs[input].count(version) > 0) {
                    op->set_input(j, input + "_M");
                }
            }

            // swap output blob
            for (auto j = 0; j < currentOp.output_size(); j++) {
                auto& output = currentOp.output(j);
                auto version = analysis.ssa[i].outVersions[output];
                op->set_output(j, output + "_M");
                gpu_blobs[output].insert(version);
            }
            // insert copyFromOpenGLOp after the last op if the last op is an OpenGL op
            if (i == def.op_size() - 1) {
                insertCopyFromGPUOp(mdef, currentOp.output(0));
            }
        } else {
            // CPU Op
            // insert copyFromOpenGLOp
            for (auto j = 0; j < currentOp.input_size(); j++) {
                auto& input = currentOp.input(j);
                auto version = analysis.ssa[i].inVersions[input];
                if (gpu_blobs[input].count(version) > 0) {
                    insertCopyFromGPUOp(mdef, input);
                }
            }
            auto* op = mdef.add_op();
            op->CopyFrom(currentOp);
            for (auto j = 0; j < currentOp.output_size(); j++) {
                auto& output = currentOp.output(j);
                auto version = analysis.ssa[i].outVersions[output];
                cpu_blobs[output].insert(version);
            }
        }
    }
    return mdef;
}

static bool tryFuseAdjacentOps(const OperatorDef& currentOp,
                               const OperatorDef& nextOp,
                               OperatorDef* fusedOp,
                               std::unordered_set<std::string>& glOps) {
    // Check for possible invalid opportunities.
    if (currentOp.output_size() != 1 || nextOp.output_size() != 1) {
        return false;
    }
    // The fused op cannot be inplace
    if (currentOp.output(0) != nextOp.input(0) || currentOp.input(0) == nextOp.output(0)) {
        return false;
    }

    static const std::map<std::pair<std::string, std::string>, std::string> fusionOpportunities = {
            {{"OpenGLInstanceNorm", "OpenGLPRelu"}, "OpenGLInstanceNormPRelu"},
            {{"OpenGLConv", "OpenGLPRelu"}, "OpenGLConvPRelu"},
            {{"OpenGLConv", "OpenGLRelu"}, "OpenGLConvRelu"},
            {{"OpenGLConvTranspose", "OpenGLPRelu"}, "OpenGLConvTransposePRelu"}};
    auto it = fusionOpportunities.find({currentOp.type(), nextOp.type()});
    if (it == fusionOpportunities.end()) {
        return false;
    }

    glOps.insert(it->second);
    fusedOp->CopyFrom(currentOp);
    fusedOp->set_output(0, nextOp.output(0));
    fusedOp->set_type(it->second);
    for (auto i = 1; i < nextOp.input_size(); i++) {
        fusedOp->add_input(nextOp.input(i));
    }
    return true;
}

static NetDef runOpenGLFusion(const NetDef& def, std::unordered_set<std::string>& glOps) {
    CHECK_GE(def.op_size(), 1);
    NetDef mdef;
    mdef.CopyFrom(def);
    mdef.clear_op();
    auto i = 0;

    while (i < def.op_size()) {
        if (i == def.op_size() - 1) {
            VLOG(2) << "Last operator, skipping";
            auto* op = mdef.add_op();
            op->CopyFrom(def.op(i));
            i += 1;
            continue;
        }

        const auto& currentOp = def.op(i);
        const auto& nextOp = def.op(i + 1);
        OperatorDef fusedOp;
        if (tryFuseAdjacentOps(currentOp, nextOp, &fusedOp, glOps)) {
            VLOG(2) << "Found an adjacent fusion for: " << currentOp.type() << ", " << nextOp.type();
            // We can fuse.
            auto* op = mdef.add_op();
            op->CopyFrom(fusedOp);
            i += 2;
            continue;
        }
        VLOG(2) << "No fusion available for: " << currentOp.type() << ", " << nextOp.type();
        // Just emit the current type.
        auto* op = mdef.add_op();
        op->CopyFrom(currentOp);
        i += 1;
    }
    return mdef;
}

NetDef rewritePredictNetForOpenGL(const NetDef& predictNet, bool useTextureInput, bool useTiling, bool runFusion) {
    CAFFE_ENFORCE_GE(predictNet.op_size(), 1);
    NetDef net;
    net.CopyFrom(predictNet);

    std::unordered_map<std::string, std::string> replacements(
            {{"OpenGLPackedInt8BGRANHWCToNCHWCStylizerPreprocess",
                     useTextureInput ? "OpenGLTextureToTextureStylizerPreprocess"
                                     : "OpenGLTensorToTextureStylizerPreprocess"},
             {"OpenGLBRGNCHWCToPackedInt8BGRAStylizerDeprocess",
                     useTextureInput ? "OpenGLTextureToTextureStylizerDeprocess"
                                     : "OpenGLTextureToTensorStylizerDeprocess"}});

    std::unordered_set<std::string> openGLOps; // Used to insert copy ops
    bool needCopyOps = false;

    const auto& opKeyList = CPUOperatorRegistry()->Keys();
    auto opKeySet = std::set<std::string>(opKeyList.begin(), opKeyList.end());

#ifdef CAFFE2_ANDROID
    // TODO: debug InstanceNorm models on Mali devices
    AndroidGLContext* context = (AndroidGLContext*)GLContext::getGLContext();
    if (context->get_platform() == Mali) {
        opKeySet.erase("OpenGLInstanceNorm");
        opKeySet.erase("OpenGLInstanceNormPRelu");
    }
#endif
    for (auto i = 0; i < net.op_size(); ++i) {
        auto* op = net.mutable_op(i);
        string openGLOp = std::string("OpenGL") + op->type();
        if (replacements.count(openGLOp) > 0) {
            openGLOp = replacements[openGLOp];
        }

        if (opKeySet.find(openGLOp) != opKeySet.end()) {
            op->set_type(openGLOp);
            openGLOps.insert(openGLOp);

            if (useTiling) {
                auto* arg = op->add_arg();
                arg->set_name("tiling");
                arg->set_i(1);
            }
        } else {
            needCopyOps = true;
        }
    }

    if (useTextureInput && needCopyOps) {
        CAFFE_THROW("OpenGL operator missing");
    }

    if (runFusion) {
        net = runOpenGLFusion(net, openGLOps);
    }

    if (net.op(0).type() == replacements["OpenGLPackedInt8BGRANHWCToNCHWCStylizerPreprocess"]) {
        // For end-to-end testing
        if (net.op(net.op_size() - 1).type() !=
            replacements["OpenGLBRGNCHWCToPackedInt8BGRAStylizerDeprocess"]) {
            auto* last_op = net.mutable_op(net.op_size() - 1);
            auto output = last_op->output(0) + "M";
            last_op->set_output(0, output);
            auto* copy_op = net.add_op();
            copy_op->set_name("CopyFromOpenGL");
            copy_op->set_type("CopyFromOpenGL");
            copy_op->add_input(output);
            // rename output blob in case input and output blob has the same name
            copy_op->add_output(net.external_output(0));
        }
    } else {
        if (!useTextureInput) {
            needCopyOps = true;
        }
    }

    // copy ops are needed when the input is not a texture
    if (needCopyOps) {
        // For non style transfer cases
        net = insertInputOutputCopyOps(net, openGLOps);
    }

    return net;
}

static TensorCPU * _input_128;
static TensorCPU * _input_64;
static TensorCPU * _input_32;
static TensorCPU * _input_16;
static TensorCPU * _input_8;
static TensorCPU * _input;

static TensorCPU * _output;

static float input_hwc_128[MAX_DATA_SIZE];
static float input_chw_128[MAX_DATA_SIZE];
static float output_chw[MAX_O_DATA_SIZE];
static float output_hwc[MAX_O_DATA_SIZE];

static Mat & m_8UC4_128 = * (new Mat(IMG_H, IMG_W, CV_8UC4));
static Mat & m_8UC3_128 = * (new Mat(IMG_H, IMG_W, CV_8UC3));
static Mat & m_32FC3_128 = * (new Mat(IMG_H, IMG_W, CV_32FC3));

static Mat & m_8UC3_O = * (new Mat(IMG_O_H, IMG_O_W, CV_8UC3));
static Mat & m_32FC3_O = * (new Mat(IMG_O_H, IMG_O_W, CV_32FC3));

void inference(const NetDef& initNet, NetDef predictNet) {

    if(opengl) {
        if (_predictor) {

            TensorCPU input;
            _input = &input;
            _input->Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));

            memcpy(_input->mutable_data<float>(), input_chw_128, IMG_H * IMG_W * IMG_C * sizeof(float));

            caffe2::Predictor::TensorVector input_vec{_input};
            caffe2::Predictor::TensorVector output_vec;

            _predictor->run(input_vec, &output_vec);

            if (!isInit) {

                alog("set output blob name");
                auto output_blob = "_OUTPUT_BLOB__";
                predictNet.set_external_output(0, output_blob);
                predictNet.mutable_op(predictNet.op_size() - 1)->set_output(0, output_blob);

                alog("rewritePredictNetForOpenGL");
                _glPreictNet = rewritePredictNetForOpenGL(predictNet, false, false, true);

                alog("init graph");
                workspace.RunNetOnce(initNet);

                alog("init input");

                _input_128 = workspace.CreateBlob(_glPreictNet.external_input(4))->GetMutable<TensorCPU>();
                _input_64 = workspace.CreateBlob(_glPreictNet.external_input(1))->GetMutable<TensorCPU>();
                _input_32 = workspace.CreateBlob(_glPreictNet.external_input(2))->GetMutable<TensorCPU>();
                _input_16 = workspace.CreateBlob(_glPreictNet.external_input(0))->GetMutable<TensorCPU>();
                _input_8 = workspace.CreateBlob(_glPreictNet.external_input(3))->GetMutable<TensorCPU>();

                _input_128->Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));
                _input_64->Resize(std::vector<int>({1, IMG_C, IMG_H/2, IMG_W/2}));
                _input_32->Resize(std::vector<int>({1, IMG_C, IMG_H/4, IMG_W/4}));
                _input_16->Resize(std::vector<int>({1, IMG_C, IMG_H/8, IMG_W/8}));
                _input_8->Resize(std::vector<int>({1, IMG_C, IMG_H/16, IMG_W/16}));

                memcpy(_input_128->mutable_data<float>(), input_chw_128, IMG_H * IMG_W * IMG_C * sizeof(float));
                memcpy(_input_64->mutable_data<float>(), output_vec[2]->data<float>(), IMG_H/2 * IMG_W/2 * IMG_C * sizeof(float));
                memcpy(_input_32->mutable_data<float>(), output_vec[3]->data<float>(), IMG_H/4 * IMG_W/4 * IMG_C * sizeof(float));
                memcpy(_input_16->mutable_data<float>(), output_vec[0]->data<float>(), IMG_H/8 * IMG_W/8 * IMG_C * sizeof(float));
                memcpy(_input_8->mutable_data<float>(), output_vec[1]->data<float>(), IMG_H/16 * IMG_W/16 * IMG_C * sizeof(float));

                alog("create net");
                workspace.CreateNet(_glPreictNet, true);

                isInit = true;

                alog("init done");
            } else {
                memcpy(_input_128->mutable_data<float>(), input_chw_128, IMG_H * IMG_W * IMG_C * sizeof(float));
                memcpy(_input_64->mutable_data<float>(), output_vec[2]->data<float>(), IMG_H/2 * IMG_W/2 * IMG_C * sizeof(float));
                memcpy(_input_32->mutable_data<float>(), output_vec[3]->data<float>(), IMG_H/4 * IMG_W/4 * IMG_C * sizeof(float));
                memcpy(_input_16->mutable_data<float>(), output_vec[0]->data<float>(), IMG_H/8 * IMG_W/8 * IMG_C * sizeof(float));
                memcpy(_input_8->mutable_data<float>(), output_vec[1]->data<float>(), IMG_H/16 * IMG_W/16 * IMG_C * sizeof(float));
            }

            workspace.RunNet(_glPreictNet.name());

            _output = workspace.GetBlob("_OUTPUT_BLOB__")->GetMutable<TensorCPU>();
            memcpy(output_chw, _output->mutable_data<float>(), IMG_O_H * IMG_O_W * IMG_C * sizeof(float));
        }
    } else {
        if (_predictor) {

            TensorCPU input_128;

            _input_128 = &input_128;
            _input_128->Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));

            memcpy(_input_128->mutable_data<float>(), input_chw_128, IMG_H * IMG_W * IMG_C * sizeof(float));

            caffe2::Predictor::TensorVector input_vec{_input_128};
            caffe2::Predictor::TensorVector output_vec;

            _predictor->run(input_vec, &output_vec);

            memcpy(output_chw, output_vec[0]->data<float>(), IMG_O_H * IMG_O_W * IMG_C * sizeof(float));
        }
    }
}

// A function to load the NetDefs from protobufs.
void loadToNetDef(AAssetManager* mgr, NetDef* net, const char *filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    assert(asset != nullptr);
    const void *data = AAsset_getBuffer(asset);
    assert(data != nullptr);
    off_t len = AAsset_getLength(asset);
    assert(len != 0);
    if (!net->ParseFromArray(data, len)) {
        alog("Couldn't parse net from data.\n");
    }
    AAsset_close(asset);
}

extern "C"
void
Java_com_example_beomseok_opencv_1face_1detect_Caffe2_initCaffe2(
        JNIEnv* env,
        jobject /* this */,
        jobject assetManager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    alog("Attempting to load protobuf netdefs...");

    alog("done.");
    alog("Instantiating predictor...");

    if(opengl) {
        loadToNetDef(mgr, &_initNet, "iu_d16_s128_init.pb");
        loadToNetDef(mgr, &_predictNet, "iu_d16_s128_predict.pb");

//        loadToNetDef(mgr, &_initNet, "ma_95_init.pb");
//        loadToNetDef(mgr, &_predictNet, "ma_95_predict.pb");

//        loadToNetDef(mgr, &_initNet, "ronaldo_95_init.pb");
//        loadToNetDef(mgr, &_predictNet, "ronaldo_95_predict.pb");

        loadToNetDef(mgr, &_resizeInitNet,   "iu_resize_init.pb");
        loadToNetDef(mgr, &_resizePredictNet,"iu_resize_predict.pb");
        _predictor = new Predictor(_resizeInitNet, _resizePredictNet);
    } else {
        loadToNetDef(mgr, &_initNet,   "iu_cpu_init.pb");
        loadToNetDef(mgr, &_predictNet,"iu_cpu_predict.pb");

        _predictor = new Predictor(_initNet, _predictNet);
    }

    isLoad = true;
    alog("done.")
}








#include <caffe2/core/workspace.h>
#include <caffe2/utils/math.h>

#include <caffe2/mobile/contrib/opengl/core/GL.h>
#include <caffe2/mobile/contrib/opengl/core/GLLogging.h>
#include <caffe2/mobile/contrib/opengl/core/arm_neon_support.h>
#include <caffe2/mobile/contrib/opengl/operators/gl_tiling_utils.h>

void AddNoiseInput(const std::vector<caffe2::TIndex>& shape,
                   const std::string& name,
                   caffe2::Workspace* ws) {
    caffe2::CPUContext context;
    caffe2::Blob* blob = ws->CreateBlob(name);
    auto* tensor = blob->GetMutable<caffe2::TensorCPU>();
    tensor->Resize(shape);

    caffe2::math::RandGaussian<float, caffe2::CPUContext>(
            tensor->size(), 0.0f, 10.0f, tensor->mutable_data<float>(), &context);
}

template <typename T>
static double BenchGLConvolution(int input_channels,
                                 int output_channels,
                                 int kernel_width,
                                 int kernel_height,
                                 int input_width,
                                 int input_height,
                                 int input_padding,
                                 int input_stride,
                                 bool transposed,
                                 caffe2::Workspace* ws = nullptr) {
    int tile_x = 1, tile_y = 1;
    caffe2::squareFactors((input_channels + 3) / 4, tile_x, tile_y);

    gl_log(GL_LOG, "Input Tiles Factors: %d, %d\n", tile_x, tile_y);

    caffe2::Workspace localWs;
    if (!ws) {
        ws = &localWs;
    }

    AddNoiseInput(
            std::vector<caffe2::TIndex>{1, input_channels, input_height, input_width}, "X_cpu", ws);
    if (transposed) {
        AddNoiseInput(
                std::vector<caffe2::TIndex>{input_channels, output_channels, kernel_height, kernel_width},
                "W",
                ws);
    } else {
        AddNoiseInput(
                std::vector<caffe2::TIndex>{output_channels, input_channels, kernel_height, kernel_width},
                "W",
                ws);
    }
    AddNoiseInput(std::vector<caffe2::TIndex>{output_channels}, "b", ws);

    caffe2::NetDef netdef;
    {
        auto& op = *(netdef.add_op());
        op.set_type("CopyToOpenGL");
        op.add_input("X_cpu");
        op.add_output("X_gl");
        {
            auto& arg = *(op.add_arg());
            arg.set_name("tile_x");
            arg.set_i(tile_x);
        }
        {
            auto& arg = *(op.add_arg());
            arg.set_name("tile_y");
            arg.set_i(tile_y);
        }
    }

    {
        auto& op = *(netdef.add_op());
        op.set_type(transposed ? "OpenGLConvTranspose" : "OpenGLConv");
        op.add_input("X_gl");
        {
            op.add_input("W");
            op.add_input("b");
        }
        {
            auto& arg = *(op.add_arg());
            arg.set_name("order");
            arg.set_s("NCHW");
        }
        {
            auto& arg = *(op.add_arg());
            arg.set_name("kernel");
            arg.set_i(kernel_height);
        }
        {
            auto& arg = *(op.add_arg());
            arg.set_name("pad");
            arg.set_i(input_padding);
        }
        {
            auto& arg = *(op.add_arg());
            arg.set_name("stride");
            arg.set_i(input_stride);
        }
        {
            auto& arg = *(op.add_arg());
            arg.set_name("is_last");
            arg.set_i(1);
        }
        op.add_output("Y_gl");
    }

    std::vector<std::unique_ptr<caffe2::OperatorBase>> ops;

    for (auto& op : netdef.op()) {
        ops.push_back(CreateOperator(op, ws));
    }

    // Run the Copy Operator
    ops[0]->Run();

    // Make sure the tested operator is precompiled
    ops[1]->Run();
    glFinish();

    // Measure one iteration
    caffe2::Timer timer;
    timer.Start();

    ops[1]->Run();
    glFinish();

    float one_iteration = timer.MilliSeconds();

    int target_iterations = std::max((int)(1000 / one_iteration), 1);
    int warmup_iterations = std::max((int)(200 / one_iteration), 1);

    // warm up
    for (int i = 0; i < warmup_iterations; i++) {
        ops[1]->Run();
    }
    glFinish();

    timer.Start();

    int runs = target_iterations;
    for (int i = 0; i < runs; i++) {
        ops[1]->Run();
    }
    glFinish();

    const double gpuIterTime = double(timer.MilliSeconds()) / runs;

    gl_log(GL_LOG,
           "%s(%d -> %d, %dx%d - %dx%d - OpenGL) took: %.4f ms/iter\n",
           transposed ? "ConvTranspose" : "Conv",
           input_channels,
           output_channels,
           input_width,
           input_height,
           kernel_width,
           kernel_height,
           gpuIterTime);

    return gpuIterTime;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_beomseok_opencv_1face_1detect_Caffe2_nativeFaceSwap(
        JNIEnv *env,
        jobject /* this */,
        jlong input,
        jlong output,
        jint w,
        jint h
) {
    Mat &input_mat = *(Mat *) input;
    Mat &output_mat = *(Mat *) output;

//    BenchGLConvolution<float16_t>(16, 16, 3, 3, 1280, 720, 0, 1, false);

    resize(input_mat, m_8UC4_128, Size(IMG_H, IMG_W), CV_INTER_AREA);
    cvtColor(m_8UC4_128, m_8UC3_128, COLOR_BGRA2BGR);
    m_8UC3_128.convertTo(m_32FC3_128, CV_32FC3);

    memcpy(input_hwc_128, m_32FC3_128.data, 128 * 128 * IMG_C * sizeof(float));

    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 128; ++j) {
            for (int c = 0; c < IMG_C; ++c) {
                input_chw_128[128 * 128 * c +  128 * i + j] = input_hwc_128[128 * IMG_C * i + IMG_C * j + c];
            }
        }
    }

    Timer timer;
    timer.Start();

    inference(_initNet, _predictNet);

    alog("inference time : %.4f", timer.MilliSeconds());

    for (int i = 0; i < IMG_O_H; ++i) {
        for (int j = 0; j < IMG_O_W; ++j) {
            for (int c = 0; c < IMG_C; ++c) {
                output_hwc[IMG_O_W * IMG_C * i + IMG_C * j + c] = output_chw[IMG_O_H * IMG_O_W * c +  IMG_O_W * i + j];
            }
        }
    }

    m_32FC3_O = * (new Mat(IMG_O_H, IMG_O_W, CV_32FC3, output_hwc));

    m_32FC3_O.convertTo(m_8UC3_O, CV_8UC3);
    cvtColor(m_8UC3_O, output_mat, COLOR_BGR2BGRA);
}




// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, vector<Point2f> &srcTri, vector<Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform( srcTri, dstTri );

    // Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}


// Calculate Delaunay triangles for set of points
// Returns the vector of indices of 3 points for each triangle
static void calculateDelaunayTriangles(Rect rect, vector<Point2f> &points, vector< vector<int> > &delaunayTri){
    // Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);

    // Insert points into subdiv
    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
        subdiv.insert(*it);

    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point2f> pt(3);
    vector<int> ind(3);

    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        pt[0] = Point2f(t[0], t[1]);
        pt[1] = Point2f(t[2], t[3]);
        pt[2] = Point2f(t[4], t[5 ]);

        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){
            for(int j = 0; j < 3; j++)
                for(size_t k = 0; k < points.size(); k++)
                    if(abs(pt[j].x - points[k].x) < 1.0 && abs(pt[j].y - points[k].y) < 1)
                        ind[j] = k;

            delaunayTri.push_back(ind);
        }
    }
}


// Warps and alpha blends triangular regions from img1 and img2 to img
void warpTriangle(Mat &img1, Mat &img2, vector<Point2f> &t1, vector<Point2f> &t2)
{

    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);

    // Offset points by left top corner of the respective rectangles
    vector<Point2f> t1Rect, t2Rect;
    vector<Point> t2RectInt;
    for(int i = 0; i < 3; i++)
    {

        t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
        t2RectInt.push_back( Point(t2[i].x - r2.x, t2[i].y - r2.y) ); // for fillConvexPoly

    }

    // Get mask by filling triangle
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

    // Apply warpImage to small rectangular patches
    Mat img1Rect;
    img1(r1).copyTo(img1Rect);

    Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());

    applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

    multiply(img2Rect,mask, img2Rect);
    multiply(img2(r2), Scalar(1.0,1.0,1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + img2Rect;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_beomseok_opencv_1face_1detect_Caffe2_nativeSeamlessClone(
        JNIEnv *env,
        jobject /* this */,
        jlong content,
        jlong style,
        jlong output,
        jfloatArray jlandmarks
) {
    int len = env->GetArrayLength(jlandmarks);
    jfloat* landmarks = new jfloat[len];
    env->GetFloatArrayRegion(jlandmarks, 0, len, landmarks);

    vector<Point2f> points1, points2;

    for (int i = 0; i < len / 2; i = i + 2) {
        points1.push_back(Point2f(landmarks[i], landmarks[i+1]));
        points2.push_back(Point2f(landmarks[i], landmarks[i+1]));

//        alog("[%d] x : %f, y: %f", i/2, landmarks[i], landmarks[i+1]);
    }

    Mat &img1 = *(Mat *) style;
    Mat &img2 = *(Mat *) content    ;
    Mat &output_mat = *(Mat *) output;

    cvtColor(img1, img1, CV_RGBA2RGB);
    cvtColor(img2, img2, CV_RGBA2RGB);

    Mat img1Warped = img2.clone();

    //convert Mat to float data type
    img1.convertTo(img1, CV_32F);
    img1Warped.convertTo(img1Warped, CV_32F);

    // Find convex hull
    vector<Point2f> hull1;
    vector<Point2f> hull2;
    vector<int> hullIndex;

    convexHull(points2, hullIndex, false, false);

    for(int i = 0; i < hullIndex.size(); i++)
    {
        hull1.push_back(points1[hullIndex[i]]);
        hull2.push_back(points2[hullIndex[i]]);
    }

    // Find delaunay triangulation for points on the convex hull
    vector< vector<int> > dt;
    Rect rect(0, 0, img1Warped.cols, img1Warped.rows);

    calculateDelaunayTriangles(rect, hull2, dt);

    // Apply affine transformation to Delaunay triangles
    for(size_t i = 0; i < dt.size(); i++)
    {
        vector<Point2f> t1, t2;
        // Get points for img1, img2 corresponding to the triangles
        for(size_t j = 0; j < 3; j++)
        {
            t1.push_back(hull1[dt[i][j]]);
            t2.push_back(hull2[dt[i][j]]);
        }

        warpTriangle(img1, img1Warped, t1, t2);
    }

    // Calculate mask
    vector<Point> hull8U;
    for(int i = 0; i < hull2.size(); i++)
    {
        Point pt(hull2[i].x, hull2[i].y);
        hull8U.push_back(pt);
    }

    Mat mask = Mat::zeros(img2.rows, img2.cols, img2.depth());
    fillConvexPoly(mask,&hull8U[0], hull8U.size(), Scalar(255,255,255));

    // Clone seamlessly.
    Rect r = boundingRect(hull2);
    Point center = (r.tl() + r.br()) / 2;

    img1Warped.convertTo(img1Warped, CV_8UC3);
    seamlessClone(img1Warped,img2, mask, center, output_mat, NORMAL_CLONE);

    cvtColor(output_mat, output_mat, CV_RGB2RGBA);
}