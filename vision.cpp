#include <yarp/os/Bottle.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/Network.h>
#include <yarp/os/Port.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/Time.h>
#include <yarp/sig/Image.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/video.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::array<int, 6> kLandmarkIds = {30, 8, 36, 45, 48, 54};

const cv::Mat kFace3dModel = (cv::Mat_<double>(6, 3) <<
    0.0,   0.0,   0.0,
    0.0,  -63.6, -12.5,
   -43.3,  32.7, -26.0,
    43.3,  32.7, -26.0,
   -28.9, -28.9, -24.1,
    28.9, -28.9, -24.1);

std::string rfString(yarp::os::ResourceFinder& rf, const std::string& key, const std::string& fallback) {
    return rf.check(key) ? rf.find(key).asString() : fallback;
}

double rfFloat(yarp::os::ResourceFinder& rf, const std::string& key, double fallback) {
    return rf.check(key) ? rf.find(key).asFloat64() : fallback;
}

int rfInt(yarp::os::ResourceFinder& rf, const std::string& key, int fallback) {
    return rf.check(key) ? rf.find(key).asInt32() : fallback;
}

bool rfBool(yarp::os::ResourceFinder& rf, const std::string& key, bool fallback) {
    return rf.check(key) ? rf.find(key).asBool() : fallback;
}

double nowSeconds() {
    return yarp::os::Time::now();
}

bool isImageFile(const fs::path& path) {
    const std::string ext = path.extension().string();
    const std::string lower = [&]() {
        std::string out = ext;
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return out;
    }();
    return lower == ".jpg" || lower == ".jpeg" || lower == ".png";
}

double rectArea(const cv::Rect2f& rect) {
    return std::max(0.0f, rect.width) * std::max(0.0f, rect.height);
}

double iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);
    const double inter = static_cast<double>(w) * static_cast<double>(h);
    const double uni = rectArea(a) + rectArea(b) - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

double centerDistance(const cv::Rect2f& a, const cv::Rect2f& b) {
    const cv::Point2f ca(a.x + a.width * 0.5f, a.y + a.height * 0.5f);
    const cv::Point2f cb(b.x + b.width * 0.5f, b.y + b.height * 0.5f);
    return cv::norm(ca - cb);
}

double stddev(const std::deque<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double acc = 0.0;
    for (double v : values) {
        const double delta = v - mean;
        acc += delta * delta;
    }
    return std::sqrt(acc / static_cast<double>(values.size()));
}

cv::Rect2f clampRect(const cv::Rect2f& rect, const cv::Size& size) {
    const float x1 = std::clamp(rect.x, 0.0f, static_cast<float>(size.width));
    const float y1 = std::clamp(rect.y, 0.0f, static_cast<float>(size.height));
    const float x2 = std::clamp(rect.x + rect.width, 0.0f, static_cast<float>(size.width));
    const float y2 = std::clamp(rect.y + rect.height, 0.0f, static_cast<float>(size.height));
    return cv::Rect2f(x1, y1, std::max(0.0f, x2 - x1), std::max(0.0f, y2 - y1));
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

}  // namespace

class VisionAnalyzer : public yarp::os::RFModule {
public:
    bool configure(yarp::os::ResourceFinder& rf) override;
    double getPeriod() override { return rate_; }
    bool updateModule() override;
    bool respond(const yarp::os::Bottle& command, yarp::os::Bottle& reply) override;
    bool interruptModule() override;
    bool close() override;

private:
    struct FaceDetection {
        cv::Rect2f box;
        cv::Mat faceRow;
        float score{0.0f};
        int trackId{-1};
        std::string className{"face"};
        std::string faceId{"unknown"};
        double idConfidence{0.0};
    };

    struct DetectedFaceData {
        std::string faceId{"unknown"};
        int trackId{-1};
        cv::Rect2f bbox;
        double detectionScore{0.0};
        double idConfidence{0.0};
        std::string attention{"UNKNOWN"};
        std::string distance{"UNKNOWN"};
        std::string zone{"UNKNOWN"};
    };

    struct TrackState {
        int id{-1};
        cv::Rect2f box;
        float score{0.0f};
        int lostFrames{0};
        double lastSeenTs{0.0};
    };

    struct StickyIdentity {
        std::string name;
        double confidence{0.0};
        double timestamp{0.0};
    };

    struct RetryState {
        int attempts{0};
        double lastRetryTs{0.0};
    };

    struct EnvironmentState {
        int faces{0};
        int people{0};
        double motion{0.0};
        double light{0.0};
        int mutualGaze{0};
    };

    bool handleFaceNaming(const yarp::os::Bottle& command, yarp::os::Bottle& reply);

    std::vector<std::string> buildModelCandidates(const std::string& primary, const std::string& fallbackCsv) const;
    std::string resolveModelPath(const std::string& modelPath) const;
    bool initializeFaceDetector(const std::vector<std::string>& candidates);
    bool initializeFaceRecognizer(const std::string& recognizerModel);
    bool initializeFacemark(const std::string& configuredModel);

    cv::Mat yarpImageToBgr(const yarp::sig::ImageOf<yarp::sig::PixelRgb>& image);
    void writeAnnotatedImage(const cv::Mat& annotatedBgr);

    void detectQrCodes();
    void detectPeopleObj();
    void detectLight();
    void detectMotion();
    void detectMutualGaze();
    void drawAndPublishFacesView();
    void handleTargetCommand();
    void fillBottle();

    std::vector<FaceDetection> runFaceDetector(const cv::Mat& frame);
    void updateTracks(std::vector<FaceDetection>& detections, double currentTime);
    void updateFaceIdentities(std::vector<FaceDetection>& detections, const cv::Mat& frame, double currentTime);
    void refreshDetectedFaces(const std::vector<FaceDetection>& detections, double currentTime);

    bool computeEmbeddingForDetection(const cv::Mat& frame, const FaceDetection& detection, cv::Mat& embedding) const;
    bool computeEmbeddingFromImage(const cv::Mat& image, cv::Mat& embedding);
    std::pair<std::string, double> compareEmbeddings(const cv::Mat& frame, const FaceDetection& detection) const;
    void loadKnownFaces(const std::string& facesPath);

    DetectedFaceData* matchFaceToBbox(double faceX, double faceY, const std::unordered_set<int>& matchedTrackIds);
    void publishLandmarks(DetectedFaceData* faceData,
                          const cv::Vec3d& gazeDirection,
                          double pitch,
                          double yaw,
                          double roll,
                          double cosAngle,
                          const std::string& attention,
                          int isTalking,
                          double timeInView);

    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> imgInPort_;
    yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb>> faceDetectionImgPort_;
    yarp::os::BufferedPort<yarp::os::Bottle> targetCmdPort_;
    yarp::os::BufferedPort<yarp::os::Bottle> targetBoxPort_;
    yarp::os::BufferedPort<yarp::os::Bottle> qrPort_;
    yarp::os::Port visionFeaturesPort_;
    yarp::os::Port landmarksPort_;
    yarp::os::Port handlePort_;

    yarp::os::Bottle visionFeaturesBottle_;
    yarp::os::Bottle landmarksBottle_;

    std::string name_{"alwayson/vision"};
    double rate_{0.05};
    int imgWidth_{640};
    int imgHeight_{480};
    const int defaultWidth_{640};
    const int defaultHeight_{480};
    std::string landmarkModelPath_{"face_landmarker.task"};
    std::string facesPath_;
    fs::path moduleRoot_{fs::current_path()};

    bool track_{true};
    bool identifyFaces_{true};
    bool verbose_{false};
    bool debug_{false};
    bool autoDownloadModel_{false};
    float confThreshold_{0.7f};
    double tolerance_{0.62};
    double identityStickySec_{1.5};
    int maxEnrollSamples_{5};
    double unknownRetryIntervalSec_{1.0};
    int unknownRetryMaxAttempts_{3};
    double mutualGazeThreshold_{10.0};
    double maxFaceMatchDistance_{100.0};
    int lostTrackBuffer_{120};
    int nextTrackId_{1};

    int mouthBufferSize_{10};
    double talkingThreshold_{0.012};

    int currentTargetTrackId_{-1};
    double currentTargetIps_{0.0};

    std::string activeQrValue_;
    int qrMissingScans_{0};
    int qrLostResetScans_{3};
    int qrSeenFrames_{0};

    double timestamp_{0.0};
    double facesSyncInfo_{0.0};
    double execTime_{0.15};

    EnvironmentState envState_;
    cv::Mat image_;
    cv::Mat lastFrame_;
    std::deque<cv::Mat> optFlowBuf_;

    cv::QRCodeDetector qrDetector_;
    cv::Ptr<cv::FaceDetectorYN> yunetDetector_;
    cv::CascadeClassifier cascadeDetector_;
    cv::Ptr<cv::FaceRecognizerSF> faceRecognizerSf_;
    cv::Ptr<cv::face::Facemark> facemark_;
    bool detectorUsesYunet_{false};
    bool recognizerUsesSFace_{false};

    std::vector<TrackState> trackStates_;
    std::vector<FaceDetection> currentDetections_;
    std::vector<DetectedFaceData> detectedFaces_;

    std::unordered_map<std::string, std::vector<cv::Mat>> knownFaces_;
    std::unordered_map<int, std::pair<std::string, double>> trackedFaces_;
    std::unordered_map<int, StickyIdentity> lastKnownIdentity_;
    std::unordered_map<int, RetryState> unknownRetryState_;
    mutable std::mutex faceIdentityMutex_;

    std::unordered_map<int, std::deque<double>> mouthMotionHistory_;
    std::unordered_map<int, double> lastSeenTrack_;
    std::unordered_map<int, double> firstSeenTrack_;
};

std::vector<std::string> VisionAnalyzer::buildModelCandidates(const std::string& primary, const std::string& fallbackCsv) const {
    std::vector<std::string> candidates;
    if (!primary.empty()) {
        candidates.push_back(primary);
    }

    std::size_t start = 0;
    while (start < fallbackCsv.size()) {
        const std::size_t end = fallbackCsv.find(',', start);
        std::string token = fallbackCsv.substr(start, end == std::string::npos ? std::string::npos : end - start);
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) { return std::isspace(c) != 0; }), token.end());
        if (!token.empty() && std::find(candidates.begin(), candidates.end(), token) == candidates.end()) {
            candidates.push_back(token);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    const std::vector<std::string> builtIns = {
        "face_detection_yunet_2023mar.onnx",
        "face_detection_yunet.onnx",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    };
    for (const auto& candidate : builtIns) {
        if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
            candidates.push_back(candidate);
        }
    }

    return candidates;
}

std::string VisionAnalyzer::resolveModelPath(const std::string& modelPath) const {
    if (modelPath.empty()) {
        return {};
    }

    const fs::path path(modelPath);
    const std::vector<fs::path> candidates = {
        path,
        moduleRoot_ / path,
        moduleRoot_ / "model" / path,
        fs::current_path() / path
    };

    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (fs::exists(candidate, ec)) {
            return fs::absolute(candidate, ec).string();
        }
    }

    return {};
}

bool VisionAnalyzer::initializeFaceDetector(const std::vector<std::string>& candidates) {
    for (const auto& candidate : candidates) {
        const std::string resolved = resolveModelPath(candidate);
        if (resolved.empty()) {
            continue;
        }

        const std::string lower = toLower(fs::path(resolved).extension().string());
        try {
            if (lower == ".onnx") {
                yunetDetector_ = cv::FaceDetectorYN::create(resolved, "", cv::Size(defaultWidth_, defaultHeight_), confThreshold_, 0.3f, 5000);
                detectorUsesYunet_ = true;
                std::cout << "[INFO] Loaded YuNet face detector: " << resolved << '\n';
                return true;
            }
            if (lower == ".xml" && cascadeDetector_.load(resolved)) {
                detectorUsesYunet_ = false;
                std::cout << "[INFO] Loaded cascade face detector: " << resolved << '\n';
                return true;
            }
            if (lower == ".pt") {
                std::cerr << "[WARNING] PyTorch face models are not directly usable from this C++ module: " << resolved << '\n';
            }
        } catch (const std::exception& e) {
            std::cerr << "[WARNING] Failed to initialize detector from " << resolved << ": " << e.what() << '\n';
        }
    }

    std::cerr << "[ERROR] Could not initialize any face detector model.\n";
    return false;
}

bool VisionAnalyzer::initializeFaceRecognizer(const std::string& recognizerModel) {
    const std::string resolved = resolveModelPath(recognizerModel);
    if (resolved.empty()) {
        recognizerUsesSFace_ = false;
        std::cerr << "[WARNING] Face recognizer model not found, falling back to descriptor matching.\n";
        return true;
    }

    try {
        faceRecognizerSf_ = cv::FaceRecognizerSF::create(resolved, "");
        recognizerUsesSFace_ = true;
        std::cout << "[INFO] Loaded SFace recognizer: " << resolved << '\n';
        return true;
    } catch (const std::exception& e) {
        recognizerUsesSFace_ = false;
        std::cerr << "[WARNING] Failed to initialize face recognizer from " << resolved << ": " << e.what() << '\n';
        return true;
    }
}

bool VisionAnalyzer::initializeFacemark(const std::string& configuredModel) {
    if (toLower(fs::path(configuredModel).extension().string()) == ".task") {
        std::cerr << "[WARNING] MediaPipe .task landmarker assets are replaced here with OpenCV FacemarkLBF in C++.\n";
    }

    std::vector<std::string> candidates;
    if (!configuredModel.empty()) {
        candidates.push_back(configuredModel);
    }
    candidates.push_back("lbfmodel.yaml");
    candidates.push_back("lbfmodel.yaml.gz");

    for (const auto& model : candidates) {
        const std::string resolved = resolveModelPath(model);
        if (resolved.empty()) {
            continue;
        }

        try {
            facemark_ = cv::face::FacemarkLBF::create();
            facemark_->loadModel(resolved);
            std::cout << "[INFO] Loaded facemark model: " << resolved << '\n';
            return true;
        } catch (const std::exception& e) {
            facemark_.release();
            std::cerr << "[WARNING] Failed to initialize facemark model from " << resolved << ": " << e.what() << '\n';
        }
    }

    std::cerr << "[WARNING] No OpenCV facemark model available. Gaze/talking fallback will publish UNKNOWN attention.\n";
    return true;
}

cv::Mat VisionAnalyzer::yarpImageToBgr(const yarp::sig::ImageOf<yarp::sig::PixelRgb>& image) {
    if (image.width() != imgWidth_ || image.height() != imgHeight_) {
        imgWidth_ = image.width();
        imgHeight_ = image.height();
        std::cout << "[INFO] Input image size changed to " << imgWidth_ << "x" << imgHeight_ << '\n';
    }

    cv::Mat rgb(image.height(), image.width(), CV_8UC3,
                const_cast<unsigned char*>(image.getRawImage()),
                static_cast<std::size_t>(image.getRowSize()));
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    if (imgWidth_ != defaultWidth_ || imgHeight_ != defaultHeight_) {
        cv::resize(bgr, bgr, cv::Size(defaultWidth_, defaultHeight_));
    }
    return bgr;
}

void VisionAnalyzer::writeAnnotatedImage(const cv::Mat& annotatedBgr) {
    cv::Mat annotatedRgb;
    cv::cvtColor(annotatedBgr, annotatedRgb, cv::COLOR_BGR2RGB);
    auto& out = faceDetectionImgPort_.prepare();
    out.resize(annotatedRgb.cols, annotatedRgb.rows);
    cv::Mat outMat(annotatedRgb.rows, annotatedRgb.cols, CV_8UC3, out.getRawImage(), static_cast<std::size_t>(out.getRowSize()));
    annotatedRgb.copyTo(outMat);
    faceDetectionImgPort_.write();
}

bool VisionAnalyzer::configure(yarp::os::ResourceFinder& rf) {
    name_ = rfString(rf, "name", name_);
    rate_ = rfFloat(rf, "rate", 0.05);
    imgWidth_ = rfInt(rf, "width", 640);
    imgHeight_ = rfInt(rf, "height", 480);
    landmarkModelPath_ = rfString(rf, "model", "face_landmarker.task");

    track_ = rfBool(rf, "track", true);
    identifyFaces_ = rfBool(rf, "identify_faces", true);
    verbose_ = rfBool(rf, "verbose_yolo", false);
    debug_ = rfBool(rf, "debug", false);
    autoDownloadModel_ = rfBool(rf, "auto_download_model", false);
    confThreshold_ = static_cast<float>(rfFloat(rf, "conf_threshold", 0.7));
    tolerance_ = rfFloat(rf, "id_tolerance", 0.62);
    identityStickySec_ = rfFloat(rf, "identity_sticky_sec", 1.5);
    maxEnrollSamples_ = rfInt(rf, "id_enroll_samples", 5);
    unknownRetryIntervalSec_ = rfFloat(rf, "unknown_retry_interval_sec", 1.0);
    unknownRetryMaxAttempts_ = rfInt(rf, "unknown_retry_max_attempts", 3);
    qrLostResetScans_ = rfInt(rf, "qr_lost_reset_scans", 3);

    facesPath_ = rfString(rf, "faces_path", (moduleRoot_ / "faces").string());
    const std::string detectorPrimary = rfString(rf, "yolo_model", "face_detection_yunet_2023mar.onnx");
    const std::string detectorFallbacks = rfString(rf, "fallback_models", "face_detection_yunet.onnx");
    const std::string recognizerModel = rfString(rf, "face_recognizer_model", "face_recognition_sface_2021dec.onnx");
    const std::string facemarkModel = rfString(rf, "facemark_model", landmarkModelPath_);
    const std::string rpcName = rfString(rf, "rpc_name", name_ + "/rpc");

    std::cout << "IMAGE W: " << imgWidth_ << '\n';
    std::cout << "IMAGE H: " << imgHeight_ << '\n';
    std::cout << "RATE: " << rate_ << '\n';
    std::cout << "MODEL PATH: " << landmarkModelPath_ << '\n';

    imgInPort_.open("/" + name_ + "/img:i");
    visionFeaturesPort_.open("/" + name_ + "/features:o");
    landmarksPort_.open("/" + name_ + "/landmarks:o");
    targetCmdPort_.open("/" + name_ + "/targetCmd:i");
    targetBoxPort_.open("/" + name_ + "/targetBox:o");
    faceDetectionImgPort_.open("/" + name_ + "/faces_view:o");
    qrPort_.open("/" + name_ + "/qr:o");
    handlePort_.open("/" + rpcName);
    attach(handlePort_);

    if (!initializeFaceDetector(buildModelCandidates(detectorPrimary, detectorFallbacks))) {
        return false;
    }
    if (!initializeFaceRecognizer(recognizerModel)) {
        return false;
    }
    if (!initializeFacemark(facemarkModel)) {
        return false;
    }

    if (identifyFaces_) {
        loadKnownFaces(facesPath_);
        std::cout << "[INFO] Face ID enabled with " << knownFaces_.size() << " identities from " << facesPath_ << '\n';
    }

    std::cout << "[INFO] Start processing video (vision monolith C++17)\n";
    return true;
}

bool VisionAnalyzer::updateModule() {
    visionFeaturesBottle_.clear();
    landmarksBottle_.clear();

    const bool hasFeaturesSubscriber = visionFeaturesPort_.getOutputCount() > 0;
    const bool hasLandmarksSubscriber = landmarksPort_.getOutputCount() > 0;
    const bool hasTargetSubscriber = targetBoxPort_.getOutputCount() > 0;
    const bool hasViewSubscriber = faceDetectionImgPort_.getOutputCount() > 0;
    const bool hasQrSubscriber = qrPort_.getOutputCount() > 0;

    if (hasFeaturesSubscriber || hasLandmarksSubscriber || hasTargetSubscriber || hasViewSubscriber || hasQrSubscriber) {
        auto* imagePtr = imgInPort_.read(true);
        while (auto* newest = imgInPort_.read(false)) {
            imagePtr = newest;
        }

        if (imagePtr != nullptr) {
            image_ = yarpImageToBgr(*imagePtr);
            if (hasQrSubscriber) {
                detectQrCodes();
            }
            detectPeopleObj();
            detectMutualGaze();
            detectLight();
            detectMotion();
            if (hasViewSubscriber) {
                drawAndPublishFacesView();
            }
        }

        timestamp_ = nowSeconds();

        if (hasFeaturesSubscriber) {
            fillBottle();
            visionFeaturesPort_.write(visionFeaturesBottle_);
        }
        if (hasLandmarksSubscriber) {
            landmarksPort_.write(landmarksBottle_);
        }
    }

    handleTargetCommand();
    return true;
}

void VisionAnalyzer::detectQrCodes() {
    if (image_.empty()) {
        return;
    }

    ++qrSeenFrames_;
    if (qrSeenFrames_ % 10 != 0) {
        return;
    }

    cv::Mat gray;
    cv::cvtColor(image_, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point> points;
    std::string rawValue = qrDetector_.detectAndDecode(gray, points);

    if (rawValue.empty()) {
        ++qrMissingScans_;
        if (qrMissingScans_ >= qrLostResetScans_) {
            activeQrValue_.clear();
        }
        return;
    }

    std::string value = toLower(rawValue);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (value.empty()) {
        return;
    }

    qrMissingScans_ = 0;
    if (value == activeQrValue_) {
        return;
    }

    auto& bottle = qrPort_.prepare();
    bottle.clear();
    bottle.addString(value);
    qrPort_.write();
    activeQrValue_ = value;
}

std::vector<VisionAnalyzer::FaceDetection> VisionAnalyzer::runFaceDetector(const cv::Mat& frame) {
    std::vector<FaceDetection> detections;

    if (frame.empty()) {
        return detections;
    }

    if (detectorUsesYunet_ && yunetDetector_) {
        cv::Mat faces;
        yunetDetector_->setInputSize(frame.size());
        yunetDetector_->detect(frame, faces);
        for (int row = 0; row < faces.rows; ++row) {
            FaceDetection detection;
            const float x = faces.at<float>(row, 0);
            const float y = faces.at<float>(row, 1);
            const float w = faces.at<float>(row, 2);
            const float h = faces.at<float>(row, 3);
            detection.score = faces.cols > 14 ? faces.at<float>(row, 14) : 1.0f;
            if (detection.score < confThreshold_) {
                continue;
            }

            detection.faceRow = faces.row(row).clone();
            const float expandW = w * 0.10f;
            const float expandH = h * 0.10f;
            detection.box = clampRect(cv::Rect2f(x - expandW, y - expandH, w + 2.0f * expandW, h + 2.0f * expandH), frame.size());
            detections.push_back(detection);
        }
        return detections;
    }

    if (!cascadeDetector_.empty()) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        cascadeDetector_.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(40, 40));
        for (const auto& face : faces) {
            FaceDetection detection;
            detection.score = 1.0f;
            const float expandW = static_cast<float>(face.width) * 0.10f;
            const float expandH = static_cast<float>(face.height) * 0.10f;
            detection.box = clampRect(cv::Rect2f(static_cast<float>(face.x) - expandW,
                                                 static_cast<float>(face.y) - expandH,
                                                 static_cast<float>(face.width) + 2.0f * expandW,
                                                 static_cast<float>(face.height) + 2.0f * expandH), frame.size());
            detections.push_back(detection);
        }
    }

    return detections;
}

void VisionAnalyzer::updateTracks(std::vector<FaceDetection>& detections, double currentTime) {
    if (!track_) {
        return;
    }

    struct CandidateMatch {
        int trackIndex{-1};
        int detectionIndex{-1};
        double score{-std::numeric_limits<double>::infinity()};
    };

    std::vector<CandidateMatch> candidates;
    for (int t = 0; t < static_cast<int>(trackStates_.size()); ++t) {
        for (int d = 0; d < static_cast<int>(detections.size()); ++d) {
            const double overlap = iou(trackStates_[t].box, detections[d].box);
            const double dist = centerDistance(trackStates_[t].box, detections[d].box);
            if (overlap >= 0.15 || dist <= maxFaceMatchDistance_) {
                candidates.push_back({t, d, overlap - 0.001 * dist});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const CandidateMatch& a, const CandidateMatch& b) {
        return a.score > b.score;
    });

    std::vector<bool> detectionMatched(detections.size(), false);
    std::vector<bool> trackMatched(trackStates_.size(), false);

    for (const auto& candidate : candidates) {
        if (trackMatched[candidate.trackIndex] || detectionMatched[candidate.detectionIndex]) {
            continue;
        }
        TrackState& track = trackStates_[candidate.trackIndex];
        FaceDetection& detection = detections[candidate.detectionIndex];
        track.box = detection.box;
        track.score = detection.score;
        track.lostFrames = 0;
        track.lastSeenTs = currentTime;
        detection.trackId = track.id;
        trackMatched[candidate.trackIndex] = true;
        detectionMatched[candidate.detectionIndex] = true;
    }

    for (std::size_t i = 0; i < trackStates_.size(); ++i) {
        if (!trackMatched[i]) {
            ++trackStates_[i].lostFrames;
        }
    }

    trackStates_.erase(std::remove_if(trackStates_.begin(), trackStates_.end(), [&](const TrackState& track) {
        return track.lostFrames > lostTrackBuffer_;
    }), trackStates_.end());

    for (std::size_t i = 0; i < detections.size(); ++i) {
        if (detectionMatched[i]) {
            continue;
        }
        TrackState track;
        track.id = nextTrackId_++;
        track.box = detections[i].box;
        track.score = detections[i].score;
        track.lostFrames = 0;
        track.lastSeenTs = currentTime;
        trackStates_.push_back(track);
        detections[i].trackId = track.id;
    }
}

bool VisionAnalyzer::computeEmbeddingForDetection(const cv::Mat& frame, const FaceDetection& detection, cv::Mat& embedding) const {
    const cv::Rect cropRect(cvRound(detection.box.x), cvRound(detection.box.y),
                            cvRound(detection.box.width), cvRound(detection.box.height));
    const cv::Rect bounded = cropRect & cv::Rect(0, 0, frame.cols, frame.rows);
    if (bounded.width <= 0 || bounded.height <= 0) {
        return false;
    }

    if (recognizerUsesSFace_ && detectorUsesYunet_ && faceRecognizerSf_ && !detection.faceRow.empty()) {
        try {
            cv::Mat aligned;
            faceRecognizerSf_->alignCrop(frame, detection.faceRow, aligned);
            faceRecognizerSf_->feature(aligned, embedding);
            return !embedding.empty();
        } catch (const std::exception&) {
        }
    }

    cv::Mat crop = frame(bounded).clone();
    cv::Mat gray;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(48, 48));
    cv::equalizeHist(gray, gray);
    gray.convertTo(embedding, CV_32F, 1.0 / 255.0);
    embedding = embedding.reshape(1, 1);
    const double n = cv::norm(embedding);
    if (n > 1e-9) {
        embedding /= n;
    }
    return true;
}

bool VisionAnalyzer::computeEmbeddingFromImage(const cv::Mat& image, cv::Mat& embedding) {
    if (image.empty()) {
        return false;
    }

    const auto detections = runFaceDetector(image);
    if (!detections.empty()) {
        auto best = std::max_element(detections.begin(), detections.end(), [](const FaceDetection& a, const FaceDetection& b) {
            return rectArea(a.box) < rectArea(b.box);
        });
        return computeEmbeddingForDetection(image, *best, embedding);
    }

    FaceDetection whole;
    whole.box = cv::Rect2f(0.0f, 0.0f, static_cast<float>(image.cols), static_cast<float>(image.rows));
    return computeEmbeddingForDetection(image, whole, embedding);
}

std::pair<std::string, double> VisionAnalyzer::compareEmbeddings(const cv::Mat& frame, const FaceDetection& detection) const {
    cv::Mat unknownEmbedding;
    if (!computeEmbeddingForDetection(frame, detection, unknownEmbedding)) {
        return {"recognizing", 0.0};
    }

    std::unordered_map<std::string, std::vector<cv::Mat>> snapshot;
    {
        std::lock_guard<std::mutex> lock(faceIdentityMutex_);
        snapshot = knownFaces_;
    }
    if (snapshot.empty()) {
        return {"unknown", 0.0};
    }

    std::string bestName;
    double bestEffectiveDistance = std::numeric_limits<double>::infinity();

    for (const auto& entry : snapshot) {
        if (entry.second.empty()) {
            continue;
        }

        std::vector<double> distances;
        distances.reserve(entry.second.size());
        for (const auto& sample : entry.second) {
            if (sample.empty() || sample.total() != unknownEmbedding.total()) {
                continue;
            }

            double similarity = 0.0;
            if (recognizerUsesSFace_ && faceRecognizerSf_ && sample.cols == unknownEmbedding.cols) {
                similarity = faceRecognizerSf_->match(sample, unknownEmbedding, cv::FaceRecognizerSF::FR_COSINE);
            } else {
                similarity = sample.dot(unknownEmbedding);
            }

            distances.push_back(1.0 - similarity);
        }

        if (distances.empty()) {
            continue;
        }

        std::sort(distances.begin(), distances.end());
        const double minDistance = distances.front();
        const double medianDistance = distances[distances.size() / 2];
        const double effectiveDistance = 0.5 * (minDistance + medianDistance);

        if (minDistance <= tolerance_ && effectiveDistance < bestEffectiveDistance) {
            bestEffectiveDistance = effectiveDistance;
            bestName = entry.first;
        }
    }

    if (!bestName.empty()) {
        return {bestName, 1.0 - bestEffectiveDistance};
    }
    return {"unknown", 0.0};
}

void VisionAnalyzer::updateFaceIdentities(std::vector<FaceDetection>& detections, const cv::Mat& frame, double currentTime) {
    if (!identifyFaces_ || !track_) {
        return;
    }

    std::unordered_set<int> currentIds;
    for (const auto& detection : detections) {
        if (detection.trackId >= 0) {
            currentIds.insert(detection.trackId);
        }
    }

    {
        std::lock_guard<std::mutex> lock(faceIdentityMutex_);
        std::vector<int> lostIds;
        for (const auto& entry : trackedFaces_) {
            if (currentIds.find(entry.first) == currentIds.end()) {
                lostIds.push_back(entry.first);
            }
        }
        for (int id : lostIds) {
            const auto cached = trackedFaces_.find(id);
            if (cached != trackedFaces_.end()) {
                if (cached->second.first != "recognizing" && cached->second.first != "unknown") {
                    lastKnownIdentity_[id] = StickyIdentity{cached->second.first, cached->second.second, currentTime};
                }
                trackedFaces_.erase(cached);
            }
            unknownRetryState_.erase(id);
        }

        std::vector<int> staleIds;
        for (const auto& entry : lastKnownIdentity_) {
            if (currentTime - entry.second.timestamp > identityStickySec_) {
                staleIds.push_back(entry.first);
            }
        }
        for (int id : staleIds) {
            lastKnownIdentity_.erase(id);
        }
    }

    for (auto& detection : detections) {
        if (detection.trackId < 0) {
            continue;
        }

        std::pair<std::string, double> trackedEntry;
        bool hasTracked = false;
        {
            std::lock_guard<std::mutex> lock(faceIdentityMutex_);
            const auto it = trackedFaces_.find(detection.trackId);
            if (it != trackedFaces_.end()) {
                trackedEntry = it->second;
                hasTracked = true;
            }
        }

        if (!hasTracked) {
            auto [faceId, confidence] = compareEmbeddings(frame, detection);
            if ((faceId == "unknown" || faceId == "recognizing")) {
                std::lock_guard<std::mutex> lock(faceIdentityMutex_);
                const auto sticky = lastKnownIdentity_.find(detection.trackId);
                if (sticky != lastKnownIdentity_.end() && (currentTime - sticky->second.timestamp) <= identityStickySec_) {
                    faceId = sticky->second.name;
                    confidence = sticky->second.confidence;
                }
            }

            std::lock_guard<std::mutex> lock(faceIdentityMutex_);
            if (faceId == "unknown" || faceId == "recognizing") {
                trackedFaces_[detection.trackId] = {"unknown", 0.0};
                unknownRetryState_[detection.trackId] = RetryState{1, currentTime};
            } else {
                trackedFaces_[detection.trackId] = {faceId, confidence};
                lastKnownIdentity_[detection.trackId] = StickyIdentity{faceId, confidence, currentTime};
                unknownRetryState_.erase(detection.trackId);
            }
        } else if (trackedEntry.first == "unknown" || trackedEntry.first == "recognizing") {
            RetryState retry;
            {
                std::lock_guard<std::mutex> lock(faceIdentityMutex_);
                retry = unknownRetryState_[detection.trackId];
            }
            const bool shouldRetry = retry.attempts < unknownRetryMaxAttempts_ &&
                                     (currentTime - retry.lastRetryTs) >= unknownRetryIntervalSec_;

            if (shouldRetry) {
                auto [faceId, confidence] = compareEmbeddings(frame, detection);
                std::lock_guard<std::mutex> lock(faceIdentityMutex_);
                if (faceId == "unknown" || faceId == "recognizing") {
                    trackedFaces_[detection.trackId] = {"unknown", 0.0};
                    unknownRetryState_[detection.trackId] = RetryState{retry.attempts + 1, currentTime};
                } else {
                    trackedFaces_[detection.trackId] = {faceId, confidence};
                    lastKnownIdentity_[detection.trackId] = StickyIdentity{faceId, confidence, currentTime};
                    unknownRetryState_.erase(detection.trackId);
                }
            }
        } else {
            std::lock_guard<std::mutex> lock(faceIdentityMutex_);
            lastKnownIdentity_[detection.trackId] = StickyIdentity{trackedEntry.first, trackedEntry.second, currentTime};
        }

        std::lock_guard<std::mutex> lock(faceIdentityMutex_);
        const auto it = trackedFaces_.find(detection.trackId);
        if (it != trackedFaces_.end()) {
            detection.faceId = it->second.first;
            detection.idConfidence = it->second.second;
        }
    }
}

void VisionAnalyzer::refreshDetectedFaces(const std::vector<FaceDetection>& detections, double currentTime) {
    detectedFaces_.clear();
    int faceCount = 0;

    for (const auto& detection : detections) {
        if (detection.className != "face" || detection.score <= 0.5f) {
            continue;
        }

        ++faceCount;
        DetectedFaceData faceData;
        faceData.faceId = detection.faceId;
        faceData.trackId = detection.trackId;
        faceData.bbox = detection.box;
        faceData.detectionScore = detection.score;
        faceData.idConfidence = detection.idConfidence;
        detectedFaces_.push_back(faceData);
    }

    envState_.faces = faceCount;
    facesSyncInfo_ = currentTime;
}

void VisionAnalyzer::detectPeopleObj() {
    if (image_.empty()) {
        return;
    }

    cv::Mat frame = image_.clone();
    lastFrame_ = frame.clone();
    const double currentTime = nowSeconds();

    std::vector<FaceDetection> detections = runFaceDetector(frame);
    updateTracks(detections, currentTime);
    updateFaceIdentities(detections, frame, currentTime);

    currentDetections_ = detections;
    refreshDetectedFaces(currentDetections_, currentTime);
}

VisionAnalyzer::DetectedFaceData* VisionAnalyzer::matchFaceToBbox(double faceX, double faceY, const std::unordered_set<int>& matchedTrackIds) {
    DetectedFaceData* bestMatch = nullptr;
    double bestDistance = std::numeric_limits<double>::infinity();

    for (auto& faceData : detectedFaces_) {
        if (faceData.trackId >= 0 && matchedTrackIds.find(faceData.trackId) != matchedTrackIds.end()) {
            continue;
        }

        const double bboxCenterX = faceData.bbox.x + faceData.bbox.width * 0.5;
        const double bboxCenterY = faceData.bbox.y + faceData.bbox.height * 0.5;
        const double distance = std::hypot(faceX - bboxCenterX, faceY - bboxCenterY);

        if (distance < bestDistance) {
            bestDistance = distance;
            bestMatch = &faceData;
        } else if (distance == bestDistance && bestMatch != nullptr) {
            if (rectArea(faceData.bbox) > rectArea(bestMatch->bbox)) {
                bestMatch = &faceData;
            }
        }
    }

    if (bestDistance > maxFaceMatchDistance_) {
        return nullptr;
    }
    return bestMatch;
}

void VisionAnalyzer::publishLandmarks(DetectedFaceData* faceData,
                                      const cv::Vec3d& gazeDirection,
                                      double pitch,
                                      double yaw,
                                      double roll,
                                      double cosAngle,
                                      const std::string& attention,
                                      int isTalking,
                                      double timeInView) {
    yarp::os::Bottle faceBottle;

    if (faceData != nullptr) {
        faceBottle.addString("face_id");
        faceBottle.addString(faceData->faceId);
        faceBottle.addString("track_id");
        faceBottle.addInt32(faceData->trackId);

        auto& bboxBottle = faceBottle.addList();
        bboxBottle.addString("bbox");
        bboxBottle.addFloat64(faceData->bbox.x);
        bboxBottle.addFloat64(faceData->bbox.y);
        bboxBottle.addFloat64(faceData->bbox.width);
        bboxBottle.addFloat64(faceData->bbox.height);

        const double cx = faceData->bbox.x + faceData->bbox.width * 0.5;
        const double cy = faceData->bbox.y + faceData->bbox.height * 0.5;
        const double cxn = std::clamp(cx / static_cast<double>(defaultWidth_), 0.0, 1.0);
        const double cyn = std::clamp(cy / static_cast<double>(defaultHeight_), 0.0, 1.0);
        (void)cyn;

        if (cxn < 0.2) {
            faceData->zone = "FAR_LEFT";
        } else if (cxn < 0.4) {
            faceData->zone = "LEFT";
        } else if (cxn < 0.6) {
            faceData->zone = "CENTER";
        } else if (cxn < 0.8) {
            faceData->zone = "RIGHT";
        } else {
            faceData->zone = "FAR_RIGHT";
        }

        const double hNorm = faceData->bbox.height / static_cast<double>(defaultHeight_);
        if (hNorm > 0.4) {
            faceData->distance = "SO_CLOSE";
        } else if (hNorm > 0.2) {
            faceData->distance = "CLOSE";
        } else if (hNorm > 0.1) {
            faceData->distance = "FAR";
        } else {
            faceData->distance = "VERY_FAR";
        }

        faceData->attention = attention;
        faceBottle.addString("zone");
        faceBottle.addString(faceData->zone);
        faceBottle.addString("distance");
        faceBottle.addString(faceData->distance);
    } else {
        faceBottle.addString("face_id");
        faceBottle.addString("unmatched");
        faceBottle.addString("track_id");
        faceBottle.addInt32(-1);

        auto& bboxBottle = faceBottle.addList();
        bboxBottle.addString("bbox");
        bboxBottle.addFloat64(0.0);
        bboxBottle.addFloat64(0.0);
        bboxBottle.addFloat64(0.0);
        bboxBottle.addFloat64(0.0);

        faceBottle.addString("zone");
        faceBottle.addString("UNKNOWN");
        faceBottle.addString("distance");
        faceBottle.addString("UNKNOWN");
    }

    auto& gazeBottle = faceBottle.addList();
    gazeBottle.addString("gaze_direction");
    gazeBottle.addFloat64(gazeDirection[0]);
    gazeBottle.addFloat64(gazeDirection[1]);
    gazeBottle.addFloat64(gazeDirection[2]);

    faceBottle.addString("pitch");
    faceBottle.addFloat64(pitch);
    faceBottle.addString("yaw");
    faceBottle.addFloat64(yaw);
    faceBottle.addString("roll");
    faceBottle.addFloat64(roll);
    faceBottle.addString("cos_angle");
    faceBottle.addFloat64(cosAngle);
    faceBottle.addString("attention");
    faceBottle.addString(attention);
    faceBottle.addString("is_talking");
    faceBottle.addInt32(isTalking);
    faceBottle.addString("time_in_view");
    faceBottle.addFloat64(timeInView);

    auto& dst = landmarksBottle_.addList();
    dst.copy(faceBottle);
}

void VisionAnalyzer::detectMutualGaze() {
    envState_.mutualGaze = 0;
    const double currentTime = nowSeconds();

    if (detectedFaces_.empty()) {
        return;
    }

    if (!facemark_) {
        for (auto& faceData : detectedFaces_) {
            if (faceData.trackId >= 0) {
                if (firstSeenTrack_.find(faceData.trackId) == firstSeenTrack_.end()) {
                    firstSeenTrack_[faceData.trackId] = currentTime;
                }
                lastSeenTrack_[faceData.trackId] = currentTime;
            }
            const double timeInView = faceData.trackId >= 0 ? currentTime - firstSeenTrack_[faceData.trackId] : 0.0;
            publishLandmarks(&faceData, cv::Vec3d(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, "UNKNOWN", 0, timeInView);
        }
        return;
    }

    std::vector<cv::Rect> faceRects;
    faceRects.reserve(detectedFaces_.size());
    for (const auto& faceData : detectedFaces_) {
        faceRects.emplace_back(cvRound(faceData.bbox.x), cvRound(faceData.bbox.y),
                               cvRound(faceData.bbox.width), cvRound(faceData.bbox.height));
    }

    std::vector<std::vector<cv::Point2f>> landmarks;
    const bool ok = facemark_->fit(image_, faceRects, landmarks);
    std::unordered_set<int> matchedTrackIds;

    if (ok) {
        for (const auto& faceLandmarks : landmarks) {
            if (faceLandmarks.size() < 67) {
                continue;
            }

            std::vector<cv::Point2d> face2d;
            face2d.reserve(kLandmarkIds.size());
            for (int idx : kLandmarkIds) {
                face2d.emplace_back(faceLandmarks[idx].x, faceLandmarks[idx].y);
            }

            const double faceCenterX = std::accumulate(face2d.begin(), face2d.end(), 0.0, [](double acc, const cv::Point2d& p) {
                return acc + p.x;
            }) / static_cast<double>(face2d.size());
            const double faceCenterY = std::accumulate(face2d.begin(), face2d.end(), 0.0, [](double acc, const cv::Point2d& p) {
                return acc + p.y;
            }) / static_cast<double>(face2d.size());

            cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
                static_cast<double>(image_.cols), 0.0, static_cast<double>(image_.cols) / 2.0,
                0.0, static_cast<double>(image_.cols), static_cast<double>(image_.rows) / 2.0,
                0.0, 0.0, 1.0);
            cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
            cv::Mat rvec, tvec;
            if (!cv::solvePnP(kFace3dModel, face2d, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE)) {
                continue;
            }

            cv::Mat rmat;
            cv::Rodrigues(rvec, rmat);
            cv::Mat mtxR, mtxQ;
            const cv::Vec3d angles = cv::RQDecomp3x3(rmat, mtxR, mtxQ);
            const double pitch = angles[0];
            const double yaw = angles[1];
            const double roll = angles[2];

            const cv::Mat faceForwardMat = rmat * (cv::Mat_<double>(3, 1) << 0.0, 0.0, -1.0);
            cv::Vec3d faceForward(faceForwardMat.at<double>(0), faceForwardMat.at<double>(1), faceForwardMat.at<double>(2));
            const cv::Vec3d cameraForward(0.0, 0.0, 1.0);
            const double cosAngle = faceForward.dot(cameraForward) / (cv::norm(faceForward) * cv::norm(cameraForward));

            std::string attention = "AWAY";
            if (cosAngle > 0.90) {
                attention = "MUTUAL_GAZE";
                ++envState_.mutualGaze;
            } else if (cosAngle > 0.7) {
                attention = "NEAR_GAZE";
            }

            DetectedFaceData* matchedFace = matchFaceToBbox(faceCenterX, faceCenterY, matchedTrackIds);
            if (matchedFace != nullptr && matchedFace->trackId >= 0) {
                matchedTrackIds.insert(matchedFace->trackId);
            }

            int isTalking = 0;
            if (faceLandmarks.size() > 66) {
                const cv::Point2f& upperLip = faceLandmarks[62];
                const cv::Point2f& lowerLip = faceLandmarks[66];
                const double mouthOpenRaw = cv::norm(upperLip - lowerLip);

                int trackId = -1;
                double mouthOpen = mouthOpenRaw;
                if (matchedFace != nullptr) {
                    const double h = matchedFace->bbox.height;
                    mouthOpen = h > 0.0 ? mouthOpenRaw / (h / static_cast<double>(defaultHeight_)) : mouthOpenRaw;
                    trackId = matchedFace->trackId;
                }

                if (trackId >= 0) {
                    auto& history = mouthMotionHistory_[trackId];
                    if (history.size() >= static_cast<std::size_t>(mouthBufferSize_)) {
                        history.pop_front();
                    }
                    history.push_back(mouthOpen);
                    lastSeenTrack_[trackId] = currentTime;
                    if (history.size() >= 3) {
                        isTalking = stddev(history) > talkingThreshold_ ? 1 : 0;
                    }
                }
            }

            double timeInView = 0.0;
            if (matchedFace != nullptr && matchedFace->trackId >= 0) {
                if (firstSeenTrack_.find(matchedFace->trackId) == firstSeenTrack_.end()) {
                    firstSeenTrack_[matchedFace->trackId] = currentTime;
                }
                lastSeenTrack_[matchedFace->trackId] = currentTime;
                timeInView = currentTime - firstSeenTrack_[matchedFace->trackId];
            }

            publishLandmarks(matchedFace, faceForward, pitch, yaw, roll, cosAngle, attention, isTalking, timeInView);
        }
    }

    for (auto& faceData : detectedFaces_) {
        if (faceData.trackId >= 0 && matchedTrackIds.find(faceData.trackId) != matchedTrackIds.end()) {
            continue;
        }
        if (faceData.trackId >= 0) {
            if (firstSeenTrack_.find(faceData.trackId) == firstSeenTrack_.end()) {
                firstSeenTrack_[faceData.trackId] = currentTime;
            }
            lastSeenTrack_[faceData.trackId] = currentTime;
        }
        const double timeInView = faceData.trackId >= 0 ? currentTime - firstSeenTrack_[faceData.trackId] : 0.0;
        publishLandmarks(&faceData, cv::Vec3d(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, "UNKNOWN", 0, timeInView);
    }

    std::vector<int> expired;
    for (const auto& entry : lastSeenTrack_) {
        if (currentTime - entry.second > 1.0) {
            expired.push_back(entry.first);
        }
    }
    for (int trackId : expired) {
        mouthMotionHistory_.erase(trackId);
        lastSeenTrack_.erase(trackId);
        firstSeenTrack_.erase(trackId);
    }
}

void VisionAnalyzer::detectLight() {
    if (image_.empty()) {
        return;
    }

    if (cv::mean(image_)[0] != 0.0 || cv::mean(image_)[1] != 0.0 || cv::mean(image_)[2] != 0.0) {
        cv::Mat gray;
        cv::cvtColor(image_, gray, cv::COLOR_BGR2GRAY);
        if (optFlowBuf_.size() >= 2) {
            optFlowBuf_.pop_front();
        }
        optFlowBuf_.push_back(gray);
    }

    cv::Mat hsv;
    cv::cvtColor(image_, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::Mat normalized;
    cv::normalize(channels[2], normalized, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    envState_.light = std::round(cv::mean(normalized)[0] * 100.0) / 100.0;
}

void VisionAnalyzer::detectMotion() {
    if (optFlowBuf_.size() != 2) {
        return;
    }

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(optFlowBuf_[0], optFlowBuf_[1], flow, 0.5, 3, 15, 3, 5, 0.2, 0);
    std::vector<cv::Mat> flowChannels(2);
    cv::split(flow, flowChannels);
    cv::Mat mag, ang;
    cv::cartToPolar(flowChannels[0], flowChannels[1], mag, ang);
    envState_.motion = std::round(cv::mean(mag)[0] * 100.0) / 100.0;
}

void VisionAnalyzer::drawAndPublishFacesView() {
    if (faceDetectionImgPort_.getOutputCount() == 0 || lastFrame_.empty()) {
        return;
    }

    cv::Mat frame = lastFrame_.clone();
    for (const auto& detection : currentDetections_) {
        const cv::Rect rect(cvRound(detection.box.x), cvRound(detection.box.y),
                            cvRound(detection.box.width), cvRound(detection.box.height));
        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);

        std::string name = detection.className;
        if (identifyFaces_) {
            name = detection.faceId == "recognizing" ? "recognizing..." : detection.faceId;
        }

        std::string attention;
        std::string distance;
        std::string zone;
        for (const auto& faceData : detectedFaces_) {
            if (faceData.trackId == detection.trackId) {
                attention = faceData.attention;
                distance = faceData.distance;
                zone = faceData.zone;
                break;
            }
        }

        std::string label = "id:" + std::to_string(detection.trackId) + " | " + name;
        if (!attention.empty() && attention != "UNKNOWN") {
            label += " | " + attention;
        }
        if (!distance.empty() && distance != "UNKNOWN") {
            label += " | " + distance;
        }
        if (!zone.empty() && zone != "UNKNOWN") {
            label += " | " + zone;
        }

        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const cv::Point origin(rect.x, std::max(0, rect.y - 6));
        cv::rectangle(frame,
                      cv::Rect(origin.x, std::max(0, origin.y - textSize.height - 4), textSize.width + 6, textSize.height + 6),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    writeAnnotatedImage(frame);
}

void VisionAnalyzer::handleTargetCommand() {
    while (auto* cmdBottle = targetCmdPort_.read(false)) {
        if (cmdBottle->size() >= 2) {
            currentTargetTrackId_ = cmdBottle->get(0).asInt32();
            currentTargetIps_ = cmdBottle->get(1).asFloat64();
        }
    }

    if (currentTargetTrackId_ < 0 || targetBoxPort_.getOutputCount() == 0) {
        return;
    }

    auto targetIt = std::find_if(detectedFaces_.begin(), detectedFaces_.end(), [&](const DetectedFaceData& face) {
        return face.trackId == currentTargetTrackId_;
    });
    if (targetIt == detectedFaces_.end()) {
        return;
    }

    const double xMin = targetIt->bbox.x;
    const double yMin = targetIt->bbox.y;
    const double xMax = targetIt->bbox.x + targetIt->bbox.width;
    const double yMax = targetIt->bbox.y + targetIt->bbox.height;

    auto& out = targetBoxPort_.prepare();
    out.clear();
    auto& obj = out.addList();
    auto& classBtl = obj.addList();
    classBtl.addString("class");
    classBtl.addString("face");
    auto& scoreBtl = obj.addList();
    scoreBtl.addString("score");
    scoreBtl.addFloat64(currentTargetIps_);
    auto& boxBtl = obj.addList();
    boxBtl.addString("box");
    auto& coords = boxBtl.addList();
    coords.addFloat64(xMin);
    coords.addFloat64(yMin);
    coords.addFloat64(xMax);
    coords.addFloat64(yMax);
    targetBoxPort_.write();
}

void VisionAnalyzer::fillBottle() {
    yarp::os::Bottle timeBottle;
    timeBottle.addString("Time");
    timeBottle.addFloat64(timestamp_);
    visionFeaturesBottle_.addList().copy(timeBottle);

    const std::vector<std::pair<std::string, double>> numericFeatures = {
        {"Faces", static_cast<double>(envState_.faces)},
        {"People", static_cast<double>(envState_.people)},
        {"Motion", envState_.motion},
        {"Light", envState_.light},
        {"MutualGaze", static_cast<double>(envState_.mutualGaze)}
    };

    for (const auto& feature : numericFeatures) {
        yarp::os::Bottle bottle;
        bottle.addString(feature.first);
        if (feature.first == "Motion" || feature.first == "Light") {
            bottle.addFloat64(feature.second);
        } else {
            bottle.addInt32(static_cast<int>(feature.second));
        }
        visionFeaturesBottle_.addList().copy(bottle);
    }
}

void VisionAnalyzer::loadKnownFaces(const std::string& facesPath) {
    knownFaces_.clear();
    std::error_code ec;
    if (!fs::exists(facesPath, ec)) {
        std::cerr << "[WARNING] Faces folder does not exist: " << facesPath << '\n';
        return;
    }

    for (const auto& entry : fs::directory_iterator(facesPath)) {
        if (!entry.is_regular_file() || !isImageFile(entry.path())) {
            continue;
        }

        const std::string personName = entry.path().stem().string();
        if (knownFaces_.find(personName) != knownFaces_.end()) {
            std::cerr << "[WARNING] Duplicate face identity basename '" << personName << "' in " << facesPath << '\n';
            continue;
        }

        cv::Mat image = cv::imread(entry.path().string());
        cv::Mat embedding;
        if (computeEmbeddingFromImage(image, embedding)) {
            knownFaces_[personName].push_back(embedding);
        }
    }
}

bool VisionAnalyzer::handleFaceNaming(const yarp::os::Bottle& command, yarp::os::Bottle& reply) {
    if (command.size() != 4 || command.get(0).asString() != "name" || command.get(2).asString() != "id") {
        reply.addString("nack");
        reply.addString("Usage: name <person_name> id <track_id>");
        return true;
    }

    if (!track_) {
        reply.addString("nack");
        reply.addString("Face naming requires --track true");
        return true;
    }

    if (!identifyFaces_) {
        reply.addString("nack");
        reply.addString("Face naming requires --identify_faces true");
        return true;
    }

    if (lastFrame_.empty()) {
        reply.addString("nack");
        reply.addString("No frame available yet");
        return true;
    }

    const std::string personName = command.get(1).asString();
    const int trackId = command.get(3).asInt32();

    auto it = std::find_if(currentDetections_.begin(), currentDetections_.end(), [&](const FaceDetection& detection) {
        return detection.trackId == trackId;
    });
    if (it == currentDetections_.end()) {
        reply.addString("nack");
        reply.addString("Track ID not found in current detections");
        return true;
    }

    const cv::Rect cropRect(cvRound(it->box.x), cvRound(it->box.y), cvRound(it->box.width), cvRound(it->box.height));
    const cv::Rect bounded = cropRect & cv::Rect(0, 0, lastFrame_.cols, lastFrame_.rows);
    if (bounded.width <= 0 || bounded.height <= 0) {
        reply.addString("nack");
        reply.addString("Invalid bounding box");
        return true;
    }

    cv::Mat faceCrop = lastFrame_(bounded).clone();
    cv::Mat embedding;
    if (!computeEmbeddingForDetection(lastFrame_, *it, embedding)) {
        reply.addString("nack");
        reply.addString("Could not extract face encoding from crop");
        return true;
    }

    fs::create_directories(facesPath_);
    const fs::path outputPath = fs::path(facesPath_) / (personName + ".jpg");
    cv::imwrite(outputPath.string(), faceCrop);

    {
        std::lock_guard<std::mutex> lock(faceIdentityMutex_);
        auto& samples = knownFaces_[personName];
        samples.push_back(embedding);
        if (static_cast<int>(samples.size()) > maxEnrollSamples_) {
            samples.erase(samples.begin(), samples.end() - maxEnrollSamples_);
        }
        trackedFaces_[trackId] = {personName, 1.0};
        lastKnownIdentity_[trackId] = StickyIdentity{personName, 1.0, nowSeconds()};
    }

    reply.addString("ok");
    reply.addString(outputPath.string());
    return true;
}

bool VisionAnalyzer::respond(const yarp::os::Bottle& command, yarp::os::Bottle& reply) {
    reply.clear();
    if (command.size() == 0) {
        reply.addString("nack");
        return true;
    }

    const std::string verb = command.get(0).asString();
    if (verb == "quit") {
        reply.addString("quitting");
        return false;
    }
    if (verb == "process") {
        if (command.size() < 2) {
            reply.addString("nack");
            reply.addString("Usage: process on/off");
            return true;
        }
        reply.addString("ok");
        return true;
    }
    if (verb == "name") {
        return handleFaceNaming(command, reply);
    }
    if (verb == "help") {
        reply.addString("Commands: quit | process on/off | name <person_name> id <track_id>");
        return true;
    }

    reply.addString("nack");
    return true;
}

bool VisionAnalyzer::interruptModule() {
    std::cout << "stopping the module\n";
    imgInPort_.interrupt();
    visionFeaturesPort_.interrupt();
    landmarksPort_.interrupt();
    targetCmdPort_.interrupt();
    targetBoxPort_.interrupt();
    qrPort_.interrupt();
    handlePort_.interrupt();
    faceDetectionImgPort_.interrupt();
    return true;
}

bool VisionAnalyzer::close() {
    std::cout << "closing the module\n";
    imgInPort_.close();
    visionFeaturesPort_.close();
    landmarksPort_.close();
    targetCmdPort_.close();
    targetBoxPort_.close();
    qrPort_.close();
    handlePort_.close();
    faceDetectionImgPort_.close();
    return true;
}

int main(int argc, char* argv[]) {
    if (!yarp::os::Network::checkNetwork()) {
        std::cerr << "Unable to find a yarp server, exiting.\n";
        return 1;
    }

    yarp::os::Network yarp;
    VisionAnalyzer analyzer;
    yarp::os::ResourceFinder rf;
    rf.setVerbose(true);
    rf.setDefaultContext("alwaysOn");
    rf.configure(argc, argv);

    return analyzer.runModule(rf);
}
