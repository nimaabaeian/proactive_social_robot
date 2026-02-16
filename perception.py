import sys
import cv2
import numpy as np
import yarp
import time
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
import colorlog


def get_colored_logger(name):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(name)s] - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("alwaysOn.log")])

    logger = logging.getLogger(name)
    colored_handler = colorlog.StreamHandler()
    colored_handler.setFormatter(colorlog.ColoredFormatter(
        '%(asctime)s %(log_color)s[%(name)s][%(levelname)s] %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'purple',
        },

    ))
    logger.addHandler(colored_handler)
    return logger

LANDMARK_IDS = [
    1,    # Nose tip
    199,  # Chin
    33,   # Left eye left corner
    263,  # Right eye right corner
    61,   # Left mouth corner
    291   # Right mouth corner
]

FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),   # Chin
    (-43.3, 32.7, -26.0),  # Left eye left corner
    (43.3, 32.7, -26.0),   # Right eye right corner
    (-28.9, -28.9, -24.1), # Left mouth corner
    (28.9, -28.9, -24.1)   # Right mouth corner
], dtype=np.float64)


class VisionAnalyzer(yarp.RFModule):
    def __init__(self):
        yarp.RFModule.__init__(self)

        self.logger = get_colored_logger("Vision Analyzer")
        self.rate = 0.05

        self.img_in_port = yarp.BufferedPortImageRgb()              # Raw images
        self.object_recognition_port = yarp.BufferedPortBottle()    # Object Recognition detection
        self.landmarks_port = yarp.Port()                           # Per-face detailed information

        self.img_in_btl = yarp.ImageRgb()
        self.object_recognition_btl = yarp.Bottle()
        self.landmarks_btl = yarp.Bottle()

        self.name = "alwayson/vision"
        self.img_width = 640
        self.img_height = 480
        self.default_width = 640
        self.default_height = 480
        self.input_img_array = None
        self.image = None
        self.faces_sync_info = 0
        self.max_face_match_distance = 100.0  # Max distance (pixels) for matching MediaPipe to bbox

        self.face_mesh = None
        self.detected_faces = []  # Store face data from object recognition (bbox, face_id)
        
        # Talking detection based on lip motion
        self.mouth_motion_history = {}  # dict[track_id] -> deque of normalized mouth_open values
        self.mouth_buffer_size = 10  # Number of frames to track for motion detection
        self.talking_threshold = 0.012  # Std threshold for mouth motion (tunable: increase to reduce false positives)
        self.last_seen_track = {}  # dict[track_id] -> timestamp for cleanup
        self.first_seen_track = {}  # dict[track_id] -> timestamp when first seen

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        self.name = rf.check("name", yarp.Value(self.name)).asString()
        self.rate = rf.check("rate", yarp.Value("0.05")).asFloat64()
        self.img_width = rf.check("width", yarp.Value(640)).asInt64()
        self.img_height = rf.check("height", yarp.Value(480)).asInt64()
        self.landmark_model_path = rf.check("model", yarp.Value("face_landmarker.task")).asString()

        self.logger.info(f"Configuration: {self.img_width}x{self.img_height} @ {self.rate}Hz")
        self.logger.info(f"Model path: {self.landmark_model_path}")
        
        model_full_path = rf.findFileByName(self.landmark_model_path)
        if not model_full_path:
            self.logger.error(f"Could not find model file: {self.landmark_model_path}")
            return False
        self.logger.info(f"Model loaded from: {model_full_path}")

        self.img_in_port.open(f'/{self.name}/img:i')
        self.object_recognition_port.open(f'/{self.name}/recognition:i')
        self.landmarks_port.open(f'/{self.name}/landmarks:o')

        self.input_img_array = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        base_options = python.BaseOptions(model_asset_path=model_full_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.face_mesh = vision.FaceLandmarker.create_from_options(options)

        self.logger.info("Start processing video")
        return True

    def getPeriod(self):
        return self.rate

    def updateModule(self):
        self.landmarks_btl.clear()
        if self.landmarks_port.getOutputCount() > 0:
            self.img_in_btl = self.img_in_port.read(shouldWait=True)
            if self.img_in_btl:
                self.image = self.__img_yarp_to_cv(self.img_in_btl)
                self.detect_people_obj()
                self.detect_mutual_gaze()
            self.landmarks_port.write(self.landmarks_btl)
        return True

    def detect_people_obj(self):
        self.object_recognition_btl = self.object_recognition_port.read(shouldWait=False)
        if self.object_recognition_btl:
            self.detected_faces = []
            
            # Iterate through outer bottle (each element is a detection)
            for i in range(self.object_recognition_btl.size()):
                det = self.object_recognition_btl.get(i).asList()
                if not det:
                    continue
                
                # Initialize extraction variables
                class_name = ""
                track_id = -1
                face_id = "unknown"
                detection_score = 0.0
                id_confidence = 0.0
                x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
                
                for j in range(det.size()):
                    field = det.get(j).asList()
                    if not field or field.size() < 2:
                        continue
                    
                    key = field.get(0).asString()
                    
                    if key == "class":
                        class_name = field.get(1).asString()
                    elif key == "track_id":
                        track_id = field.get(1).asInt32()
                    elif key == "face_id":
                        face_id = field.get(1).asString()
                    elif key == "score":
                        detection_score = field.get(1).asFloat64()
                    elif key == "id_confidence":
                        id_confidence = field.get(1).asFloat64()
                    elif key == "box":
                        box_list = field.get(1).asList()
                        if box_list and box_list.size() >= 4:
                            x1 = box_list.get(0).asFloat64()
                            y1 = box_list.get(1).asFloat64()
                            x2 = box_list.get(2).asFloat64()
                            y2 = box_list.get(3).asFloat64()
                
                if class_name == "face" and detection_score > 0.5:
                    self.detected_faces.append({
                        'face_id': face_id,
                        'track_id': track_id,
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'detection_score': detection_score,
                        'id_confidence': id_confidence
                    })
            
            self.faces_sync_info = time.time()
        elif (time.time() - self.faces_sync_info) > 0.5:
            self.detected_faces = []

    def detect_mutual_gaze(self):
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        results = self.face_mesh.detect(mp_image)
        img_h, img_w, img_c = self.image.shape

        matched_track_ids = set()
        current_time = time.time()

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:

                face_2d = []

                for idx in LANDMARK_IDS:
                    lm = face_landmarks[idx]
                    x = lm.x * img_w
                    y = lm.y * img_h
                    face_2d.append([x, y])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_center_x = np.mean(face_2d[:, 0])
                face_center_y = np.mean(face_2d[:, 1])

                focal_length = img_w
                cam_matrix = np.array([
                    [focal_length, 0, img_w / 2],
                    [0, focal_length, img_h / 2],
                    [0, 0, 1]
                ], dtype=np.float64)

                dist_coeffs = np.zeros((4, 1))

                success, rvec, tvec = cv2.solvePnP(
                    FACE_3D_MODEL,
                    face_2d,
                    cam_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if not success:
                    continue

                rmat, _ = cv2.Rodrigues(rvec)
                angles, *_ = cv2.RQDecomp3x3(rmat)
                pitch, yaw, roll = angles[0], angles[1], angles[2]

                face_forward = rmat @ np.array([0, 0, -1])
                cos_angle = np.dot(face_forward, np.array([0, 0, 1]))

                if cos_angle > 0.95:
                    attention = "MUTUAL_GAZE"
                elif cos_angle > 0.7:
                    attention = "NEAR_GAZE"
                else:
                    attention = "AWAY"

                matched_face = self._match_face_to_bbox(face_center_x, face_center_y, matched_track_ids)
                if matched_face:
                    matched_track_ids.add(matched_face['track_id'])
                is_talking = 0
                if len(face_landmarks) > 14:
                    upper_lip = face_landmarks[13]
                    lower_lip = face_landmarks[14]
                    mouth_open_raw = np.hypot(upper_lip.x - lower_lip.x, upper_lip.y - lower_lip.y)
                    
                    if matched_face:
                        x, y, w, h = matched_face['bbox']
                        mouth_open = mouth_open_raw / (h / self.default_height) if h > 0 else mouth_open_raw
                        track_id = matched_face['track_id']
                    else:
                        mouth_open = mouth_open_raw
                        track_id = -1
                    
                    if track_id != -1:
                        if track_id not in self.mouth_motion_history:
                            self.mouth_motion_history[track_id] = deque(maxlen=self.mouth_buffer_size)
                        
                        self.mouth_motion_history[track_id].append(mouth_open)
                        self.last_seen_track[track_id] = current_time
                        
                        if len(self.mouth_motion_history[track_id]) >= 3:
                            mouth_motion = np.std(self.mouth_motion_history[track_id])
                            is_talking = 1 if mouth_motion > self.talking_threshold else 0

                if matched_face:
                    track_id = matched_face['track_id']
                    if track_id not in self.first_seen_track:
                        self.first_seen_track[track_id] = current_time
                    time_in_view = current_time - self.first_seen_track[track_id]
                else:
                    time_in_view = 0.0
                
                self._publish_landmarks(matched_face, face_forward, pitch, yaw, roll, cos_angle, attention, is_talking, time_in_view)
        
        tracks_to_remove = [tid for tid, last_time in self.last_seen_track.items() if current_time - last_time > 1.0]
        for tid in tracks_to_remove:
            if tid in self.mouth_motion_history:
                del self.mouth_motion_history[tid]
            if tid in self.last_seen_track:
                del self.last_seen_track[tid]
            if tid in self.first_seen_track:
                del self.first_seen_track[tid]

    def _match_face_to_bbox(self, face_x, face_y, matched_track_ids):
        if not self.detected_faces:
            return None
        
        best_match = None
        best_distance = float('inf')
        
        for face_data in self.detected_faces:
            # Skip faces that have already been matched (one-to-one constraint)
            if face_data['track_id'] in matched_track_ids:
                continue
            
            x, y, w, h = face_data['bbox']
            bbox_center_x = x + w / 2.0
            bbox_center_y = y + h / 2.0
            distance = np.hypot(face_x - bbox_center_x, face_y - bbox_center_y)
            
            if distance < best_distance:
                best_distance = distance
                best_match = face_data
            elif distance == best_distance and best_match is not None:
                current_area = w * h
                best_area = best_match['bbox'][2] * best_match['bbox'][3]
                if current_area > best_area:
                    best_match = face_data
        
        if best_distance > self.max_face_match_distance:
            return None
        
        return best_match

    def _publish_landmarks(self, face_data, gaze_direction, pitch, yaw, roll, cos_angle, attention, is_talking, time_in_view):
        """Publish detailed landmarks information for a single face."""
        face_btl = yarp.Bottle()
        
        # Add face identification
        if face_data:
            face_btl.addString("face_id")
            face_btl.addString(face_data['face_id'])
            face_btl.addString("track_id")
            face_btl.addInt32(face_data['track_id'])
            
            # Add bounding box
            bbox_btl = face_btl.addList()
            bbox_btl.addString("bbox")
            x, y, w, h = face_data['bbox']
            bbox_btl.addFloat64(x)
            bbox_btl.addFloat64(y)
            bbox_btl.addFloat64(w)
            bbox_btl.addFloat64(h)
            
            cx = x + w / 2.0
            cy = y + h / 2.0
            cx_n = cx / self.default_width
            cy_n = cy / self.default_height
            cx_n = max(0.0, min(1.0, cx_n))
            cy_n = max(0.0, min(1.0, cy_n))
            
            if cx_n < 0.2:
                zone = "FAR_LEFT"
            elif cx_n < 0.4:
                zone = "LEFT"
            elif cx_n < 0.6:
                zone = "CENTER"
            elif cx_n < 0.8:
                zone = "RIGHT"
            else:
                zone = "FAR_RIGHT"
            
            face_btl.addString("zone")
            face_btl.addString(zone)
            
            h_norm = h / self.default_height
            
            if h_norm > 0.5:
                distance = "SO_CLOSE"
            elif h_norm > 0.3:
                distance = "CLOSE"
            elif h_norm > 0.15:
                distance = "FAR"
            else:
                distance = "VERY_FAR"
            
            face_btl.addString("distance")
            face_btl.addString(distance)
        else:
            face_btl.addString("face_id")
            face_btl.addString("unmatched")
            face_btl.addString("track_id")
            face_btl.addInt32(-1)
            
            # Add empty bounding box
            bbox_btl = face_btl.addList()
            bbox_btl.addString("bbox")
            bbox_btl.addFloat64(0.0)
            bbox_btl.addFloat64(0.0)
            bbox_btl.addFloat64(0.0)
            bbox_btl.addFloat64(0.0)
            
            face_btl.addString("zone")
            face_btl.addString("UNKNOWN")
            
            face_btl.addString("distance")
            face_btl.addString("UNKNOWN")
        
        gaze_btl = face_btl.addList()
        gaze_btl.addString("gaze_direction")
        gaze_btl.addFloat64(float(gaze_direction[0]))
        gaze_btl.addFloat64(float(gaze_direction[1]))
        gaze_btl.addFloat64(float(gaze_direction[2]))
        
        face_btl.addString("pitch")
        face_btl.addFloat64(float(pitch))
        face_btl.addString("yaw")
        face_btl.addFloat64(float(yaw))
        face_btl.addString("roll")
        face_btl.addFloat64(float(roll))
        
        face_btl.addString("cos_angle")
        face_btl.addFloat64(float(cos_angle))
        face_btl.addString("attention")
        face_btl.addString(attention)
        face_btl.addString("is_talking")
        face_btl.addInt32(is_talking)
        face_btl.addString("time_in_view")
        face_btl.addFloat64(float(time_in_view))
        
        self.landmarks_btl.addList().read(face_btl)

    def __img_yarp_to_cv(self, image):
        """Convert YARP image to OpenCV format and resize to default dimensions."""
        if image.width() != self.img_width or image.height() != self.img_height:
            self.img_width = image.width()
            self.img_height = image.height()
            self.input_img_array = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            self.logger.warning(f"Input image size changed to {self.img_width}x{self.img_height}")

        image.setExternal(self.input_img_array.data, self.img_width, self.img_height)
        img = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
            (self.img_height, self.img_width, 3)).copy()

        if self.img_width != self.default_width or self.img_height != self.default_height:
            img = cv2.resize(img, (self.default_width, self.default_height))

        return img

    def interruptModule(self):
        self.logger.info("Stopping module")
        self.img_in_port.interrupt()
        self.object_recognition_port.interrupt()
        self.landmarks_port.interrupt()
        return True

    def close(self):
        self.logger.info("Closing module")
        self.img_in_port.close()
        self.object_recognition_port.close()
        self.landmarks_port.close()
        return True


if __name__ == '__main__':
    logger = get_colored_logger("Vision Analyzer")
    
    if not yarp.Network.checkNetwork():
        logger.error("Unable to find YARP server")
        sys.exit(1)

    yarp.Network.init()
    vision_analyzer = VisionAnalyzer()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('alwaysOn')

    if rf.configure(sys.argv):
        vision_analyzer.runModule(rf)

    yarp.Network.fini()
    sys.exit(0)
