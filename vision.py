import sys
import cv2
import numpy as np
import yarp
import time
from datetime import datetime
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

        self.logger = get_colored_logger("Video Features")
        self.rate = 0.05

        self.img_in_port = yarp.BufferedPortImageRgb()              # Raw images
        self.face_detection_port = yarp.BufferedPortBottle()        # FaceDetection detections
        self.vision_features_port = yarp.Port()                     # Output Features
        self.landmarks_port = yarp.Port()                           # Per-face detailed information

        self.img_in_btl = yarp.ImageRgb()
        self.face_detection_btl = yarp.Bottle()
        self.vision_features_btl = yarp.Bottle()
        self.landmarks_btl = yarp.Bottle()

        self.name = "alwayson/vision"
        self.img_width = 640
        self.img_height = 480
        self.default_width = 640
        self.default_height = 480
        self.input_img_array = None
        self.image = None
        self.opt_flow_buf = deque()
        self.timestamp = None

        self.env_dict = {
            "Faces": 0,
            "People": 0,
            "Motion": 0.0,
            "Light": 0.0,
            "MutualGaze": 0,
        }

        # To make information stable and not instantaneously change
        self.faces_sync_info = 0
        self.exec_time = 0.15  # Execution mean time for the module

        self.mutual_gaze_threshold = 10
        self.max_face_match_distance = 100.0  # Max distance (pixels) for matching MediaPipe to bbox

        self.face_mesh = None
        self.detected_faces = []  # Store face data from faceDetection (bbox, face_id)
        
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

        print(f"IMAGE W: {self.img_width}")
        print(f"IMAGE H: {self.img_height}")
        print(f"RATE: {self.rate}")
        print(f"MODEL PATH: {self.landmark_model_path}")
        
        # Use ResourceFinder to locate the model file in the context directory
        model_full_path = rf.findFileByName(self.landmark_model_path)
        if not model_full_path:
            self.logger.error(f"Could not find model file: {self.landmark_model_path}")
            return False
        print(f"MODEL FULL PATH: {model_full_path}")

        self.img_in_port.open(f'/{self.name}/img:i')
        self.face_detection_port.open(f'/{self.name}/recognition:i')
        self.vision_features_port.open(f'/{self.name}/features:o')
        self.landmarks_port.open(f'/{self.name}/landmarks:o')

        self.input_img_array = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        base_options = python.BaseOptions(
            model_asset_path=model_full_path
        )


        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,

        )

        self.face_mesh = vision.FaceLandmarker.create_from_options(options)

        self.logger.info("Start processing video")
        return True

    def getPeriod(self):
        return self.rate

    def updateModule(self):
        self.vision_features_btl.clear()
        self.landmarks_btl.clear()
        has_features_subscriber = self.vision_features_port.getOutputCount() > 0
        has_landmarks_subscriber = self.landmarks_port.getOutputCount() > 0

        if has_features_subscriber or has_landmarks_subscriber:
            self.img_in_btl = self.img_in_port.read(shouldWait=True)
            if self.img_in_btl:
                self.image = self.__img_yarp_to_cv(self.img_in_btl)
                self.detect_people_obj()        # Count how many people and objects in the scene
                self.detect_mutual_gaze()       # Count # people looking at the camera and publish per-face details
                self.detect_light()             # Extract from a HSV space, the V component of the image
                self.detect_motion()            # Only presence (no magnitude or orientation)
            self.timestamp = datetime.now().timestamp()

            if has_features_subscriber:
                self.fill_bottle()
                self.vision_features_port.write(self.vision_features_btl)

            if has_landmarks_subscriber:
                self.landmarks_port.write(self.landmarks_btl)

        return True

    def detect_people_obj(self):
        # Input from faceDetection: nested bottle structure
        # Outer bottle contains N lists, each list is one detection:
        # ((class face) (score ...) (track_id 1) (box (... ... ... ...)) (face_id nima) ...)
        # ((class face) ...)
        init = time.time()
        # Always read the latest data from faceDetection
        self.face_detection_btl = self.face_detection_port.read(shouldWait=False)
        if self.face_detection_btl:
            # Clear and rebuild with fresh data
            self.detected_faces = []
            num_faces = 0
            
            # Iterate through outer bottle (each element is a detection)
            for i in range(self.face_detection_btl.size()):
                det = self.face_detection_btl.get(i).asList()
                if not det:
                    continue
                
                # Initialize extraction variables
                class_name = ""
                track_id = -1
                face_id = "unknown"
                detection_score = 0.0
                id_confidence = 0.0
                x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
                
                # Iterate through fields in this detection
                # Each field is itself a list like (class face), (track_id 1), (box x1 y1 x2 y2), etc.
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
                        # box format: (box (x1 y1 x2 y2))
                        box_list = field.get(1).asList()
                        if box_list and box_list.size() >= 4:
                            x1 = box_list.get(0).asFloat64()
                            y1 = box_list.get(1).asFloat64()
                            x2 = box_list.get(2).asFloat64()
                            y2 = box_list.get(3).asFloat64()
                
                # Only process face detections with sufficient detection confidence
                if class_name == "face" and detection_score > 0.5:
                    num_faces += 1
                    
                    # Convert to (x, y, w, h) format
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Store face info for matching with MediaPipe
                    self.detected_faces.append({
                        'face_id': face_id,
                        'track_id': track_id,
                        'bbox': (x, y, w, h),
                        'detection_score': detection_score,
                        'id_confidence': id_confidence
                    })
            
            # Update face count immediately when we receive valid data
            self.env_dict["Faces"] = num_faces
            self.faces_sync_info = time.time()
        else:
            # sync with faceDetection; if no info comes for >0.5s, set faces=0
            if (time.time() - self.faces_sync_info) > 0.5:
                if self.env_dict["Faces"] != 0:
                    self.env_dict["Faces"] = 0
                    self.detected_faces = []

    def detect_light(self):
        if self.image.mean() != 0.0:
            self.opt_flow_add_img(self.image)
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(image)
        cv2.normalize(v, v, 0, 1.0, cv2.NORM_MINMAX)
        self.env_dict["Light"] = round(v.mean(), 2)


    def detect_mutual_gaze(self):

        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        results = self.face_mesh.detect(mp_image)
        img_h, img_w, img_c = self.image.shape

        self.env_dict['MutualGaze'] = 0
        
        # Track which bboxes have been matched (one-to-one matching)
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

                # Calculate face center from landmarks for matching with bboxes
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
                pitch = angles[0]  # up/down
                yaw = angles[1]    # left/right
                roll = angles[2]   # tilt

                # Face forward vector in camera coordinates
                face_forward = rmat @ np.array([0, 0, -1])

                # Camera looks along +Z
                camera_forward = np.array([0, 0, 1])

                cos_angle = np.dot(face_forward, camera_forward)

                # Determine attention state based on cos_angle
                if cos_angle > 0.95:
                    attention = "MUTUAL_GAZE"
                    self.env_dict["MutualGaze"] += 1
                elif cos_angle > 0.7:
                    attention = "NEAR_GAZE"
                else:
                    attention = "AWAY"

                # Match MediaPipe detection with faceDetection face (one-to-one)
                matched_face = self._match_face_to_bbox(face_center_x, face_center_y, matched_track_ids)
                
                # Mark this face as matched to prevent duplicate assignments
                if matched_face:
                    matched_track_ids.add(matched_face['track_id'])
                
                # Compute talking detection based on lip motion
                # Landmarks 13 (upper inner lip) and 14 (lower inner lip)
                is_talking = 0
                if len(face_landmarks) > 14:
                    upper_lip = face_landmarks[13]
                    lower_lip = face_landmarks[14]
                    
                    # Compute mouth opening in normalized coordinates
                    mouth_open_raw = np.hypot(upper_lip.x - lower_lip.x, upper_lip.y - lower_lip.y)
                    
                    # Normalize by face bbox height if available (scale-invariant)
                    if matched_face:
                        x, y, w, h = matched_face['bbox']
                        # Normalize: mouth_open as fraction of face height
                        mouth_open = mouth_open_raw / (h / self.default_height) if h > 0 else mouth_open_raw
                        track_id = matched_face['track_id']
                    else:
                        # If unmatched, use raw normalized coords (less reliable)
                        mouth_open = mouth_open_raw
                        track_id = -1  # Unmatched
                    
                    # Update motion history for this track_id
                    if track_id != -1:
                        if track_id not in self.mouth_motion_history:
                            self.mouth_motion_history[track_id] = deque(maxlen=self.mouth_buffer_size)
                        
                        self.mouth_motion_history[track_id].append(mouth_open)
                        self.last_seen_track[track_id] = current_time
                        
                        # Compute motion as std of mouth opening history
                        if len(self.mouth_motion_history[track_id]) >= 3:
                            mouth_motion = np.std(self.mouth_motion_history[track_id])
                            # Threshold: tune this for sensitivity (higher = less sensitive)
                            # 0.015 works well for normalized values; adjust if needed
                            is_talking = 1 if mouth_motion > self.talking_threshold else 0
                
                # Compute time_in_view
                if matched_face:
                    track_id = matched_face['track_id']
                    if track_id not in self.first_seen_track:
                        self.first_seen_track[track_id] = current_time
                    time_in_view = current_time - self.first_seen_track[track_id]
                else:
                    time_in_view = 0.0
                
                # Publish per-face landmarks data
                self._publish_landmarks(matched_face, face_forward, pitch, yaw, roll, cos_angle, attention, is_talking, time_in_view)
        
        # Publish data for faces detected by faceDetection but not matched by MediaPipe
        # (e.g., faces too small for landmark detection)
        for face_data in self.detected_faces:
            if face_data['track_id'] not in matched_track_ids:
                # Compute time_in_view for unmatched face
                track_id = face_data['track_id']
                if track_id not in self.first_seen_track:
                    self.first_seen_track[track_id] = current_time
                self.last_seen_track[track_id] = current_time
                time_in_view = current_time - self.first_seen_track[track_id]
                
                # Publish with default/unknown gaze values since no landmarks available
                self._publish_landmarks(
                    face_data=face_data,
                    gaze_direction=np.array([0.0, 0.0, 0.0]),
                    pitch=0.0,
                    yaw=0.0,
                    roll=0.0,
                    cos_angle=0.0,
                    attention="UNKNOWN",
                    is_talking=0,
                    time_in_view=time_in_view
                )

        # Cleanup: remove histories for tracks not seen in over 1 second
        tracks_to_remove = [tid for tid, last_time in self.last_seen_track.items() 
                           if current_time - last_time > 1.0]
        for tid in tracks_to_remove:
            if tid in self.mouth_motion_history:
                del self.mouth_motion_history[tid]
            if tid in self.last_seen_track:
                del self.last_seen_track[tid]
            if tid in self.first_seen_track:
                del self.first_seen_track[tid]

    def _match_face_to_bbox(self, face_x, face_y, matched_track_ids):
        """Match MediaPipe face detection to object recognition face using distance-based scoring.
        
        Implements one-to-one matching: each bbox can only be assigned to one MediaPipe face.
        Computes distance from face center to each bbox center, selects the closest match
        within MAX_FACE_MATCH_DISTANCE threshold. Ties are broken by preferring larger bbox area.
        
        Args:
            face_x: X coordinate of MediaPipe face center
            face_y: Y coordinate of MediaPipe face center
            matched_track_ids: Set of track_ids that have already been matched
            
        Returns:
            Best matching face_data dict or None if no valid match found
        """
        if not self.detected_faces:
            return None
        
        best_match = None
        best_distance = float('inf')
        
        for face_data in self.detected_faces:
            # Skip faces that have already been matched (one-to-one constraint)
            if face_data['track_id'] in matched_track_ids:
                continue
            
            x, y, w, h = face_data['bbox']
            
            # Compute bbox center
            bbox_center_x = x + w / 2.0
            bbox_center_y = y + h / 2.0
            
            # Compute Euclidean distance from face center to bbox center
            distance = np.hypot(face_x - bbox_center_x, face_y - bbox_center_y)
            
            # Update best match if this is closer
            if distance < best_distance:
                best_distance = distance
                best_match = face_data
            elif distance == best_distance and best_match is not None:
                # Tie-breaker: prefer larger bbox area for stability
                current_area = w * h
                best_area = best_match['bbox'][2] * best_match['bbox'][3]
                if current_area > best_area:
                    best_match = face_data
        
        # Apply gating threshold to reject poor matches
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
            
            # Add spatial information
            # Compute bbox center
            cx = x + w / 2.0
            cy = y + h / 2.0
            
            # Normalize center using default resolution
            cx_n = cx / self.default_width
            cy_n = cy / self.default_height
            
            # Clamp to [0, 1]
            cx_n = max(0.0, min(1.0, cx_n))
            cy_n = max(0.0, min(1.0, cy_n))
            
            # Determine horizontal zone
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
            
            # Add zone
            face_btl.addString("zone")
            face_btl.addString(zone)
            
            # Determine distance based on face bbox height
            # Normalized height relative to default image height
            h_norm = h / self.default_height
            
            if h_norm > 0.4:
                distance = "SO_CLOSE"
            elif h_norm > 0.2:
                distance = "CLOSE"
            elif h_norm > 0.1:
                distance = "FAR"
            else:
                distance = "VERY_FAR"
            
            # Add distance
            face_btl.addString("distance")
            face_btl.addString(distance)
        else:
            # Unmatched face - use default/null values for all fields
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
            
            # Add default zone
            face_btl.addString("zone")
            face_btl.addString("UNKNOWN")
            
            # Add default distance
            face_btl.addString("distance")
            face_btl.addString("UNKNOWN")
        
        # Add gaze direction vector
        gaze_btl = face_btl.addList()
        gaze_btl.addString("gaze_direction")
        gaze_btl.addFloat64(float(gaze_direction[0]))
        gaze_btl.addFloat64(float(gaze_direction[1]))
        gaze_btl.addFloat64(float(gaze_direction[2]))
        
        # Add head orientation angles (degrees)
        face_btl.addString("pitch")
        face_btl.addFloat64(float(pitch))
        face_btl.addString("yaw")
        face_btl.addFloat64(float(yaw))
        face_btl.addString("roll")
        face_btl.addFloat64(float(roll))
        
        # Add cosine angle and attention state
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

        if image.width() != self.img_width or image.height() != self.img_height:
            self.logger.warning("imput image has different size from default 640x480, fallback to automatic size detection")
            self.img_width = image.width()
            self.img_height = image.height()
            self.input_img_array = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            print(f"New image size: W: {self.img_width}, H: {self.img_height}")


        image.setExternal(self.input_img_array.data, self.img_width, self.img_height)
        img = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
            (self.img_height, self.img_width, 3)).copy()

        if self.img_width != self.default_width or self.img_height != self.default_height:
            img = cv2.resize(img, (self.default_width, self.default_height))

        return img

    def opt_flow_add_img(self, frame):
        if len(self.opt_flow_buf) < 2:
            # Add new frame to the right part of the queue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.opt_flow_buf.append(frame)
        else:
            # Remove oldest element at the beginning of the queue
            self.opt_flow_buf.popleft()

    def detect_motion(self):
        if len(self.opt_flow_buf) == 2:
            # Dense optical flow estimate
            flow = cv2.calcOpticalFlowFarneback(self.opt_flow_buf[0], self.opt_flow_buf[1],
                                                flow=None, pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=.2,
                                                flags=0)
            # Compute magnite and angle of 2D vector
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # Dim (480, 640)
            # Cast into float64 since cartToPolar is float32 to avoid conflict with types when publishing with Yarp
            self.env_dict["Motion"] = round(mag.mean(), 2).astype(np.float64)

    def fill_bottle(self):
        btl_time = yarp.Bottle()
        btl_time.addString("Time")
        btl_time.addFloat64(self.timestamp)
        self.vision_features_btl.addList().read(btl_time)
        for key, value in self.env_dict.items():
            bottle = yarp.Bottle()
            bottle.addString(key)
            if isinstance(value, int):
                bottle.addInt16(value)
            elif isinstance(value, float):
                bottle.addFloat64(value)
            elif isinstance(value, str):
                bottle.addString(value)
            elif isinstance(value, list) or isinstance(value, np.ndarray):
                # if it is a list, add each element to the bottle
                bottle_list = yarp.Bottle()
                for element in value:
                    if isinstance(element, int):
                        bottle_list.addInt16(element)
                    elif isinstance(element, float):
                        bottle_list.addFloat32(element)
                    elif isinstance(element, str):
                        bottle_list.addString(element)
                self.vision_features_btl.addList().read(bottle_list)
            self.vision_features_btl.addList().read(bottle)

    def interruptModule(self):
        print("stopping the module \n")
        self.img_in_port.interrupt()
        self.face_detection_port.interrupt()
        self.vision_features_port.interrupt()
        self.landmarks_port.interrupt()
        return True

    def close(self):
        print("closing the module \n")
        self.img_in_port.close()
        self.face_detection_port.close()
        self.vision_features_port.close()
        self.landmarks_port.close()
        return True


if __name__ == '__main__':

    logger = get_colored_logger("Video Features")
    # Initialise YARP
    if not yarp.Network.checkNetwork():
        logger.warning("Unable to find a yarp server exiting ...")
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
