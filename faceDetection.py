import os
import sys
import subprocess
import cv2
import yarp
import numpy as np
import supervision as sv
from ultralytics import YOLO


class FaceDetection(yarp.RFModule):
    def __init__(self):
        super().__init__()

        self.default_face_model = (
            "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt"
        )

        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        self.input_port = yarp.BufferedPortImageRgb()
        self.output_objects_port = yarp.Port()
        self.output_img_port = yarp.BufferedPortImageRgb()

        self.input_img_array = None
        self.width_img = 640
        self.height_img = 480
        self.display_buf_image = yarp.ImageRgb()

        self.process = True
        self.model = None
        self.byte_tracker = None

        self.module_name = "faceDetection"
        self.stream_name = "alwayson/faceDetection"
        self.rpc_name = "faceDetection"
        self.source_port = "/icub/cam/left"
        self.conf_threshold = 0.7
        self.track = True
        self.identify_faces = True
        self.tolerance = 0.55
        self.verbose = False
        self.debug = False

        self.known_faces = {}
        self.tracked_faces = {}
        self.objects = []
        self.last_frame = None
        self.faces_path = None

        self.face_recognition_available = False
        self.face_recognition = None
        self.auto_download_model = True
        self.fallback_models = ["yolov8n-face.pt"]
        self._warned_non_face_model = False

    def _pip_install(self, package: str) -> bool:
        python_exec = sys.executable

        print(f"[INFO] Installing {package} using {python_exec}")

        try:
            subprocess.run(
                [python_exec, "-m", "pip", "install", package],
                check=True,
            )
            return True
        except Exception as err:
            print(f"\033[93m[WARNING] Failed to install {package}: {err}\033[00m")
            return False

    def _install_face_recognition_stack(self, install_face_lib: bool, install_models: bool) -> bool:
        ok = True

        ok = self._pip_install("setuptools") and ok

        if install_face_lib:
            ok = self._pip_install("face-recognition") and ok

        if install_models:
            ok = self._pip_install("git+https://github.com/ageitgey/face_recognition_models") and ok

        return ok

    def _initialize_face_recognition(self):
        error_text = ""
        missing_models = False
        missing_face_library = False

        try:
            import face_recognition as fr

            self.face_recognition_available = True
            self.face_recognition = fr
            return
        except (Exception, SystemExit) as err:
            error_text = str(err) if err is not None else ""
            err_lower = error_text.lower()
            missing_models = isinstance(err, SystemExit) or (
                "face_recognition_models" in err_lower
            )
            missing_face_library = (
                "no module named 'face_recognition'" in err_lower
                or "no module named \"face_recognition\"" in err_lower
            )
            self.face_recognition_available = False
            print(f"\033[93m[WARNING] face_recognition unavailable: {error_text}\033[00m")

        if not (missing_face_library or missing_models):
            return

        print("\033[93m[WARNING] Attempting auto-install for face recognition dependencies...\033[00m")

        if not self._install_face_recognition_stack(
            install_face_lib=missing_face_library,
            install_models=missing_models or missing_face_library,
        ):
            return

        try:
            import face_recognition as fr

            self.face_recognition_available = True
            self.face_recognition = fr
            print("[INFO] face_recognition_models installed successfully")
        except (Exception, SystemExit) as err:
            self.face_recognition_available = False
            print(f"\033[93m[WARNING] face_recognition still unavailable after install: {err}\033[00m")

    def _resolve_model_path(self, model_path: str):
        candidates = []

        if os.path.isabs(model_path):
            candidates.append(model_path)
        else:
            current_script_folder = os.path.dirname(os.path.abspath(__file__))
            candidates.extend(
                [
                    model_path,
                    os.path.join(current_script_folder, model_path),
                    os.path.join(current_script_folder, "model", model_path),
                    os.path.join(os.getcwd(), model_path),
                ]
            )

        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        return None

    def _build_model_candidates(self, model_path: str, fallback_models_cfg: str):
        candidates = [model_path]

        if fallback_models_cfg:
            fallback_models = [m.strip() for m in fallback_models_cfg.split(",") if m.strip()]
            for fallback in fallback_models:
                if fallback not in candidates:
                    candidates.append(fallback)

        return candidates

    def _filter_face_detections(self, detections, names):
        if len(detections) == 0 or detections.class_id is None:
            return detections

        face_class_ids = []
        if isinstance(names, dict):
            face_class_ids = [class_id for class_id, label in names.items() if str(label).lower() == "face"]
        elif isinstance(names, list):
            face_class_ids = [idx for idx, label in enumerate(names) if str(label).lower() == "face"]

        if not face_class_ids:
            if not self._warned_non_face_model:
                print(
                    "\033[93m[WARNING] Loaded model has no 'face' class label. "
                    "Dropping all detections to enforce face-only output. "
                    "Use a face model for valid face boxes.\033[00m"
                )
                self._warned_non_face_model = True
            return detections[np.zeros(len(detections), dtype=bool)]

        return detections[np.isin(detections.class_id, face_class_ids)]

    def _initialize_yolo_model(self, model_candidates):
        last_error = None

        for candidate in model_candidates:
            resolved_path = self._resolve_model_path(candidate)

            if resolved_path is not None:
                model_source = resolved_path
            elif self.auto_download_model:
                model_source = candidate
            else:
                print(
                    f"\033[93m[WARNING] Model candidate not found locally: {candidate}\033[00m"
                )
                continue

            try:
                self.model = YOLO(model_source)
                if resolved_path is not None:
                    print(f"[INFO] Loaded YOLO model: {resolved_path}")
                else:
                    print(f"[INFO] Loaded YOLO model via Ultralytics source: {candidate}")
                return True
            except Exception as err:
                last_error = err
                print(
                    f"\033[93m[WARNING] Failed to load model candidate {candidate}: {err}\033[00m"
                )

        candidates_text = ", ".join(model_candidates)
        print(
            "\033[91m[ERROR] Could not load any YOLO model candidate: "
            f"{candidates_text}\033[00m"
        )
        if last_error is not None:
            print(f"\033[91m[ERROR] Last YOLO load error: {last_error}\033[00m")
        return False

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        self.process = rf.check("process", yarp.Value(True)).asBool()
        self.module_name = rf.check("name", yarp.Value("faceDetection")).asString()
        self.stream_name = rf.check("stream_name", yarp.Value("alwayson/faceDetection")).asString()
        if not self.stream_name:
            self.stream_name = "alwayson/faceDetection"

        default_rpc_name = self.module_name.split("/")[-1] if self.module_name else "faceDetection"
        self.rpc_name = rf.check("rpc_name", yarp.Value(default_rpc_name)).asString()
        self.source_port = rf.check("source_port", yarp.Value("/icub/cam/left")).asString()
        self.conf_threshold = rf.check("conf_threshold", yarp.Value(0.7)).asFloat32()
        self.track = rf.check("track", yarp.Value(True)).asBool()
        self.identify_faces = rf.check("identify_faces", yarp.Value(True)).asBool()
        self.tolerance = rf.check("id_tolerance", yarp.Value(0.55)).asFloat32()
        self.verbose = rf.check("verbose", yarp.Value(False)).asBool()
        self.debug = rf.check("debug", yarp.Value(False)).asBool()
        self.auto_download_model = rf.check("auto_download_model", yarp.Value(True)).asBool()

        current_script_folder = os.path.dirname(os.path.abspath(__file__))
        default_faces_path = os.path.abspath(
            os.path.join(current_script_folder, "faces")
        )
        faces_path = rf.check("faces_path", yarp.Value(default_faces_path)).asString()
        self.faces_path = faces_path

        model_path = rf.check("model", yarp.Value(self.default_face_model)).asString()
        fallback_models_cfg = rf.check(
            "fallback_models",
            yarp.Value(",".join(self.fallback_models)),
        ).asString()
        model_candidates = self._build_model_candidates(model_path, fallback_models_cfg)

        self._initialize_face_recognition()

        if not self._initialize_yolo_model(model_candidates):
            return False

        if self.track:
            self.byte_tracker = sv.ByteTrack(
                track_activation_threshold=self.conf_threshold,
                lost_track_buffer=120,
            )

        if self.identify_faces:
            if not self.face_recognition_available:
                print("\033[93m[WARNING] identify_faces requested but face_recognition is unavailable\033[00m")
            else:
                self.known_faces = self._load_known_faces(faces_path)
                print(
                    f"[INFO] Face ID enabled with {len(self.known_faces)} identities from {faces_path}"
                )

        self.handle_port.open("/" + self.rpc_name)
        self.input_port.open("/" + self.stream_name + "/image:i")
        self.output_img_port.open("/" + self.stream_name + "/faces_view:o")
        self.output_objects_port.open("/" + self.stream_name + "/faces:o")

        self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)
        self.display_buf_image.resize(self.width_img, self.height_img)

        print(f"[INFO] Running {self.module_name}")
        print(f"[INFO] RPC port: /{self.rpc_name}")
        print(f"[INFO] Input port: /{self.stream_name}/image:i")
        print(f"[INFO] Output image port: /{self.stream_name}/faces_view:o")
        print(f"[INFO] Output bottle port: /{self.stream_name}/faces:o")
        return True

    def _load_known_faces(self, faces_path: str):
        database = {}
        if not os.path.exists(faces_path):
            print(f"\033[93m[WARNING] Faces folder does not exist: {faces_path}\033[00m")
            return database

        for filename in os.listdir(faces_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            file_path = os.path.join(faces_path, filename)
            image = self.face_recognition.load_image_file(file_path)
            encoding = self.face_recognition.face_encodings(image, model="large")
            if len(encoding) > 0:
                person_name = os.path.splitext(filename)[0]
                database[person_name] = encoding[0]

        return database

    def _compare_embeddings(self, frame_bgr, box):
        if not self.face_recognition_available:
            return "unknown", 0.0

        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame_bgr.shape[:2]

        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return "recognizing", 0.0

        cropped_frame = frame_bgr[y1:y2, x1:x2]
        if cropped_frame.size == 0:
            return "recognizing", 0.0

        cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        unknown_encoding = self.face_recognition.face_encodings(
            np.array(cropped_rgb), model="large"
        )
        if len(unknown_encoding) == 0:
            return "recognizing", 0.0

        if not self.known_faces:
            return "unknown", 0.0

        unknown_encoding = unknown_encoding[0]
        known_names = list(self.known_faces.keys())
        known_encodings = list(self.known_faces.values())

        face_distances = self.face_recognition.face_distance(known_encodings, unknown_encoding)
        id_results = self.face_recognition.compare_faces(
            known_encodings,
            unknown_encoding,
            tolerance=self.tolerance,
        )

        if np.any(id_results):
            best_idx = int(np.argmin(face_distances))
            if id_results[best_idx]:
                confidence = 1.0 - float(face_distances[best_idx])
                return known_names[best_idx], confidence

        return "unknown", 0.0

    def _handle_face_naming(self, command, reply):
        """Handle runtime face naming command: name <person_name> id <track_id>."""
        reply.clear()

        if command.size() != 4:
            reply.addString("nack")
            reply.addString("Usage: name <person_name> id <track_id>")
            return

        if command.get(0).asString() != "name" or command.get(2).asString() != "id":
            reply.addString("nack")
            reply.addString("Usage: name <person_name> id <track_id>")
            return

        if not self.track:
            reply.addString("nack")
            reply.addString("Face naming requires --track true")
            return

        if not self.identify_faces:
            reply.addString("nack")
            reply.addString("Face naming requires --identify_faces true")
            return

        if not self.face_recognition_available:
            reply.addString("nack")
            reply.addString("face_recognition library is not available")
            return

        if self.last_frame is None:
            reply.addString("nack")
            reply.addString("No frame available yet")
            return

        person_name = command.get(1).asString()
        track_id = command.get(3).asInt32()

        target_obj = None
        for obj in self.objects:
            if "track_id" in obj and obj["track_id"] == track_id:
                target_obj = obj
                break

        if target_obj is None:
            reply.addString("nack")
            reply.addString(f"Track ID {track_id} not found in current detections")
            return

        box = target_obj["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        h, w = self.last_frame.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            reply.addString("nack")
            reply.addString("Invalid bounding box")
            return

        face_crop = self.last_frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            reply.addString("nack")
            reply.addString("Empty face crop")
            return

        if self.faces_path is None:
            reply.addString("nack")
            reply.addString("Faces path is not configured")
            return

        os.makedirs(self.faces_path, exist_ok=True)
        face_path = os.path.join(self.faces_path, f"{person_name}.jpg")
        cv2.imwrite(face_path, face_crop)

        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        encoding = self.face_recognition.face_encodings(face_crop_rgb, model="large")

        if len(encoding) == 0:
            reply.addString("nack")
            reply.addString("Could not extract face encoding from crop")
            return

        self.known_faces[person_name] = encoding[0]
        self.tracked_faces[track_id] = (person_name, 1.0)

        reply.addString("ok")
        reply.addString(face_path)

    def _get_image_from_port(self, yarp_image):
        if yarp_image.width() != self.width_img or yarp_image.height() != self.height_img:
            self.width_img = yarp_image.width()
            self.height_img = yarp_image.height()
            self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8)

        yarp_image.setExternal(self.input_img_array.data, self.width_img, self.height_img)
        frame = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
            (self.height_img, self.width_img, 3)
        )
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _dict2bottle(self, key, value):
        bottle = yarp.Bottle()
        bottle.addString(key)

        if isinstance(value, int):
            bottle.addInt32(value)
        elif isinstance(value, float):
            bottle.addFloat64(value)
        elif isinstance(value, str):
            bottle.addString(value)
        elif isinstance(value, (list, np.ndarray)):
            b_list = yarp.Bottle()
            for element in value:
                if isinstance(element, int):
                    b_list.addInt32(element)
                elif isinstance(element, float):
                    b_list.addFloat64(element)
                elif isinstance(element, str):
                    b_list.addString(element)
            bottle.addList().read(b_list)

        return bottle

    def _write_objects(self, objects):
        if len(objects) == 0:
            return

        output_bottle = yarp.Bottle()
        obj_bottle = yarp.Bottle()

        for obj in objects:
            obj_bottle.addList().read(self._dict2bottle("class", str(obj["class"])))
            obj_bottle.addList().read(self._dict2bottle("score", float(obj["score"])))

            if "track_id" in obj:
                obj_bottle.addList().read(self._dict2bottle("track_id", int(obj["track_id"])))

            obj_bottle.addList().read(self._dict2bottle("box", obj["box"]))

            if "face_id" in obj:
                obj_bottle.addList().read(self._dict2bottle("face_id", str(obj["face_id"])))
            if "id_confidence" in obj:
                obj_bottle.addList().read(
                    self._dict2bottle("id_confidence", float(obj["id_confidence"]))
                )

            output_bottle.addList().read(obj_bottle)
            obj_bottle.clear()

        self.output_objects_port.write(output_bottle)

    def _write_annotated_image(self, annotated_image_bgr):
        annotated_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
        self.display_buf_image = self.output_img_port.prepare()
        self.display_buf_image.resize(self.width_img, self.height_img)
        self.display_buf_image.setExternal(
            annotated_rgb.tobytes(),
            self.width_img,
            self.height_img,
        )
        self.output_img_port.write()

    def updateModule(self):
        if not self.process:
            return True

        message = self.input_port.read(False)
        if message is None:
            return True

        frame = self._get_image_from_port(message)
        self.last_frame = frame.copy()

        result = self.model(frame, verbose=self.verbose)[0]
        detections = sv.Detections.from_ultralytics(result)

        if self.conf_threshold:
            detections = detections[detections.confidence > self.conf_threshold]

        detections = self._filter_face_detections(detections, result.names)

        # Expand bounding boxes for face crops
        if len(detections) > 0:
            xyxy = detections.xyxy
            widths = xyxy[:, 2] - xyxy[:, 0]
            heights = xyxy[:, 3] - xyxy[:, 1]

            # 10% expansion per side
            expand_w = widths * 0.10
            expand_h = heights * 0.10

            xyxy[:, 0] -= expand_w
            xyxy[:, 1] -= expand_h
            xyxy[:, 2] += expand_w
            xyxy[:, 3] += expand_h

            xyxy[:, 0] = np.maximum(xyxy[:, 0], 0)
            xyxy[:, 1] = np.maximum(xyxy[:, 1], 0)
            xyxy[:, 2] = np.minimum(xyxy[:, 2], frame.shape[1])
            xyxy[:, 3] = np.minimum(xyxy[:, 3], frame.shape[0])

            detections.xyxy = xyxy

        if self.track and len(detections) > 0:
            detections = self.byte_tracker.update_with_detections(detections)

        labels = [result.names[class_id] for class_id in detections.class_id] if len(detections) > 0 else []

        if self.identify_faces and self.track and len(detections) > 0 and detections.tracker_id is not None:
            current_ids = set(detections.tracker_id)
            lost_ids = [tid for tid in self.tracked_faces.keys() if tid not in current_ids]
            for tid in lost_ids:
                del self.tracked_faces[tid]

            for tid, box in zip(detections.tracker_id, detections.xyxy):
                if tid not in self.tracked_faces:
                    face_id, id_conf = self._compare_embeddings(frame, box)
                    self.tracked_faces[tid] = (face_id, id_conf)
                else:
                    cached_name, _ = self.tracked_faces[tid]
                    if cached_name == "recognizing":
                        face_id, id_conf = self._compare_embeddings(frame, box)
                        self.tracked_faces[tid] = (face_id, id_conf)

        objects = []
        for i in range(len(detections)):
            obj = {
                "class": labels[i],
                "score": float(detections.confidence[i]),
                "box": detections.xyxy[i].tolist(),
            }

            if self.track and detections.tracker_id is not None:
                track_id = detections.tracker_id[i]
                if track_id is not None:
                    obj["track_id"] = int(track_id)
                    if self.identify_faces and track_id in self.tracked_faces:
                        face_id, id_conf = self.tracked_faces[track_id]
                        obj["face_id"] = face_id
                        obj["id_confidence"] = float(id_conf)

            objects.append(obj)

        self.objects = objects

        display_labels = labels.copy()
        if self.identify_faces and self.track and len(detections) > 0 and detections.tracker_id is not None:
            display_labels = []
            for label, track_id in zip(labels, detections.tracker_id):
                if track_id in self.tracked_faces:
                    face_id, _ = self.tracked_faces[track_id]
                    if face_id != "recognizing":
                        display_labels.append(face_id)
                    else:
                        display_labels.append("recognizing...")
                else:
                    display_labels.append(label)

        if len(detections) > 0:
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
            annotated_image = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_image = label_annotator.annotate(
                scene=annotated_image,
                detections=detections,
                labels=display_labels,
            )
        else:
            annotated_image = frame

        if self.output_objects_port.getOutputCount() or self.debug:
            self._write_objects(objects)
        if self.output_img_port.getOutputCount():
            self._write_annotated_image(annotated_image)
        return True

    def respond(self, command, reply):
        reply.clear()
        if command.get(0).asString() == "quit":
            reply.addString("quitting")
            return False
        if command.get(0).asString() == "process":
            if command.size() < 2:
                reply.addString("nack")
                reply.addString("Usage: process on/off")
                return True
            self.process = True if command.get(1).asString() == "on" else False
            reply.addString("ok")
            return True
        if command.get(0).asString() == "name":
            self._handle_face_naming(command, reply)
            return True
        if command.get(0).asString() == "help":
            reply.addString("Commands: quit | process on/off | name <person_name> id <track_id>")
            return True

        reply.addString("nack")
        return True

    def getPeriod(self):
        return 0.02

    def interruptModule(self):
        self.handle_port.interrupt()
        self.input_port.interrupt()
        self.output_img_port.interrupt()
        self.output_objects_port.interrupt()
        return True

    def close(self):
        self.handle_port.close()
        self.input_port.close()
        self.output_img_port.close()
        self.output_objects_port.close()
        return True


if __name__ == "__main__":
    if not yarp.Network.checkNetwork():
        print("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()

    module = FaceDetection()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)

    if rf.configure(sys.argv):
        module.runModule(rf)

    yarp.Network.fini()
    sys.exit()
