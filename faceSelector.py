"""
faceSelector.py - YARP RFModule for Real-Time Face Selection

This module continuously reads face landmarks, computes social/spatial/learning states,
selects the best candidate face, and triggers interactions via /interactionManager RPC.

YARP Connections (run after starting):
    yarp connect /alwayson/vision/landmarks:o /faceSelector/landmarks:i
    yarp connect /icub/camcalib/left/out /faceSelector/img:i
"""

import json
import os
import sys
import tempfile
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: OpenCV and NumPy are required. Install with: pip install opencv-python numpy")
    sys.exit(1)

try:
    import yarp
except ImportError:
    print("ERROR: YARP Python bindings are required.")
    sys.exit(1)

# Timezone support
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


class FaceSelectorModule(yarp.RFModule):
    """
    Real-time face selection module that:
    - Reads face landmarks from vision system
    - Computes social states (SS1-SS5) and spatial states
    - Maintains learning states (LS1-LS4) per person
    - Selects best candidate face based on priority rules
    - Triggers interactions via /interactionManager RPC
    - Publishes annotated image with face boxes and states
    """

    # ==================== Constants ====================

    # Social State definitions
    SS1 = 1  # Unknown, Not Greeted Today
    SS2 = 2  # Unknown, Greeted Today
    SS3 = 3  # Known, Not Greeted Today
    SS4 = 4  # Known, Greeted Today, Not talked to
    SS5 = 5  # Known, Greeted Today, Talked to

    SS_NAMES = {1: "SS1", 2: "SS2", 3: "SS3", 4: "SS4", 5: "SS5"}
    SS_DESCRIPTIONS = {
        1: "Unknown, Not Greeted",
        2: "Unknown, Greeted",
        3: "Known, Not Greeted",
        4: "Known, Greeted, No Talk",
        5: "Known, Greeted, Talked"
    }

    # Learning State definitions
    LS1 = 1  # Any spatial state allowed
    LS2 = 2  # Zone: LEFT/CENTER/RIGHT, Distance: SO_CLOSE/CLOSE/FAR, any attention
    LS3 = 3  # Zone: LEFT/CENTER/RIGHT, Distance: SO_CLOSE/CLOSE, Attention: MUTUAL_GAZE/NEAR_GAZE
    LS4 = 4  # Zone: LEFT/CENTER/RIGHT, Distance: SO_CLOSE/CLOSE, Attention: MUTUAL_GAZE only

    LS_NAMES = {1: "LS1", 2: "LS2", 3: "LS3", 4: "LS4"}

    # Valid zones/distances/attentions for each LS
    LS_VALID_ZONES = {
        1: {"FAR_LEFT", "LEFT", "CENTER", "RIGHT", "FAR_RIGHT", "UNKNOWN"},
        2: {"LEFT", "CENTER", "RIGHT"},
        3: {"LEFT", "CENTER", "RIGHT"},
        4: {"LEFT", "CENTER", "RIGHT"}
    }
    LS_VALID_DISTANCES = {
        1: {"SO_CLOSE", "CLOSE", "FAR", "VERY_FAR", "UNKNOWN"},
        2: {"SO_CLOSE", "CLOSE", "FAR"},
        3: {"SO_CLOSE", "CLOSE"},
        4: {"SO_CLOSE", "CLOSE"}
    }
    LS_VALID_ATTENTIONS = {
        1: {"MUTUAL_GAZE", "NEAR_GAZE", "AWAY"},
        2: {"MUTUAL_GAZE", "NEAR_GAZE", "AWAY"},
        3: {"MUTUAL_GAZE", "NEAR_GAZE"},
        4: {"MUTUAL_GAZE"}
    }

    # Attention priority (higher = better)
    ATTENTION_PRIORITY = {"MUTUAL_GAZE": 3, "NEAR_GAZE": 2, "AWAY": 1}
    
    # Distance priority (higher = better)
    DISTANCE_PRIORITY = {"SO_CLOSE": 4, "CLOSE": 3, "FAR": 2, "VERY_FAR": 1, "UNKNOWN": 0}

    # Colors for drawing (BGR format)
    COLOR_GREEN = (0, 255, 0)      # Selected/active target
    COLOR_YELLOW = (0, 255, 255)   # Eligible faces
    COLOR_WHITE = (255, 255, 255)  # Non-eligible faces
    COLOR_RED = (0, 0, 255)        # Errors/blocked

    # Timezone for "today" computation
    TIMEZONE = ZoneInfo("Europe/Rome")

    # ==================== Lifecycle ====================

    def __init__(self):
        super().__init__()
        
        # Module configuration
        self.module_name = "faceSelector"
        self.period = 0.05  # 20 Hz
        self._running = True

        # RPC target names (configurable)
        self.interaction_manager_rpc_name = "/interactionManager"
        self.interaction_interface_rpc_name = "/interactionInterface"

        # File paths (configurable)
        self.learning_path = Path("./learning.json")
        self.greeted_path = Path("./greeted_today.json")
        self.talked_path = Path("./talked_today.json")

        # YARP ports
        self.landmarks_port: Optional[yarp.BufferedPortBottle] = None
        self.img_in_port: Optional[yarp.BufferedPortImageRgb] = None
        self.img_out_port: Optional[yarp.BufferedPortImageRgb] = None
        self.debug_port: Optional[yarp.Port] = None
        self.interaction_manager_rpc: Optional[yarp.RpcClient] = None
        self.interaction_interface_rpc: Optional[yarp.RpcClient] = None

        # Image handling
        self.img_width = 640
        self.img_height = 480
        self.last_annotated_frame: Optional[np.ndarray] = None

        # State tracking (thread-safe)
        self.state_lock = threading.Lock()
        self.current_faces: List[Dict[str, Any]] = []
        self.selected_target: Optional[Dict[str, Any]] = None
        self.selected_bbox_last: Optional[Tuple[float, float, float, float]] = None  # Keep green box visible
        self.interaction_busy = False
        self.interaction_thread: Optional[threading.Thread] = None

        # Memory caches (loaded from JSON files)
        self.greeted_today: Dict[str, str] = {}  # person_id -> ISO timestamp
        self.talked_today: Dict[str, str] = {}   # person_id -> ISO timestamp
        self.learning_data: Dict[str, Dict] = {} # person_id -> {"ls": int, "updated_at": str}
        
        # Session tracking (track_id -> stable person_id mapping)
        self.track_to_person: Dict[int, str] = {}
        
        # Day tracking for automatic pruning at midnight
        self._current_day: Optional[date] = None

        # Configuration flags
        self.allow_ss5_selection = False  # By default, SS5 faces are not selected

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        """Configure module from ResourceFinder parameters."""
        try:
            # Read configuration parameters
            if rf.check("name"):
                self.module_name = rf.find("name").asString()
            
            # Set official YARP module name
            try:
                self.setName(self.module_name)
            except Exception:
                pass
            
            if rf.check("interaction_manager_rpc"):
                self.interaction_manager_rpc_name = rf.find("interaction_manager_rpc").asString()
            
            if rf.check("interaction_interface_rpc"):
                self.interaction_interface_rpc_name = rf.find("interaction_interface_rpc").asString()
            
            if rf.check("learning_path"):
                self.learning_path = Path(rf.find("learning_path").asString())
            
            if rf.check("greeted_path"):
                self.greeted_path = Path(rf.find("greeted_path").asString())
            
            if rf.check("talked_path"):
                self.talked_path = Path(rf.find("talked_path").asString())
            
            if rf.check("rate"):
                self.period = rf.find("rate").asFloat64()
            
            if rf.check("allow_ss5"):
                self.allow_ss5_selection = rf.find("allow_ss5").asBool()

            # Open input ports
            self.landmarks_port = yarp.BufferedPortBottle()
            if not self.landmarks_port.open(f"/{self.module_name}/landmarks:i"):
                self._log("ERROR", "Failed to open landmarks input port")
                return False

            self.img_in_port = yarp.BufferedPortImageRgb()
            if not self.img_in_port.open(f"/{self.module_name}/img:i"):
                self._log("ERROR", "Failed to open image input port")
                return False

            # Open output ports
            self.img_out_port = yarp.BufferedPortImageRgb()
            if not self.img_out_port.open(f"/{self.module_name}/img:o"):
                self._log("ERROR", "Failed to open image output port")
                return False

            self.debug_port = yarp.Port()
            if not self.debug_port.open(f"/{self.module_name}/debug:o"):
                self._log("ERROR", "Failed to open debug output port")
                return False

            # Open RPC client ports
            self.interaction_manager_rpc = yarp.RpcClient()
            if not self.interaction_manager_rpc.open(f"/{self.module_name}/interactionManager:rpc"):
                self._log("ERROR", "Failed to open interactionManager RPC client port")
                return False

            self.interaction_interface_rpc = yarp.RpcClient()
            if not self.interaction_interface_rpc.open(f"/{self.module_name}/interactionInterface:rpc"):
                self._log("ERROR", "Failed to open interactionInterface RPC client port")
                return False

            # Automatically connect RPC client ports
            self._log("INFO", "Attempting to connect RPC ports...")
            
            # Connect to interactionManager
            if not yarp.Network.connect(f"/{self.module_name}/interactionManager:rpc", 
                                        self.interaction_manager_rpc_name):
                self._log("ERROR", f"Failed to connect to {self.interaction_manager_rpc_name}")
                self._log("ERROR", "Make sure interactionManager is running")
            else:
                self._log("INFO", f"Connected to {self.interaction_manager_rpc_name}")
            
            # Connect to interactionInterface
            if not yarp.Network.connect(f"/{self.module_name}/interactionInterface:rpc", 
                                        self.interaction_interface_rpc_name):
                self._log("ERROR", f"Failed to connect to {self.interaction_interface_rpc_name}")
                self._log("ERROR", "Make sure interactionInterface is running")
            else:
                self._log("INFO", f"Connected to {self.interaction_interface_rpc_name}")

            # Load persistent data
            self._load_all_json_files()
            
            # Initialize current day tracking
            self._current_day = self._get_today_date()
            self._log("INFO", f"Initialized current day: {self._current_day}")

            self._log("INFO", f"FaceSelectorModule configured successfully")
            self._log("INFO", f"  Module name: {self.module_name}")
            self._log("INFO", f"  Rate: {self.period}s ({1.0/self.period:.1f} Hz)")
            
            return True

        except Exception as e:
            self._log("ERROR", f"Configuration failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def interruptModule(self) -> bool:
        """Interrupt all ports."""
        self._log("INFO", "Interrupting module...")
        self._running = False
        
        for port in [self.landmarks_port, self.img_in_port, self.img_out_port, 
                     self.debug_port, self.interaction_manager_rpc, self.interaction_interface_rpc]:
            if port:
                port.interrupt()
        
        return True

    def close(self) -> bool:
        """Close all ports and save state."""
        self._log("INFO", "Closing module...")
        
        # Wait for interaction thread to finish
        if self.interaction_thread and self.interaction_thread.is_alive():
            self._log("INFO", "Waiting for interaction thread to finish...")
            self.interaction_thread.join(timeout=5.0)
        
        # Save all JSON files
        self._save_all_json_files()
        
        # Close ports
        for port in [self.landmarks_port, self.img_in_port, self.img_out_port, 
                     self.debug_port, self.interaction_manager_rpc, self.interaction_interface_rpc]:
            if port:
                port.close()
        
        return True

    def getPeriod(self) -> float:
        return self.period

    def updateModule(self) -> bool:
        """Main update loop - runs at configured rate."""
        if not self._running:
            return False
        
        try:
            # Check if day has changed and prune if needed
            today = self._get_today_date()
            if self._current_day != today:
                self._log("INFO", f"=== DAY CHANGE: {self._current_day} → {today} ===")
                self._log("INFO", "Pruning old interaction records...")
                with self.state_lock:
                    self.greeted_today = self._prune_to_today(self.greeted_today)
                    self.talked_today = self._prune_to_today(self.talked_today)
                self._save_greeted_json()
                self._save_talked_json()
                self._current_day = today
                self._log("INFO", f"Pruning complete. Greeted: {len(self.greeted_today)}, Talked: {len(self.talked_today)}")
            
            # 1. Read latest landmarks (non-blocking)
            faces = self._read_landmarks()
            if faces:
                self._log("DEBUG", f"Step 1/6: Read {len(faces)} face(s) from landmarks")
            
            # 2. Read latest image (non-blocking)
            frame = self._read_image()
            if frame is not None:
                self._log("DEBUG", f"Step 2/6: Read image frame {frame.shape}")
            
            # 3. Compute states for all faces
            with self.state_lock:
                self.current_faces = self._compute_face_states(faces)
            if faces:
                self._log("DEBUG", f"Step 3/6: Computed states for {len(self.current_faces)} face(s)")
            
            # 4. Select target face (if not busy with interaction)
            with self.state_lock:
                if not self.interaction_busy:
                    candidate = self._select_best_face(self.current_faces)
                    if candidate:
                        self.selected_target = candidate
                        self.selected_bbox_last = candidate["bbox"]  # Store for green box persistence
                        self.interaction_busy = True
                        
                        face_id = candidate.get("face_id", "unknown")
                        track_id = candidate.get("track_id", -1)
                        ss = self.SS_NAMES.get(candidate.get("social_state", 0), "?")
                        ls = self.LS_NAMES.get(candidate.get("learning_state", 1), "?")
                        self._log("INFO", f"=== SELECTED TARGET: {face_id} (track={track_id}, {ss}, {ls}) ===")
                        
                        # Start interaction in background thread
                        self.interaction_thread = threading.Thread(
                            target=self._run_interaction_thread,
                            args=(candidate,),
                            daemon=True
                        )
                        self.interaction_thread.start()
                        self._log("DEBUG", "Step 4/6: Started interaction thread")
                    else:
                        self._log("DEBUG", "Step 4/6: No eligible face selected")
                else:
                    self._log("DEBUG", "Step 4/6: Interaction busy, skipping selection")
            
            # 5. Annotate and publish image
            if frame is not None:
                with self.state_lock:
                    annotated = self._annotate_image(frame, self.current_faces, self.selected_target)
                self.last_annotated_frame = annotated
                self._publish_image(annotated)
                self._log("DEBUG", "Step 5/6: Published annotated image")
            elif self.last_annotated_frame is not None:
                # Republish last known frame if no new image
                self._publish_image(self.last_annotated_frame)
            
            # 6. Publish debug info
            self._publish_debug()
            self._log("DEBUG", "Step 6/6: Published debug info")
            
        except Exception as e:
            self._log("ERROR", f"Error in updateModule: {e}")
            import traceback
            traceback.print_exc()
        
        return True

    # ==================== Landmark Parsing ====================

    def _read_landmarks(self) -> List[Dict[str, Any]]:
        """Read and parse landmarks from YARP port (non-blocking)."""
        faces = []
        
        bottle = self.landmarks_port.read(False)  # Non-blocking read
        if not bottle:
            return faces
        
        # Each outer element is a face
        for i in range(bottle.size()):
            face_btl = bottle.get(i)
            if not face_btl.isList():
                continue
            
            face_data = self._parse_face_bottle(face_btl.asList())
            if face_data:
                faces.append(face_data)
        
        return faces

    def _parse_face_bottle(self, bottle: yarp.Bottle) -> Optional[Dict[str, Any]]:
        """Parse a single face bottle into a dictionary."""
        if not bottle:
            return None
        
        data = {
            "face_id": "unknown",
            "track_id": -1,
            "bbox": (0.0, 0.0, 0.0, 0.0),  # x, y, w, h
            "zone": "UNKNOWN",
            "distance": "UNKNOWN",
            "gaze_direction": (0.0, 0.0, 1.0),
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "cos_angle": 0.0,
            "attention": "AWAY",
            "is_talking": 0,
            "time_in_view": 0.0
        }
        
        try:
            i = 0
            while i < bottle.size():
                item = bottle.get(i)
                
                # Handle key-value pairs: "key" value
                if item.isString():
                    key = item.asString()
                    
                    if i + 1 < bottle.size():
                        next_item = bottle.get(i + 1)
                        
                        if key == "face_id" and next_item.isString():
                            data["face_id"] = next_item.asString()
                            i += 2
                        elif key == "track_id" and (next_item.isInt32() or next_item.isInt64()):
                            data["track_id"] = next_item.asInt32()
                            i += 2
                        elif key == "zone" and next_item.isString():
                            data["zone"] = next_item.asString()
                            i += 2
                        elif key == "distance" and next_item.isString():
                            data["distance"] = next_item.asString()
                            i += 2
                        elif key == "attention" and next_item.isString():
                            data["attention"] = next_item.asString()
                            i += 2
                        elif key == "pitch" and next_item.isFloat64():
                            data["pitch"] = next_item.asFloat64()
                            i += 2
                        elif key == "yaw" and next_item.isFloat64():
                            data["yaw"] = next_item.asFloat64()
                            i += 2
                        elif key == "roll" and next_item.isFloat64():
                            data["roll"] = next_item.asFloat64()
                            i += 2
                        elif key == "cos_angle" and next_item.isFloat64():
                            data["cos_angle"] = next_item.asFloat64()
                            i += 2
                        elif key == "is_talking" and (next_item.isInt32() or next_item.isInt64()):
                            data["is_talking"] = next_item.asInt32()
                            i += 2
                        elif key == "time_in_view" and next_item.isFloat64():
                            data["time_in_view"] = next_item.asFloat64()
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1
                
                # Handle nested lists: ("key" v1 v2 ...) or ("bbox" x y w h)
                elif item.isList():
                    nested = item.asList()
                    if nested.size() >= 2:
                        key = nested.get(0).asString() if nested.get(0).isString() else ""
                        
                        if key == "bbox" and nested.size() >= 5:
                            x = nested.get(1).asFloat64()
                            y = nested.get(2).asFloat64()
                            w = nested.get(3).asFloat64()
                            h = nested.get(4).asFloat64()
                            data["bbox"] = (x, y, w, h)
                        elif key == "gaze_direction" and nested.size() >= 4:
                            gx = nested.get(1).asFloat64()
                            gy = nested.get(2).asFloat64()
                            gz = nested.get(3).asFloat64()
                            data["gaze_direction"] = (gx, gy, gz)
                    i += 1
                else:
                    i += 1
            
            return data
            
        except Exception as e:
            self._log("WARNING", f"Failed to parse face bottle: {e}")
            return None

    # ==================== Image Handling ====================

    def _read_image(self) -> Optional[np.ndarray]:
        """Read image from YARP port (non-blocking) and return RGB numpy array (stride-safe)."""
        yimg = self.img_in_port.read(False)
        if not yimg:
            return None
        
        w, h = yimg.width(), yimg.height()
        if w <= 0 or h <= 0:
            return None
        
        row = yimg.getRowSize()  # bytes per row (may include padding)
        if row < w * 3:
            self._log("WARNING", f"Image rowSize too small: row={row}, w*3={w*3}")
            return None
        
        # Use YARP's __array_interface__ to get numpy array directly
        try:
            # Create numpy array from YARP image using array interface
            img_array = np.asarray(yimg.__array_interface__)
            if img_array.size == 0:
                return None
            
            # Reshape to (height, rowSize) then extract valid pixels
            raw = img_array.reshape((h, row))
            rgb = raw[:, : w * 3].reshape((h, w, 3)).copy()
            
            self.img_width, self.img_height = w, h
            return rgb
        except Exception as e:
            self._log("WARNING", f"Failed to convert YARP image to numpy: {e}")
            return None

    def _publish_image(self, frame_rgb: np.ndarray):
        """Publish annotated image to YARP port (stride-safe)."""
        if self.img_out_port.getOutputCount() == 0:
            return
        
        try:
            h, w, _ = frame_rgb.shape
            out = self.img_out_port.prepare()
            out.resize(w, h)
            
            row = out.getRowSize()
            if row < w * 3:
                self._log("WARNING", f"Output rowSize too small: row={row}, w*3={w*3}")
                return
            
            # Use YARP's __array_interface__ to write numpy array directly
            try:
                out_array = np.asarray(out.__array_interface__)
                out_raw = out_array.reshape((h, row))
                
                # Write pixels; zero padding (optional but nice)
                out_raw[:, : w * 3] = frame_rgb.reshape((h, w * 3))
                if row > w * 3:
                    out_raw[:, w * 3 :] = 0
                
                self.img_out_port.write()
            except Exception as e:
                self._log("WARNING", f"Failed to write numpy array to YARP image: {e}")
                
        except Exception as e:
            self._log("WARNING", f"Failed to publish image: {e}")

    def _annotate_image(self, frame_rgb: np.ndarray, faces: List[Dict], selected: Optional[Dict]) -> np.ndarray:
        """Draw face boxes with state annotations."""
        # Convert RGB to BGR for OpenCV drawing (OpenCV uses BGR color space)
        annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        selected_track_id = selected["track_id"] if selected else None
        selected_found = False
        
        for face in faces:
            x, y, w, h = face["bbox"]
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Clamp bbox to image bounds
            x = max(0, min(x, self.img_width - 1))
            y = max(0, min(y, self.img_height - 1))
            w = max(0, min(w, self.img_width - x))
            h = max(0, min(h, self.img_height - y))
            
            track_id = face["track_id"]
            face_id = face["face_id"]
            
            # Determine box color
            if track_id == selected_track_id and self.interaction_busy:
                color = self.COLOR_GREEN
                selected_found = True
                self.selected_bbox_last = face["bbox"]  # keep it fresh
            elif face.get("eligible", False):
                color = self.COLOR_YELLOW
            else:
                color = self.COLOR_WHITE
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            ss = face.get("social_state", 0)
            ls = face.get("learning_state", 1)
            zone = face.get("zone", "?")
            distance = face.get("distance", "?")
            attention = face.get("attention", "?")
            
            ss_str = self.SS_NAMES.get(ss, "?")
            ls_str = self.LS_NAMES.get(ls, "?")
            
            # Line 1: face_id / track_id
            label1 = f"{face_id} (T:{track_id})"
            # Line 2: SS / LS
            label2 = f"{ss_str} | {ls_str}"
            # Line 3: Spatial state
            label3 = f"{zone}/{distance}/{attention[:3]}"
            
            # Draw labels with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            
            self._draw_label(annotated, label1, (x, y - 45), font, font_scale, color, thickness)
            self._draw_label(annotated, label2, (x, y - 28), font, font_scale, color, thickness)
            self._draw_label(annotated, label3, (x, y - 11), font, font_scale, color, thickness)
            
            # Show "ACTIVE" badge if selected
            if track_id == selected_track_id and self.interaction_busy:
                cv2.putText(annotated, "ACTIVE", (x + w - 55, y + 15), 
                           font, 0.5, self.COLOR_GREEN, 2)
        
        # If selected face disappeared during interaction, draw last known bbox
        if self.interaction_busy and not selected_found and self.selected_bbox_last:
            x, y, w, h = self.selected_bbox_last
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Clamp to bounds
            x = max(0, min(x, self.img_width - 1))
            y = max(0, min(y, self.img_height - 1))
            w = max(0, min(w, self.img_width - x))
            h = max(0, min(h, self.img_height - y))
            
            cv2.rectangle(annotated, (x, y), (x + w, y + h), self.COLOR_GREEN, 2)
            cv2.putText(annotated, "ACTIVE (LOST)", (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_GREEN, 2)
        
        # Draw status bar at top
        status = "BUSY" if self.interaction_busy else "IDLE"
        status_color = self.COLOR_GREEN if self.interaction_busy else self.COLOR_WHITE
        cv2.putText(annotated, f"Status: {status} | Faces: {len(faces)}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Convert back to RGB for YARP output
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    def _draw_label(self, img: np.ndarray, text: str, pos: Tuple[int, int], 
                    font, scale: float, color: Tuple[int, int, int], thickness: int):
        """Draw text label with dark background."""
        x, y = pos
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        # Background rectangle
        cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
        # Text
        cv2.putText(img, text, (x + 1, y), font, scale, color, thickness)

    # ==================== State Computation ====================

    def _compute_face_states(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute social, spatial, and learning states for all faces.
        
        Note: Reads and mutates self.track_to_person. Must be called under self.state_lock.
        """
        today = self._get_today_date()
        
        for face in faces:
            face_id = face["face_id"]
            track_id = face["track_id"]
            
            # Get person_id for lookups: prefer tracked stable ID, fallback to face_id
            person_id = self.track_to_person.get(track_id, face_id)
            
            # Determine if known: check both current face_id and stable person_id
            is_known = self._is_face_known(face_id) or self._is_face_known(person_id)
            face["is_known"] = is_known
            self._log("DEBUG", f"Face {face_id}: is_known={is_known}")
            
            # Check greeted today
            greeted_today = self._was_greeted_today(person_id, today)
            face["greeted_today"] = greeted_today
            self._log("DEBUG", f"Face {face_id}: greeted_today={greeted_today}")
            
            # Check talked today
            talked_today = self._was_talked_today(person_id, today)
            face["talked_today"] = talked_today
            self._log("DEBUG", f"Face {face_id}: talked_today={talked_today}")
            
            # Compute social state
            face["social_state"] = self._compute_social_state(is_known, greeted_today, talked_today)
            ss_name = self.SS_NAMES.get(face["social_state"], "?")
            self._log("DEBUG", f"Face {face_id}: social_state={ss_name}")
            
            # Get learning state
            face["learning_state"] = self._get_learning_state(person_id)
            ls_name = self.LS_NAMES.get(face["learning_state"], "?")
            self._log("DEBUG", f"Face {face_id}: learning_state={ls_name}")
            
            # Check eligibility based on learning state and spatial state
            face["eligible"] = self._is_eligible(face)
            zone = face.get("zone", "?")
            dist = face.get("distance", "?")
            attn = face.get("attention", "?")
            self._log("DEBUG", f"Face {face_id}: spatial=({zone}/{dist}/{attn}), eligible={face['eligible']}")
        
        # Prune track_to_person to prevent unbounded growth
        active_tracks = {f["track_id"] for f in faces if f.get("track_id", -1) >= 0}
        self.track_to_person = {tid: pid for tid, pid in self.track_to_person.items() if tid in active_tracks}
        
        return faces

    def _is_face_known(self, face_id: str) -> bool:
        """Determine if face is known based on face_id."""
        if not face_id or face_id.lower() == "unknown":
            return False
        
        # If it's a 5-digit number string, it's unknown (temporary code)
        if face_id.isdigit() and len(face_id) == 5:
            return False
        
        # Otherwise, it's a known person with a name
        return True

    def _was_greeted_today(self, person_id: str, today: date) -> bool:
        """Check if person was greeted today."""
        if person_id not in self.greeted_today:
            return False
        
        try:
            ts_str = self.greeted_today[person_id]
            greeted_dt = datetime.fromisoformat(ts_str)
            greeted_date = greeted_dt.astimezone(self.TIMEZONE).date()
            return greeted_date == today
        except Exception:
            return False

    def _was_talked_today(self, person_id: str, today: date) -> bool:
        """Check if person was talked to today."""
        if person_id not in self.talked_today:
            return False
        
        try:
            ts_str = self.talked_today[person_id]
            talked_dt = datetime.fromisoformat(ts_str)
            talked_date = talked_dt.astimezone(self.TIMEZONE).date()
            return talked_date == today
        except Exception:
            return False

    def _compute_social_state(self, is_known: bool, greeted_today: bool, talked_today: bool) -> int:
        """Compute social state (SS1-SS5) from known/greeted/talked flags."""
        if not is_known:
            if greeted_today:
                return self.SS2
            else:
                return self.SS1
        else:
            if not greeted_today:
                return self.SS3
            elif not talked_today:
                return self.SS4
            else:
                return self.SS5

    def _get_learning_state(self, person_id: str) -> int:
        """Get learning state for person (default LS1)."""
        if person_id in self.learning_data:
            return self.learning_data[person_id].get("ls", self.LS1)
        return self.LS1

    def _is_eligible(self, face: Dict[str, Any]) -> bool:
        """Check if face is eligible for selection based on learning state and spatial state."""
        ss = face.get("social_state", self.SS1)
        ls = face.get("learning_state", self.LS1)
        
        # SS5 faces are not eligible by default
        if ss == self.SS5 and not self.allow_ss5_selection:
            return False
        
        # Check spatial constraints based on learning state
        zone = face.get("zone", "UNKNOWN")
        distance = face.get("distance", "UNKNOWN")
        attention = face.get("attention", "AWAY")
        
        if zone not in self.LS_VALID_ZONES.get(ls, set()):
            return False
        if distance not in self.LS_VALID_DISTANCES.get(ls, set()):
            return False
        if attention not in self.LS_VALID_ATTENTIONS.get(ls, set()):
            return False
        
        return True

    # ==================== Face Selection ====================

    def _select_best_face(self, faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select best face based on priority rules."""
        # Filter to eligible faces only
        eligible = [f for f in faces if f.get("eligible", False)]
        self._log("DEBUG", f"Selection: {len(eligible)}/{len(faces)} faces eligible")
        
        if not eligible:
            # If no eligible faces and SS5 selection is disabled, check if we should allow SS5
            if not self.allow_ss5_selection:
                # Try SS5 faces as last resort (configurable)
                ss5_faces = [f for f in faces if f.get("social_state") == self.SS5]
                if ss5_faces:
                    self._log("DEBUG", f"Selection: No eligible faces, checking {len(ss5_faces)} SS5 faces")
                    # Still check spatial eligibility
                    ss5_eligible = [f for f in ss5_faces if self._check_spatial_only(f)]
                    if ss5_eligible:
                        self._log("DEBUG", f"Selection: {len(ss5_eligible)} SS5 faces spatially eligible")
                        eligible = ss5_eligible
            
            if not eligible:
                self._log("DEBUG", "Selection: No faces available for interaction")
                return None
        
        # Sort by priority:
        # 1. Social state (lower is better: SS1 > SS2 > SS3 > SS4 > SS5)
        # 2. Attention (MUTUAL_GAZE > NEAR_GAZE > AWAY)
        # 3. Distance (SO_CLOSE > CLOSE > FAR > VERY_FAR)
        # 4. time_in_view (higher is better)
        
        def sort_key(f: Dict) -> Tuple:
            ss = f.get("social_state", 5)
            attention = self.ATTENTION_PRIORITY.get(f.get("attention", "AWAY"), 0)
            distance = self.DISTANCE_PRIORITY.get(f.get("distance", "UNKNOWN"), 0)
            time_in_view = f.get("time_in_view", 0.0)
            
            # Lower SS is better (ascending), higher attention/distance/time is better (descending)
            return (ss, -attention, -distance, -time_in_view)
        
        eligible.sort(key=sort_key)
        best = eligible[0]
        
        ss_name = self.SS_NAMES.get(best.get("social_state", 0), "?")
        attn = best.get("attention", "?")
        dist = best.get("distance", "?")
        self._log("INFO", f"Selection: Best candidate - {best['face_id']} ({ss_name}, {attn}, {dist})")
        
        return best

    def _check_spatial_only(self, face: Dict[str, Any]) -> bool:
        """Check spatial eligibility for a face (ignoring SS)."""
        ls = face.get("learning_state", self.LS1)
        zone = face.get("zone", "UNKNOWN")
        distance = face.get("distance", "UNKNOWN")
        attention = face.get("attention", "AWAY")
        
        if zone not in self.LS_VALID_ZONES.get(ls, set()):
            return False
        if distance not in self.LS_VALID_DISTANCES.get(ls, set()):
            return False
        if attention not in self.LS_VALID_ATTENTIONS.get(ls, set()):
            return False
        
        return True

    # ==================== Interaction Execution ====================

    def _run_interaction_thread(self, target: Dict[str, Any]):
        """Run interaction in background thread."""
        try:
            track_id = target["track_id"]
            face_id = target["face_id"]
            ss = target.get("social_state", self.SS1)
            
            self._log("INFO", f"=== INTERACTION START: {face_id} (track={track_id}) ===")
            
            # Determine start state
            if ss == self.SS5:
                # SS5 - don't run by default
                self._log("INFO", "Interaction: SS5 face - skipping interaction")
                return
            
            state_map = {
                self.SS1: "ss1",
                self.SS2: "ss2",
                self.SS3: "ss3",
                self.SS4: "ss4"
            }
            start_state = state_map.get(ss, "ss1")
            self._log("INFO", f"Interaction: Starting from state '{start_state}'")
            
            # Execute ao_start first
            self._log("INFO", "Interaction: Executing ao_start behaviour")
            self._execute_interaction_interface("ao_start")
            
            try:
                # Run interaction manager
                self._log("INFO", f"Interaction: Calling interactionManager RPC")
                result = self._run_interaction_manager(track_id, face_id, start_state)
                
                if result:
                    self._log("INFO", f"Interaction: Received result (success={result.get('success')})")
                    # Process result and update states
                    self._process_interaction_result(result, target)
                else:
                    self._log("WARNING", "Interaction: No result from interactionManager")
                    
            finally:
                # Always execute ao_stop
                self._log("INFO", "Interaction: Executing ao_stop behaviour")
                self._execute_interaction_interface("ao_stop")
        
        except Exception as e:
            self._log("ERROR", f"Interaction thread error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            with self.state_lock:
                self.interaction_busy = False
                self.selected_target = None
                self.selected_bbox_last = None  # Clear stored bbox
            self._log("INFO", "=== INTERACTION COMPLETE ===")

    def _execute_interaction_interface(self, command: str) -> bool:
        """Execute a command on /interactionInterface via RPC."""
        try:
            if self.interaction_interface_rpc.getOutputCount() == 0:
                self._log("WARNING", f"RPC: interactionInterface not connected")
                return False
            
            cmd = yarp.Bottle()
            cmd.addString("exe")
            cmd.addString(command)
            
            reply = yarp.Bottle()
            
            self._log("DEBUG", f"RPC → interactionInterface: exe {command}")
            
            if self.interaction_interface_rpc.write(cmd, reply):
                self._log("DEBUG", f"RPC ← interactionInterface: {reply.toString()}")
                return True
            else:
                self._log("WARNING", f"RPC: Write failed for command '{command}'")
                return False
                
        except Exception as e:
            self._log("ERROR", f"RPC: interactionInterface exception: {e}")
            return False

    def _run_interaction_manager(self, track_id: int, face_id: str, start_state: str) -> Optional[Dict]:
        """Run interaction manager RPC and parse result."""
        try:
            if self.interaction_manager_rpc.getOutputCount() == 0:
                self._log("WARNING", "RPC: interactionManager not connected")
                return None
            
            # Build command: ["run", track_id, face_id, start_state]
            cmd = yarp.Bottle()
            cmd.addString("run")
            cmd.addInt32(track_id)
            cmd.addString(face_id)
            cmd.addString(start_state)
            
            reply = yarp.Bottle()
            
            self._log("DEBUG", f"RPC → interactionManager: {cmd.toString()}")
            
            # This call blocks until interaction completes
            if self.interaction_manager_rpc.write(cmd, reply):
                self._log("DEBUG", f"RPC ← interactionManager: reply size={reply.size()}")
                
                # Parse reply: ["ok", "json_string"]
                if reply.size() >= 2:
                    status = reply.get(0).asString()
                    json_str = reply.get(1).asString()
                    
                    if status == "ok":
                        try:
                            result = json.loads(json_str)
                            success = result.get('success', False)
                            final_state = result.get('final_state', '?')
                            steps_count = len(result.get('steps', []))
                            self._log("INFO", f"RPC: Parsed result - success={success}, final={final_state}, steps={steps_count}")
                            return result
                        except json.JSONDecodeError as e:
                            self._log("ERROR", f"RPC: JSON parse failed: {e}")
                            return None
                    else:
                        self._log("WARNING", f"RPC: Non-ok status: {status}")
                        return None
                else:
                    self._log("WARNING", f"RPC: Unexpected reply format: {reply.toString()}")
                    return None
            else:
                self._log("ERROR", "RPC: Write to interactionManager failed")
                return None
                
        except Exception as e:
            self._log("ERROR", f"RPC: interactionManager exception: {e}")
            return None

    def _process_interaction_result(self, result: Dict, target: Dict):
        """Process interaction result: update memory files and learning state."""
        try:
            success = result.get("success", False)
            final_state = result.get("final_state", "")
            steps = result.get("steps", [])
            
            self._log("INFO", f"Processing result: success={success}, final_state={final_state}, steps={len(steps)}")
            
            # Resolve person_id
            person_id = self._resolve_person_id(result, target)
            self._log("INFO", f"Result: Resolved person_id = '{person_id}'")
            
            track_id = target["track_id"]
            now_iso = datetime.now(self.TIMEZONE).isoformat()
            
            # Determine what to update
            greeted = False
            talked = False
            
            for step in steps:
                step_name = step.get("step", "")
                details = step.get("details", {})
                
                if step_name in ("ss1", "ss3"):
                    # Check if greeting was attempted
                    if step.get("status") == "success" or details.get("greet_attempt") == "successful":
                        greeted = True
                        self._log("INFO", f"Result: {step_name} completed - marking as greeted")
                
                if step_name == "ss4":
                    turns = details.get("turns_count", 0)
                    if turns >= 1:
                        talked = True
                        self._log("INFO", f"Result: ss4 had {turns} turns - marking as talked")
            
            # Thread-safe: update shared state under lock
            with self.state_lock:
                self.track_to_person[track_id] = person_id
                if greeted:
                    self.greeted_today[person_id] = now_iso
                if talked:
                    self.talked_today[person_id] = now_iso
            
            # Save files outside lock (I/O can be slow)
            if greeted:
                self._save_greeted_json()
                self._log("INFO", f"Result: Saved greeted_today for '{person_id}'")
            
            if talked:
                self._save_talked_json()
                self._log("INFO", f"Result: Saved talked_today for '{person_id}'")
            
            # Compute interaction score and update learning state
            delta = self._compute_interaction_score(result)
            self._log("INFO", f"Result: Interaction score delta = {delta:+d}")
            
            # Update learning state
            self._update_learning_state(person_id, delta)
            
        except Exception as e:
            self._log("ERROR", f"Failed to process interaction result: {e}")
            import traceback
            traceback.print_exc()

    def _resolve_person_id(self, result: Dict, target: Dict) -> str:
        """Resolve the best person_id from interaction result."""
        # Priority 1: SS2 extracted_name with high confidence
        steps = result.get("steps", [])
        for step in steps:
            if step.get("step") == "ss2":
                details = step.get("details", {})
                name = details.get("extracted_name")
                confidence = details.get("confidence", 0)
                if name and confidence >= 0.7:
                    return name
        
        # Priority 2: SS1 face_registered_as (5-digit code)
        for step in steps:
            if step.get("step") == "ss1":
                details = step.get("details", {})
                code = details.get("face_registered_as")
                if code:
                    return code
        
        # Priority 3: initial_face_id from result
        initial_face_id = result.get("initial_face_id")
        if initial_face_id and initial_face_id.lower() != "unknown":
            return initial_face_id
        
        # Priority 4: Face ID from target
        return target.get("face_id", "unknown")

    def _compute_interaction_score(self, result: Dict) -> int:
        """Compute total delta score from interaction result."""
        delta = 0
        success = result.get("success", False)
        steps = result.get("steps", [])
        
        for step in steps:
            step_name = step.get("step", "")
            details = step.get("details", {})
            
            if step_name == "ss1":
                # SS1 scoring
                response_detected = details.get("response_detected", False)
                greet_attempt = details.get("greet_attempt", "")
                
                ss1_delta = 0
                if response_detected:
                    ss1_delta = 2
                elif greet_attempt == "successful":
                    ss1_delta = 1
                else:
                    ss1_delta = -2
                delta += ss1_delta
                
                self._log("DEBUG", f"SS1: response={response_detected}, greet={greet_attempt}, delta_contrib={ss1_delta}")
            
            elif step_name == "ss2":
                # SS2 scoring
                extracted_name = details.get("extracted_name")
                confidence = details.get("confidence", 0)
                attempts_made = details.get("attempts_made", 0)
                
                ss2_delta = 0
                if extracted_name and confidence >= 0.7:
                    ss2_delta = 3
                elif extracted_name:
                    ss2_delta = 1
                elif attempts_made >= 2:
                    ss2_delta = -2
                else:
                    ss2_delta = -3
                delta += ss2_delta
                
                self._log("DEBUG", f"SS2: name={extracted_name}, conf={confidence}, attempts={attempts_made}, delta_contrib={ss2_delta}")
            
            elif step_name == "ss3":
                # SS3 scoring
                response_detected = details.get("response_detected", False)
                
                ss3_delta = 2 if response_detected else -2
                delta += ss3_delta
                
                self._log("DEBUG", f"SS3: response={response_detected}, delta_contrib={ss3_delta}")
            
            elif step_name == "ss4":
                # SS4 scoring
                turns_count = details.get("turns_count", 0)
                
                ss4_delta = 0
                if turns_count >= 5:
                    ss4_delta = 4
                elif turns_count >= 3:
                    ss4_delta = 2
                elif turns_count >= 1:
                    ss4_delta = 0  # Neutral
                else:
                    ss4_delta = -4
                delta += ss4_delta
                
                self._log("DEBUG", f"SS4: turns={turns_count}, delta_contrib={ss4_delta}")
        
        # Global failure penalty
        if not success:
            delta -= 3
            self._log("DEBUG", "Global failure penalty: -3")
        
        return delta

    def _update_learning_state(self, person_id: str, delta: int):
        """Update learning state based on interaction score delta."""
        # Get current LS (read under lock)
        with self.state_lock:
            current_ls = self.learning_data.get(person_id, {}).get("ls", self.LS1)
        
        old_ls_name = self.LS_NAMES.get(current_ls, "?")
        self._log("INFO", f"LS Update: person='{person_id}', current={old_ls_name}, delta={delta:+d}")
        
        new_ls = current_ls
        
        # Apply delta thresholds
        if delta >= 4:
            new_ls = min(4, current_ls + 1)
            self._log("DEBUG", f"LS Update: delta >= 4, attempting increase {current_ls} -> {new_ls}")
        elif delta <= -4:
            new_ls = max(1, current_ls - 1)
            self._log("DEBUG", f"LS Update: delta <= -4, attempting decrease {current_ls} -> {new_ls}")
        else:
            self._log("DEBUG", f"LS Update: |delta| < 4, no state change")
        # else: no change
        
        # Update under lock, save outside lock
        now_iso = datetime.now(self.TIMEZONE).isoformat()
        with self.state_lock:
            self.learning_data[person_id] = {
                "ls": new_ls,
                "updated_at": now_iso
            }
        
        self._save_learning_json()
        
        if new_ls != current_ls:
            new_ls_name = self.LS_NAMES.get(new_ls, "?")
            self._log("INFO", f"LS Update: '{person_id}' → {old_ls_name} to {new_ls_name}")
        else:
            self._log("INFO", f"Learning state unchanged: {person_id} LS{current_ls} (delta={delta})")

    # ==================== JSON File Management ====================

    def _load_all_json_files(self):
        """Load all persistent JSON files."""
        self.greeted_today = self._load_json(self.greeted_path, {})
        self.talked_today = self._load_json(self.talked_path, {})
        
        # Learning data has nested structure
        learning_raw = self._load_json(self.learning_path, {"people": {}})
        self.learning_data = learning_raw.get("people", {})
        
        # Prune old entries (keep only today's data)
        self.greeted_today = self._prune_to_today(self.greeted_today)
        self.talked_today = self._prune_to_today(self.talked_today)
        
        self._log("INFO", f"Loaded {len(self.greeted_today)} greeted entries (today)")
        self._log("INFO", f"Loaded {len(self.talked_today)} talked entries (today)")
        self._log("INFO", f"Loaded {len(self.learning_data)} learning entries")

    def _save_all_json_files(self):
        """Save all persistent JSON files."""
        self._save_greeted_json()
        self._save_talked_json()
        self._save_learning_json()

    def _load_json(self, path: Path, default: Any) -> Any:
        """Load JSON file or return default."""
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self._log("WARNING", f"Failed to load {path}: {e}")
        return default

    def _save_json_atomic(self, path: Path, data: Any):
        """Save JSON file atomically (write to temp then rename)."""
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temp file first
            fd, temp_path = tempfile.mkstemp(suffix='.json', dir=path.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                # Atomic rename
                os.replace(temp_path, path)
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        except Exception as e:
            self._log("ERROR", f"Failed to save {path}: {e}")

    def _save_greeted_json(self):
        """Save greeted_today.json."""
        self._save_json_atomic(self.greeted_path, self.greeted_today)

    def _save_talked_json(self):
        """Save talked_today.json."""
        self._save_json_atomic(self.talked_path, self.talked_today)

    def _save_learning_json(self):
        """Save learning.json."""
        data = {"people": self.learning_data}
        self._save_json_atomic(self.learning_path, data)

    def _prune_to_today(self, d: Dict[str, str]) -> Dict[str, str]:
        """Remove entries not from today (for daily cleanup)."""
        today = self._get_today_date()
        out = {}
        for k, ts in d.items():
            try:
                dt = datetime.fromisoformat(ts).astimezone(self.TIMEZONE)
                if dt.date() == today:
                    out[k] = ts
            except Exception:
                pass
        return out

    # ==================== Debug Output ====================

    def _publish_debug(self):
        """Publish debug information."""
        if self.debug_port.getOutputCount() == 0:
            return
        
        try:
            btl = yarp.Bottle()
            btl.clear()
            
            # Status
            btl.addString("status")
            btl.addString("busy" if self.interaction_busy else "idle")
            
            # Face count
            btl.addString("face_count")
            btl.addInt32(len(self.current_faces))
            
            # Selected target
            with self.state_lock:
                if self.selected_target:
                    btl.addString("selected_face_id")
                    btl.addString(self.selected_target.get("face_id", "?"))
                    btl.addString("selected_track_id")
                    btl.addInt32(self.selected_target.get("track_id", -1))
            
            self.debug_port.write(btl)
            
        except Exception as e:
            self._log("WARNING", f"Failed to publish debug: {e}")

    # ==================== Utilities ====================

    def _get_today_date(self) -> date:
        """Get today's date in configured timezone."""
        return datetime.now(self.TIMEZONE).date()

    def _log(self, level: str, message: str):
        """Simple logging with timestamp."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] [{level}] {message}")


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    # Initialize YARP
    yarp.Network.init()
    
    if not yarp.Network.checkNetwork():
        print("ERROR: YARP network not available. Start yarpserver first.")
        sys.exit(1)
    
    # Create module
    module = FaceSelectorModule()
    
    # Configure from command line
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("alwaysOn")
    rf.configure(sys.argv)
    
    # Print usage
    print("=" * 60)
    print("FaceSelectorModule - Real-Time Face Selection & Interaction")
    print("=" * 60)
    print()
    print("YARP Connections (run after starting):")
    print("  yarp connect /alwayson/vision/landmarks:o /faceSelector/landmarks:i")
    print("  yarp connect /alwayson/vision/img:o /faceSelector/img:i")
    print("  yarp connect /faceSelector/interactionManager:rpc /interactionManager")
    print("  yarp connect /faceSelector/interactionInterface:rpc /interactionInterface")
    print()
    print("Optional output connections:")
    print("  yarp connect /faceSelector/img:o /viewer")
    print("  yarp connect /faceSelector/debug:o /debugViewer")
    print()
    print("Configuration parameters:")
    print("  --name <module_name>          (default: faceSelector)")
    print("  --landmarks_in <port>         (default: /alwayson/vision/landmarks:o)")
    print("  --img_in <port>               (default: /alwayson/vision/img:o)")
    print("  --interaction_manager_rpc     (default: /interactionManager)")
    print("  --interaction_interface_rpc   (default: /interactionInterface)")
    print("  --learning_path <file>        (default: ./learning.json)")
    print("  --greeted_path <file>         (default: ./greeted_today.json)")
    print("  --talked_path <file>          (default: ./talked_today.json)")
    print("  --rate <seconds>              (default: 0.05)")
    print("  --allow_ss5 <true/false>      (default: false)")
    print()
    
    try:
        # Let runModule() handle configuration (avoids double-configure issues)
        print("Starting module...")
        if not module.runModule(rf):
            print("ERROR: Module failed to run.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        module.interruptModule()
        module.close()
        yarp.Network.fini()
        print("Module closed.")
