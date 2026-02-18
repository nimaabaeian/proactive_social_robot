"""
faceSelector.py - YARP RFModule for Real-Time Face Selection & Interaction Trigger

Selects target face by biggest bounding-box area, manages ao_start/ao_stop
transitions based on face presence, computes social/learning states, and
triggers interactions via /interactionManager RPC.

YARP Connections (run after starting):
    yarp connect /alwayson/vision/landmarks:o /faceSelector/landmarks:i

Social States (SS):
    ss1: unknown
    ss2: known, not greeted today
    ss3: known, greeted today, not talked to
    ss4: known, greeted today, talked to  (terminal)

"Greeted today" / "talked today" only apply to KNOWN people.
"""

import concurrent.futures
import json
import os
import queue
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yarp
except ImportError:
    print("ERROR: YARP Python bindings are required.")
    sys.exit(1)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo


# ═══════════════════════ Constants ═══════════════════════

TIMEZONE = ZoneInfo("Europe/Rome")

# Social states
SS1 = "ss1"  # unknown
SS2 = "ss2"  # known, not greeted today
SS3 = "ss3"  # known, greeted today, not talked to
SS4 = "ss4"  # known, greeted today, talked to (terminal)

SS_DESCRIPTIONS = {
    SS1: "unknown",
    SS2: "known, not greeted",
    SS3: "known, greeted, not talked",
    SS4: "known, greeted, talked (terminal)",
}

# Learning states
LS1, LS2, LS3, LS4 = 1, 2, 3, 4
LS_NAMES = {1: "LS1", 2: "LS2", 3: "LS3", 4: "LS4"}

# Valid zones/distances/attentions per LS
LS_VALID_ZONES = {
    1: {"FAR_LEFT", "LEFT", "CENTER", "RIGHT", "FAR_RIGHT", "UNKNOWN"},
    2: {"LEFT", "CENTER", "RIGHT"},
    3: {"LEFT", "CENTER", "RIGHT"},
    4: {"LEFT", "CENTER", "RIGHT"},
}
LS_VALID_DISTANCES = {
    1: {"SO_CLOSE", "CLOSE", "FAR", "VERY_FAR", "UNKNOWN"},
    2: {"SO_CLOSE", "CLOSE", "FAR"},
    3: {"SO_CLOSE", "CLOSE"},
    4: {"SO_CLOSE", "CLOSE"},
}
LS_VALID_ATTENTIONS = {
    1: {"MUTUAL_GAZE", "NEAR_GAZE", "AWAY", "UNKNOWN"},
    2: {"MUTUAL_GAZE", "NEAR_GAZE", "AWAY", "UNKNOWN"},
    3: {"MUTUAL_GAZE", "NEAR_GAZE", "UNKNOWN"},
    4: {"MUTUAL_GAZE", "UNKNOWN"},
}

# File paths (defaults, overridable via ResourceFinder)
DEFAULT_LEARNING_PATH = Path("./learning.json")
DEFAULT_GREETED_PATH = Path("./greeted_today.json")
DEFAULT_TALKED_PATH = Path("./talked_today.json")
DEFAULT_LAST_GREETED_PATH = Path("./last_greeted.json")
DEFAULT_DB_PATH = "faceSelector.db"

# Timing
DEFAULT_PERIOD = 0.05  # 20 Hz
INTERACTION_COOLDOWN = 5.0  # seconds
RPC_TIMEOUT = 10.0  # seconds for RPC calls

# Anti-thrash: require same biggest track_id for N reads before switching target
BIGGEST_STABILITY_COUNT = 3


# ═══════════════════════ Dataclasses ═══════════════════════

@dataclass
class FaceObservation:
    """Single face observation parsed from landmarks."""
    track_id: int = -1
    face_id: str = "unknown"
    bbox_x: float = 0.0
    bbox_y: float = 0.0
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    area: float = 0.0
    zone: str = "UNKNOWN"
    distance: str = "UNKNOWN"
    attention: str = "AWAY"
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    cos_angle: float = 0.0
    is_talking: int = 0
    time_in_view: float = 0.0
    timestamp: float = field(default_factory=time.time)

    # Computed fields (set after construction)
    person_id: str = ""
    is_known: bool = False
    social_state: str = SS1
    learning_state: int = 1
    eligible: bool = False
    greeted_today: bool = False
    talked_today: bool = False


@dataclass
class TargetState:
    """Tracks current module state for target selection and AO."""
    current_target_track_id: Optional[int] = None
    ao_running: bool = False
    interaction_busy: bool = False
    last_biggest_track_id: Optional[int] = None
    biggest_stable_count: int = 0


# ═══════════════════════ JSON Utilities ═══════════════════════

def load_json_safe(path: Path, default: Any) -> Any:
    """Load JSON file, return default on any error."""
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json_atomic(path: Path, data: Any) -> bool:
    """Save JSON atomically via temp-file + rename. Returns success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".json", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
            return True
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    except Exception:
        return False


def prune_to_today(d: Dict[str, str], tz: ZoneInfo) -> Dict[str, str]:
    """Keep only entries whose ISO timestamp falls on today in *tz*."""
    today = datetime.now(tz).date()
    out = {}
    for k, ts in d.items():
        try:
            dt = datetime.fromisoformat(ts).astimezone(tz)
            if dt.date() == today:
                out[k] = ts
        except Exception:
            pass
    return out


# ═══════════════════════ RPC Utility ═══════════════════════

def rpc_call(port: yarp.RpcClient, cmd: yarp.Bottle,
             timeout: float = RPC_TIMEOUT) -> Optional[yarp.Bottle]:
    """Send RPC with enforced timeout via a worker thread.

    YARP's port.write() can block indefinitely if the remote end is slow.
    We run it in a ThreadPoolExecutor future so we can abort cleanly.
    Retries once on transient failure.
    """
    if port.getOutputCount() == 0:
        return None

    def _do_write() -> Optional[yarp.Bottle]:
        reply = yarp.Bottle()
        if port.write(cmd, reply):
            return reply
        return None

    for attempt in range(2):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_do_write)
                result = future.result(timeout=timeout)
                if result is not None:
                    return result
        except concurrent.futures.TimeoutError:
            port.interrupt()  # unblock the hanging write
            return None
        except Exception:
            if attempt == 0:
                time.sleep(0.2)
    return None


def rpc_fire_and_forget(port: yarp.RpcClient, cmd: yarp.Bottle) -> bool:
    """Send RPC command without waiting for reply."""
    if port.getOutputCount() == 0:
        return False
    try:
        port.write(cmd)
        return True
    except Exception:
        return False


# ═══════════════════════ DB Writer Thread ═══════════════════════

class AsyncDBWriter:
    """Async SQLite writer: queue + single writer thread, WAL mode."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="db-writer")
        self._init_db()
        self._thread.start()

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS target_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, track_id INTEGER, face_id TEXT,
                bbox_area REAL, zone TEXT, distance TEXT, attention TEXT
            );
            CREATE TABLE IF NOT EXISTS ss_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, person TEXT, old_ss TEXT, new_ss TEXT
            );
            CREATE TABLE IF NOT EXISTS ls_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, person TEXT, old_ls INTEGER, new_ls INTEGER,
                reward_delta INTEGER, reason TEXT
            );
            CREATE TABLE IF NOT EXISTS ao_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, action TEXT
            );
        """)
        conn.commit()
        conn.close()

    def log(self, table: str, data: Dict[str, Any]):
        try:
            self._queue.put_nowait((table, data))
        except queue.Full:
            pass  # drop on overflow

    def _run(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        while not self._stop.is_set():
            try:
                table, data = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                cols = ", ".join(data.keys())
                placeholders = ", ".join("?" for _ in data)
                conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                             list(data.values()))
                conn.commit()
            except Exception:
                pass
        conn.close()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)


# ═══════════════════════ Module ═══════════════════════

class FaceSelectorModule(yarp.RFModule):
    """
    Real-time face selection module:
    - Reads face landmarks from vision system
    - Selects biggest-bbox face as target
    - Manages ao_start/ao_stop based on face presence (edge-triggered)
    - Computes social states (ss1-ss4) and learning states
    - Triggers interactions via /interactionManager RPC when LS permits
    - Logs to faceSelector.db asynchronously
    """

    def __init__(self):
        super().__init__()

        # Module config
        self.module_name = "faceSelector"
        self.period = DEFAULT_PERIOD
        self._running = True
        self._consecutive_errors = 0
        self._max_consecutive_errors = 10

        # RPC target names
        self.interaction_manager_rpc_name = "/interactionManager"
        self.interaction_interface_rpc_name = "/interactionInterface"

        # File paths
        self.learning_path = DEFAULT_LEARNING_PATH
        self.greeted_path = DEFAULT_GREETED_PATH
        self.talked_path = DEFAULT_TALKED_PATH
        self.last_greeted_path = DEFAULT_LAST_GREETED_PATH
        self.db_path = DEFAULT_DB_PATH

        # YARP ports
        self.landmarks_port: Optional[yarp.BufferedPortBottle] = None
        self.debug_port: Optional[yarp.Port] = None
        self.interaction_manager_rpc: Optional[yarp.RpcClient] = None
        self.interaction_interface_rpc: Optional[yarp.RpcClient] = None

        # State (thread-safe)
        self.state_lock = threading.Lock()
        self.current_faces: List[FaceObservation] = []
        self.target_state = TargetState()
        self.interaction_thread: Optional[threading.Thread] = None

        # AO worker: single-thread pool for non-blocking AO commands
        self._ao_executor: Optional[ThreadPoolExecutor] = None

        # Interaction metrics (in-memory, reset on restart)
        self.metrics_attempted: int = 0
        self.metrics_aborted_target_lost: int = 0
        self.metrics_aborted_no_response: int = 0

        # Cooldown
        self.last_interaction_time: Dict[str, float] = {}

        # Memory caches
        self.greeted_today: Dict[str, str] = {}
        self.talked_today: Dict[str, str] = {}
        self.learning_data: Dict[str, Dict] = {}
        self.last_greeted: List[Dict] = []

        # last_greeted.json sync throttle (sync every 2 s during runtime)
        self._last_greeted_sync_time: float = 0.0
        self._last_greeted_sync_interval: float = 2.0

        # Track-to-person mapping
        self.track_to_person: Dict[int, str] = {}

        # Day tracking
        self._current_day: Optional[date] = None

        # Config flags
        self.verbose_debug = False
        self.ports_connected_logged = False

        # DB writer (created in configure)
        self._db: Optional[AsyncDBWriter] = None

    # ──────────────── Lifecycle ────────────────

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        try:
            if rf.check("name"):
                self.module_name = rf.find("name").asString()
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
            if rf.check("last_greeted_path"):
                self.last_greeted_path = Path(rf.find("last_greeted_path").asString())
            if rf.check("rate"):
                self.period = rf.find("rate").asFloat64()
            if rf.check("verbose"):
                self.verbose_debug = rf.find("verbose").asBool()

            # Open landmarks input
            self.landmarks_port = yarp.BufferedPortBottle()
            if not self.landmarks_port.open(f"/{self.module_name}/landmarks:i"):
                self._log("ERROR", "Failed to open landmarks input port")
                return False

            # Open debug output
            self.debug_port = yarp.Port()
            if not self.debug_port.open(f"/{self.module_name}/debug:o"):
                self._log("ERROR", "Failed to open debug output port")
                return False

            # Open RPC clients
            self.interaction_manager_rpc = yarp.RpcClient()
            if not self.interaction_manager_rpc.open(f"/{self.module_name}/interactionManager:rpc"):
                self._log("ERROR", "Failed to open interactionManager RPC port")
                return False

            self.interaction_interface_rpc = yarp.RpcClient()
            if not self.interaction_interface_rpc.open(f"/{self.module_name}/interactionInterface:rpc"):
                self._log("ERROR", "Failed to open interactionInterface RPC port")
                return False

            # Auto-connect RPC ports
            self._log("INFO", "Connecting RPC ports...")
            for src, dst in [
                (f"/{self.module_name}/interactionManager:rpc", self.interaction_manager_rpc_name),
                (f"/{self.module_name}/interactionInterface:rpc", self.interaction_interface_rpc_name),
            ]:
                if yarp.Network.connect(src, dst):
                    self._log("INFO", f"Connected {src} → {dst}")
                else:
                    self._log("ERROR", f"Failed to connect {src} → {dst}")

            # AO worker pool
            self._ao_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ao")

            # Load persistent data
            self._load_all_data()
            self._current_day = self._today()

            # DB writer
            self._db = AsyncDBWriter(self.db_path)

            self._log("INFO", f"FaceSelectorModule configured ({1.0/self.period:.0f} Hz)")
            return True

        except Exception as e:
            self._log("ERROR", f"Configuration failed: {e}")
            import traceback; traceback.print_exc()
            return False

    def interruptModule(self) -> bool:
        self._log("INFO", "Interrupting module...")
        self._running = False
        for port in [self.landmarks_port, self.debug_port,
                     self.interaction_manager_rpc, self.interaction_interface_rpc]:
            if port:
                port.interrupt()
        return True

    def close(self) -> bool:
        self._log("INFO", "Closing module...")

        # Shutdown AO executor
        if self._ao_executor:
            self._ao_executor.shutdown(wait=False)

        # Wait for interaction thread
        if self.interaction_thread and self.interaction_thread.is_alive():
            self._log("INFO", "Waiting for interaction thread...")
            self.interaction_thread.join(timeout=5.0)

        # Log final metrics
        self._log(
            "INFO",
            f"Metrics: attempted={self.metrics_attempted}, "
            f"aborted_target_lost={self.metrics_aborted_target_lost}, "
            f"aborted_no_response={self.metrics_aborted_no_response}",
        )

        # Save state
        self._save_all_data()

        # Stop DB writer
        if self._db:
            self._db.stop()

        # Close ports
        for port in [self.landmarks_port, self.debug_port,
                     self.interaction_manager_rpc, self.interaction_interface_rpc]:
            if port:
                port.close()

        return True

    def getPeriod(self) -> float:
        return self.period

    # ──────────────── Main Loop ────────────────

    def updateModule(self) -> bool:
        if not self._running:
            return False

        try:
            # Wait for landmarks port
            if self.landmarks_port.getInputCount() == 0:
                if not self.ports_connected_logged:
                    self._log("INFO", "Waiting for landmarks port...")
                    self._log("INFO", "  yarp connect /alwayson/vision/landmarks:o /faceSelector/landmarks:i")
                    self.ports_connected_logged = True
                return True

            if self.ports_connected_logged:
                self._log("INFO", "✓ Landmarks port connected")
                self.ports_connected_logged = False

            # Day change check
            today = self._today()
            if self._current_day != today:
                self._log("INFO", f"=== DAY CHANGE: {self._current_day} → {today} ===")
                with self.state_lock:
                    self.greeted_today = prune_to_today(self.greeted_today, TIMEZONE)
                    self.talked_today = prune_to_today(self.talked_today, TIMEZONE)
                save_json_atomic(self.greeted_path, self.greeted_today)
                save_json_atomic(self.talked_path, self.talked_today)
                self._current_day = today

            # 1. Read landmarks
            raw_faces = self._read_landmarks()

            # 2. Throttled last_greeted sync (picks up new entries from interactionManager)
            now = time.time()
            if now - self._last_greeted_sync_time >= self._last_greeted_sync_interval:
                self._sync_last_greeted()
                self._last_greeted_sync_time = now

            # 3. Compute states
            with self.state_lock:
                self.current_faces = self._compute_face_states(raw_faces)
                faces = list(self.current_faces)

            # 3. AO start/stop (edge-triggered on face presence)
            has_faces = len(faces) > 0
            with self.state_lock:
                was_running = self.target_state.ao_running
                if has_faces and not was_running:
                    self.target_state.ao_running = True
                    self._ao_executor.submit(self._ao_command, "ao_start")
                    self._db.log("ao_transitions", {"ts": self._now_iso(), "action": "ao_start"})
                    self._log("INFO", "AO: start (face appeared)")
                elif not has_faces and was_running:
                    self.target_state.ao_running = False
                    self._ao_executor.submit(self._ao_command, "ao_stop")
                    self._db.log("ao_transitions", {"ts": self._now_iso(), "action": "ao_stop"})
                    self._log("INFO", "AO: stop (no faces)")

            # 4. Target selection + interaction trigger
            self._do_selection_and_trigger(faces)

            # 5. Debug output
            self._publish_debug(faces)

            self._consecutive_errors = 0
            return True

        except Exception as e:
            self._consecutive_errors += 1
            self._log("ERROR", f"updateModule error: {e}")
            import traceback; traceback.print_exc()
            if self._consecutive_errors >= self._max_consecutive_errors:
                self._log("CRITICAL", "Too many errors, stopping")
                return False
            return True

    # ──────────────── AO Commands ────────────────

    def _ao_command(self, command: str):
        """Send ao_start / ao_stop to interactionInterface (runs in AO worker thread)."""
        cmd = yarp.Bottle()
        cmd.addString("exe")
        cmd.addString(command)
        rpc_fire_and_forget(self.interaction_interface_rpc, cmd)

    # ──────────────── Landmark Parsing ────────────────

    def _read_landmarks(self) -> List[FaceObservation]:
        """Read and parse latest landmarks (non-blocking)."""
        bottle = self.landmarks_port.read(False)
        if not bottle:
            return []

        faces = []
        for i in range(bottle.size()):
            item = bottle.get(i)
            if not item.isList():
                continue
            obs = self._parse_face_bottle(item.asList())
            if obs:
                faces.append(obs)
        return faces

    def _parse_face_bottle(self, bottle: yarp.Bottle) -> Optional[FaceObservation]:
        """Parse a single face bottle into FaceObservation."""
        if not bottle:
            return None

        obs = FaceObservation()
        try:
            i = 0
            while i < bottle.size():
                item = bottle.get(i)

                if item.isString() and i + 1 < bottle.size():
                    key = item.asString()
                    nxt = bottle.get(i + 1)

                    if key == "face_id" and nxt.isString():
                        obs.face_id = nxt.asString(); i += 2
                    elif key == "track_id" and (nxt.isInt32() or nxt.isInt64()):
                        obs.track_id = nxt.asInt32(); i += 2
                    elif key == "zone" and nxt.isString():
                        obs.zone = nxt.asString(); i += 2
                    elif key == "distance" and nxt.isString():
                        obs.distance = nxt.asString(); i += 2
                    elif key == "attention" and nxt.isString():
                        obs.attention = nxt.asString(); i += 2
                    elif key == "pitch" and nxt.isFloat64():
                        obs.pitch = nxt.asFloat64(); i += 2
                    elif key == "yaw" and nxt.isFloat64():
                        obs.yaw = nxt.asFloat64(); i += 2
                    elif key == "roll" and nxt.isFloat64():
                        obs.roll = nxt.asFloat64(); i += 2
                    elif key == "cos_angle" and nxt.isFloat64():
                        obs.cos_angle = nxt.asFloat64(); i += 2
                    elif key == "is_talking" and (nxt.isInt32() or nxt.isInt64()):
                        obs.is_talking = nxt.asInt32(); i += 2
                    elif key == "time_in_view" and nxt.isFloat64():
                        obs.time_in_view = nxt.asFloat64(); i += 2
                    else:
                        i += 1

                elif item.isList():
                    nested = item.asList()
                    if nested.size() >= 2:
                        key = nested.get(0).asString() if nested.get(0).isString() else ""
                        if key == "bbox" and nested.size() >= 5:
                            obs.bbox_x = nested.get(1).asFloat64()
                            obs.bbox_y = nested.get(2).asFloat64()
                            obs.bbox_w = nested.get(3).asFloat64()
                            obs.bbox_h = nested.get(4).asFloat64()
                            obs.area = obs.bbox_w * obs.bbox_h
                    i += 1
                else:
                    i += 1

            obs.timestamp = time.time()
            return obs

        except Exception as e:
            self._log("WARNING", f"Parse face bottle error: {e}")
            return None

    # ──────────────── State Computation ────────────────

    def _compute_face_states(self, faces: List[FaceObservation]) -> List[FaceObservation]:
        """Compute SS, LS, eligibility for all faces. Must hold state_lock."""
        today = self._today()

        for obs in faces:
            # Resolve person_id
            person_id = self.track_to_person.get(obs.track_id, obs.face_id)
            obs.person_id = person_id

            # Known check
            obs.is_known = self._is_known(obs.face_id) or self._is_known(person_id)

            # Social state
            obs.greeted_today = self._was_greeted_today(person_id, today) if obs.is_known else False
            obs.talked_today = self._was_talked_today(person_id, today) if obs.is_known else False
            obs.social_state = self._compute_social_state(obs)

            # Learning state
            obs.learning_state = self._get_ls(person_id)

            # Eligibility (LS gate)
            obs.eligible = self._is_eligible(obs)

        # Prune stale track mappings
        active = {f.track_id for f in faces if f.track_id >= 0}
        self.track_to_person = {t: p for t, p in self.track_to_person.items() if t in active}

        return faces

    @staticmethod
    def _is_known(face_id: str) -> bool:
        """Known = resolved to a name (not unknown/unmatched/recognizing)."""
        if not face_id:
            return False
        low = face_id.lower()
        if low in ("unknown", "unmatched", "recognizing",):
            return False
        return True

    @staticmethod
    def _is_face_resolved(face_id: str) -> bool:
        """Resolved = vision system has finished recognising."""
        return face_id.lower() not in ("recognizing", "unmatched")

    def _was_greeted_today(self, person_id: str, today: date) -> bool:
        ts = self.greeted_today.get(person_id)
        if not ts:
            return False
        try:
            return datetime.fromisoformat(ts).astimezone(TIMEZONE).date() == today
        except Exception:
            return False

    def _was_talked_today(self, person_id: str, today: date) -> bool:
        ts = self.talked_today.get(person_id)
        if not ts:
            return False
        try:
            return datetime.fromisoformat(ts).astimezone(TIMEZONE).date() == today
        except Exception:
            return False

    @staticmethod
    def _compute_social_state(obs: FaceObservation) -> str:
        """
        ss1: unknown (face_id unresolved / not a known name)
        ss2: known, not greeted today
        ss3: known, greeted today, not talked to
        ss4: known, greeted today, talked to  (terminal)
        """
        if not obs.is_known:
            return SS1
        if not obs.greeted_today:
            return SS2
        if not obs.talked_today:
            return SS3
        return SS4

    def _get_ls(self, person_id: str) -> int:
        return self.learning_data.get(person_id, {}).get("ls", LS1)

    def _is_eligible(self, obs: FaceObservation) -> bool:
        """Check LS spatial gate. SS4 (terminal) faces are never eligible."""
        if obs.social_state == SS4:
            return False
        ls = obs.learning_state
        if obs.zone not in LS_VALID_ZONES.get(ls, set()):
            return False
        if obs.distance not in LS_VALID_DISTANCES.get(ls, set()):
            return False
        if obs.attention not in LS_VALID_ATTENTIONS.get(ls, set()):
            return False
        return True

    # ──────────────── Selection & Trigger ────────────────

    def _do_selection_and_trigger(self, faces: List[FaceObservation]):
        """
        Target = biggest bbox, always (anti-thrash stabilised).

        Two separate concerns:
        1. TARGET SELECTION – the biggest-bbox face is ALWAYS the target.
           This sets `current_target_track_id` unconditionally once stable.
        2. INTERACTION TRIGGER – only fires when the LS gate (zone, distance,
           attention) permits AND cooldown/busy checks pass.
        """
        current_time = time.time()

        # ── No faces ──
        if not faces:
            with self.state_lock:
                self.target_state.last_biggest_track_id = None
                self.target_state.biggest_stable_count = 0
                self.target_state.current_target_track_id = None
            return

        biggest = max(faces, key=lambda f: f.area)

        # ── Anti-thrash stability guard ──
        with self.state_lock:
            if biggest.track_id == self.target_state.last_biggest_track_id:
                self.target_state.biggest_stable_count += 1
            else:
                self.target_state.last_biggest_track_id = biggest.track_id
                self.target_state.biggest_stable_count = 1

            stable = self.target_state.biggest_stable_count >= BIGGEST_STABILITY_COUNT

            if not stable:
                if self.verbose_debug:
                    self._log("DEBUG",
                              f"Selection: waiting for stability "
                              f"({self.target_state.biggest_stable_count}/{BIGGEST_STABILITY_COUNT})")
                return

            # ── 1. TARGET SELECTION: always the biggest bbox ──
            self.target_state.current_target_track_id = biggest.track_id

            # ── 2. INTERACTION TRIGGER: gated checks below ──
            if self.target_state.interaction_busy:
                if self.verbose_debug:
                    self._log("DEBUG", "Selection: interaction busy, skipping trigger")
                return

        # Face still resolving → skip interaction, keep target
        if not self._is_face_resolved(biggest.face_id):
            if self.verbose_debug:
                self._log("DEBUG", f"Selection: face {biggest.track_id} still resolving")
            return

        # LS gate (zone / distance / attention) → skip interaction, keep target
        if not biggest.eligible:
            if self.verbose_debug:
                self._log("DEBUG", f"Selection: {biggest.face_id} not eligible "
                          f"({biggest.zone}/{biggest.distance}/{biggest.attention})")
            return

        # Cooldown → skip interaction, keep target
        person_id = biggest.person_id or biggest.face_id
        last_t = self.last_interaction_time.get(person_id, 0)
        if current_time - last_t < INTERACTION_COOLDOWN:
            if self.verbose_debug:
                remaining = INTERACTION_COOLDOWN - (current_time - last_t)
                self._log("DEBUG", f"Selection: {person_id} in cooldown ({remaining:.1f}s)")
            return

        # ── All gates passed: launch interaction ──
        with self.state_lock:
            self.target_state.interaction_busy = True
        self.last_interaction_time[person_id] = current_time

        self._log("INFO", f"=== SELECTED: {biggest.face_id} (track={biggest.track_id}, "
                  f"{biggest.social_state}, LS{biggest.learning_state}, area={biggest.area:.0f}) ===")

        self._db.log("target_events", {
            "ts": self._now_iso(),
            "track_id": biggest.track_id,
            "face_id": biggest.face_id,
            "bbox_area": biggest.area,
            "zone": biggest.zone,
            "distance": biggest.distance,
            "attention": biggest.attention,
        })

        self.interaction_thread = threading.Thread(
            target=self._run_interaction_thread,
            args=(biggest,),
            daemon=True,
        )
        self.interaction_thread.start()

    # ──────────────── Interaction Thread ────────────────

    def _run_interaction_thread(self, target: FaceObservation):
        """Run interaction via interactionManager RPC (background thread)."""
        try:
            track_id = target.track_id
            face_id = target.person_id if self._is_known(target.person_id) else target.face_id
            ss = target.social_state

            if ss == SS4:
                self._log("INFO", "Interaction: SS4 terminal, skipping")
                return

            self.metrics_attempted += 1
            self._log("INFO", f"=== INTERACTION START: {face_id} (track={track_id}, {ss}) ===")

            # Call interactionManager RPC
            result = self._call_interaction_manager(track_id, face_id, ss)
            if result:
                self._log("INFO", f"Interaction result: success={result.get('success')}")
                # Classify aborts for metrics
                if result.get("aborted"):
                    reason = result.get("abort_reason", "")
                    if "target_lost" in reason:
                        self.metrics_aborted_target_lost += 1
                    else:
                        self.metrics_aborted_no_response += 1
                self._process_interaction_result(result, target)
            else:
                self._log("WARNING", "Interaction: no result from interactionManager")

        except Exception as e:
            self._log("ERROR", f"Interaction thread error: {e}")
            import traceback; traceback.print_exc()

        finally:
            with self.state_lock:
                self.target_state.interaction_busy = False
                self.target_state.current_target_track_id = None
                final_id = self.track_to_person.get(target.track_id, target.face_id)
                self.last_interaction_time[str(final_id)] = time.time()
            self._log("INFO", "=== INTERACTION COMPLETE ===")

    def _call_interaction_manager(self, track_id: int, face_id: str,
                                   social_state: str) -> Optional[Dict]:
        """Send RPC to interactionManager and parse JSON result."""
        cmd = yarp.Bottle()
        cmd.addString("run")
        cmd.addInt32(track_id)
        cmd.addString(face_id)
        cmd.addString(social_state)

        self._log("INFO", f"RPC → interactionManager: {cmd.toString()}")
        reply = rpc_call(self.interaction_manager_rpc, cmd, timeout=RPC_TIMEOUT)
        if not reply or reply.size() < 2:
            self._log("WARNING", "RPC: no valid reply from interactionManager")
            return None

        status = reply.get(0).asString()
        json_str = reply.get(1).asString()

        if status != "ok":
            self._log("WARNING", f"RPC: non-ok status: {status}")
            return None

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self._log("ERROR", f"RPC: JSON parse failed: {e}")
            return None

    def _process_interaction_result(self, result: Dict, target: FaceObservation):
        """Update greeted/talked/learning from interaction result."""
        try:
            success = result.get("success", False)
            updates = result.get("updates", {})
            extracted_name = result.get("extracted_name") or result.get("face_id_out")
            steps = result.get("steps", result.get("trace", {}).get("steps", []))

            # Resolve person_id
            person_id = self._resolve_person_id(result, target)
            track_id = target.track_id

            if not person_id or not self._is_known(person_id):
                # For unknown interactions, check if name was extracted
                if extracted_name and self._is_known(extracted_name):
                    person_id = extracted_name
                else:
                    self._log("INFO", "No known person_id; skipping greeted/talked update")
                    # Still update learning score
                    self._update_learning_from_result(result, person_id or f"track_{track_id}")
                    return

            now_iso = self._now_iso()

            with self.state_lock:
                self.track_to_person[track_id] = person_id

                old_ss = self._compute_social_state_for(person_id)

                # Update greeted/talked from explicit updates or step analysis
                did_greet = updates.get("greeted_today", False)
                did_talk = updates.get("talked_today", False)

                # Fallback: infer from steps if updates not present
                if not updates and isinstance(steps, list):
                    for step in steps:
                        sname = step.get("step", "")
                        if sname in ("ss1", "ss2") and step.get("status") == "success":
                            did_greet = True
                        if sname == "ss3" and step.get("status") in ("success", "finished"):
                            did_talk = True

                if did_greet and self._is_known(person_id):
                    self.greeted_today[person_id] = now_iso
                if did_talk and self._is_known(person_id):
                    self.talked_today[person_id] = now_iso

                new_ss = self._compute_social_state_for(person_id)

            # Log SS change
            if old_ss != new_ss:
                self._db.log("ss_changes", {
                    "ts": now_iso, "person": person_id,
                    "old_ss": old_ss, "new_ss": new_ss,
                })
                self._log("INFO", f"SS change: {person_id} {old_ss} → {new_ss}")

            # Save files outside lock
            save_json_atomic(self.greeted_path, dict(self.greeted_today))
            save_json_atomic(self.talked_path, dict(self.talked_today))

            # Update learning
            self._update_learning_from_result(result, person_id)

        except Exception as e:
            self._log("ERROR", f"Process result error: {e}")
            import traceback; traceback.print_exc()

    def _compute_social_state_for(self, person_id: str) -> str:
        """Compute SS for a person_id from current greeted/talked data. Must hold state_lock."""
        today = self._today()
        is_known = self._is_known(person_id)
        if not is_known:
            return SS1
        greeted = self._was_greeted_today(person_id, today)
        talked = self._was_talked_today(person_id, today)
        if not greeted:
            return SS2
        if not talked:
            return SS3
        return SS4

    def _resolve_person_id(self, result: Dict, target: FaceObservation) -> Optional[str]:
        """Best-effort resolve person_id from result."""
        # From explicit field
        name = result.get("extracted_name") or result.get("face_id_out")
        if name and self._is_known(name):
            return name

        # From steps
        steps = result.get("steps", [])
        if isinstance(steps, list):
            for step in steps:
                details = step.get("details", {})
                n = details.get("extracted_name")
                if n and self._is_known(n):
                    return n

        # From target
        if self._is_known(target.person_id):
            return target.person_id
        if self._is_known(target.face_id):
            return target.face_id
        return None

    # ──────────────── Learning ────────────────

    def _update_learning_from_result(self, result: Dict, person_id: str):
        """Compute reward delta and update LS for person."""
        if not person_id:
            return

        # Don't create permanent LS entries for unknown
        if not self._is_known(person_id):
            self._log("DEBUG", f"LS: skipping update for unknown '{person_id}'")
            return

        delta = self._compute_reward(result)
        old_ls = self._get_ls(person_id)
        new_ls = old_ls
        reason = "no_change"

        if delta >= 4:
            new_ls = min(4, old_ls + 1)
            reason = f"positive_delta_{delta}"
        elif delta <= -4:
            new_ls = max(1, old_ls - 1)
            reason = f"negative_delta_{delta}"

        now_iso = self._now_iso()
        with self.state_lock:
            self.learning_data[person_id] = {"ls": new_ls, "updated_at": now_iso}

        save_json_atomic(self.learning_path, {"people": dict(self.learning_data)})

        if new_ls != old_ls:
            self._log("INFO", f"LS: {person_id} LS{old_ls} → LS{new_ls} (delta={delta:+d})")
        else:
            self._log("INFO", f"LS: {person_id} LS{old_ls} unchanged (delta={delta:+d})")

        self._db.log("ls_changes", {
            "ts": now_iso, "person": person_id,
            "old_ls": old_ls, "new_ls": new_ls,
            "reward_delta": delta, "reason": reason,
        })

    def _compute_reward(self, result: Dict) -> int:
        """Compute reward delta from interaction trace.

        Aligned to actual tree semantics:
          ss1 step: hi response + name extraction success
          ss2 step: greeting response detected
          ss3 step: conversation turns count

        Reward increases for user responses / successful steps.
        Penalise no-response or abort.
        """
        delta = 0
        success = result.get("success", False)
        steps = result.get("steps", [])

        if not isinstance(steps, list):
            return -3 if not success else 0

        for step in steps:
            sname = step.get("step", "")
            details = step.get("details", {})
            status = step.get("status", "")

            if sname == "ss1":
                # ss1: unknown person — hi + name extraction
                responded = details.get("response_detected", False)
                extracted = details.get("extracted_name") or result.get("extracted_name")
                if extracted:
                    # Full success: hi responded AND name extracted
                    delta += 4
                elif responded:
                    # Partial: hi worked but name not extracted
                    delta += 1
                else:
                    # No response at all
                    delta -= 2

            elif sname == "ss2":
                # ss2: known person — greeting response detected
                responded = details.get("response_detected", False)
                attempts = details.get("attempts", 1)
                if responded and attempts == 1:
                    # Responded on first try
                    delta += 3
                elif responded:
                    # Responded after retry
                    delta += 1
                else:
                    # No response
                    delta -= 2

            elif sname == "ss3":
                # ss3: conversation turns
                turns = details.get("turns_count", 0)
                if turns >= 3:
                    delta += 4
                elif turns >= 2:
                    delta += 2
                elif turns >= 1:
                    delta += 1
                else:
                    delta -= 2

        # Global failure penalty (abort, no response, etc.)
        if not success:
            delta -= 3

        return delta

    # ──────────────── Last Greeted Sync ────────────────

    def _sync_last_greeted(self):
        """Read last_greeted.json and update greeted_today for known people."""
        entries = load_json_safe(self.last_greeted_path, [])
        if not isinstance(entries, list):
            return

        today = self._today()
        changed = False

        for entry in entries:
            name = entry.get("assigned_code_or_name", "")
            ts = entry.get("timestamp", "")
            if not name or not ts or not self._is_known(name):
                continue
            try:
                dt = datetime.fromisoformat(ts).astimezone(TIMEZONE)
                if dt.date() == today and name not in self.greeted_today:
                    self.greeted_today[name] = ts
                    changed = True
            except Exception:
                pass

        if changed:
            save_json_atomic(self.greeted_path, dict(self.greeted_today))

        self.last_greeted = entries

    # ──────────────── Data Persistence ────────────────

    def _load_all_data(self):
        """Load all JSON files."""
        self.greeted_today = load_json_safe(self.greeted_path, {})
        self.talked_today = load_json_safe(self.talked_path, {})
        raw = load_json_safe(self.learning_path, {"people": {}})
        self.learning_data = raw.get("people", {}) if isinstance(raw, dict) else {}

        # Prune
        self.greeted_today = prune_to_today(self.greeted_today, TIMEZONE)
        self.talked_today = prune_to_today(self.talked_today, TIMEZONE)

        # Sync last_greeted
        self._sync_last_greeted()

        self._log("INFO", f"Loaded: {len(self.greeted_today)} greeted, "
                  f"{len(self.talked_today)} talked, {len(self.learning_data)} learning")

    def _save_all_data(self):
        save_json_atomic(self.greeted_path, self.greeted_today)
        save_json_atomic(self.talked_path, self.talked_today)
        save_json_atomic(self.learning_path, {"people": self.learning_data})

    # ──────────────── Debug Output ────────────────

    def _publish_debug(self, faces: List[FaceObservation]):
        if self.debug_port.getOutputCount() == 0:
            return
        try:
            btl = yarp.Bottle()
            btl.clear()

            btl.addString("status")
            with self.state_lock:
                btl.addString("busy" if self.target_state.interaction_busy else "idle")
                btl.addString("ao")
                btl.addString("on" if self.target_state.ao_running else "off")

            btl.addString("metrics_attempted")
            btl.addInt32(self.metrics_attempted)
            btl.addString("metrics_aborted_target_lost")
            btl.addInt32(self.metrics_aborted_target_lost)
            btl.addString("metrics_aborted_no_response")
            btl.addInt32(self.metrics_aborted_no_response)

            btl.addString("face_count")
            btl.addInt32(len(faces))

            if faces:
                biggest = max(faces, key=lambda f: f.area)
                btl.addString("biggest_face_id")
                btl.addString(biggest.face_id)
                btl.addString("biggest_track_id")
                btl.addInt32(biggest.track_id)
                btl.addString("biggest_ss")
                btl.addString(biggest.social_state)
                btl.addString("biggest_area")
                btl.addFloat64(biggest.area)

            # Last greeted info
            if self.last_greeted:
                last = self.last_greeted[-1]
                btl.addString("last_greeted")
                btl.addString(str(last.get("assigned_code_or_name", "?")))

            self.debug_port.write(btl)
        except Exception as e:
            self._log("WARNING", f"Debug publish error: {e}")

    # ──────────────── Utilities ────────────────

    @staticmethod
    def _today() -> date:
        return datetime.now(TIMEZONE).date()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(TIMEZONE).isoformat()

    def _log(self, level: str, message: str):
        ts = datetime.now(TIMEZONE).strftime("%H:%M:%S.%f%z")[:-5]
        print(f"[{ts}] [{level}] {message}")


# ═══════════════════════ Main ═══════════════════════

if __name__ == "__main__":
    yarp.Network.init()

    if not yarp.Network.checkNetwork():
        print("ERROR: YARP network not available. Start yarpserver first.")
        sys.exit(1)

    module = FaceSelectorModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("alwaysOn")
    rf.configure(sys.argv)

    print("=" * 60)
    print("FaceSelectorModule - Biggest-BBox Face Selection")
    print("=" * 60)
    print()
    print("YARP Connections:")
    print("  yarp connect /alwayson/vision/landmarks:o /faceSelector/landmarks:i")
    print()
    print("RPC connections (auto-connected on startup):")
    print("  /faceSelector/interactionManager:rpc → /interactionManager")
    print("  /faceSelector/interactionInterface:rpc → /interactionInterface")
    print()
    print("Optional:")
    print("  yarp connect /faceSelector/debug:o /debugViewer")
    print()
    print("Configuration:")
    print("  --name <module_name>              (default: faceSelector)")
    print("  --interaction_manager_rpc <port>   (default: /interactionManager)")
    print("  --interaction_interface_rpc <port>  (default: /interactionInterface)")
    print("  --learning_path <file>             (default: ./learning.json)")
    print("  --greeted_path <file>              (default: ./greeted_today.json)")
    print("  --talked_path <file>               (default: ./talked_today.json)")
    print("  --last_greeted_path <file>         (default: ./last_greeted.json)")
    print("  --rate <seconds>                   (default: 0.05)")
    print("  --verbose <true/false>             (default: false)")
    print()

    try:
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
