"""
interactionManager.py - YARP RFModule for Social Interaction State Trees

Executes 3 interaction trees (ss1/ss2/ss3), no context port, no ss4/ss5.
Uses Ollama LLM (Phi-3 mini) for NLU/NLG.
Monitors landmarks to abort if target is no longer the biggest bbox.

YARP Connections (run after starting):
    yarp connect /alwayson/vision/landmarks:o /interactionManager/landmarks:i
    yarp connect /speech2text/text:o /interactionManager/stt:i
    yarp connect /interactionManager/speech:o /acapelaSpeak/speech:i

RPC Usage:
    echo "run <track_id> <face_id> <ss1|ss2|ss3>" | yarp rpc /interactionManager

Result contract (JSON):
    {
        "success": bool,
        "aborted": bool,
        "abort_reason": str | null,
        "track_id": int,
        "face_id_in": str,
        "face_id_out": str | null,
        "social_state_in": str,
        "social_state_out": str,
        "updates": {"greeted_today": bool, "talked_today": bool},
        "extracted_name": str | null,
        "dialogue": {"user_utterances": [], "robot_utterances": []},
        "steps": [...],
        "timing": {"total_s": float, ...}
    }
"""

import concurrent.futures
import json
import os
import queue
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import yarp


# ═══════════════════════ Constants ═══════════════════════

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "phi3:mini"
DB_FILE = "interaction_data.db"

# Timezone (must match faceSelector)
TIMEZONE = ZoneInfo("Europe/Rome")

# Timeouts (seconds)
SS1_STT_TIMEOUT = 7.0       # Hi response
SS1_NAME_STT_TIMEOUT = 10.0 # Name ask response
SS2_STT_TIMEOUT = 8.0       # Known greeting response
SS3_STT_TIMEOUT = 12.0      # Conversation turn response
LLM_TIMEOUT = 60.0

# SS3 conversation
SS3_MAX_TURNS = 3
SS3_MAX_TIME = 120.0

# Landmarks monitor
LANDMARKS_POLL_HZ = 10.0  # How often monitor checks landmarks

# TTS timing (words per second)
TTS_WPS = 3.0
TTS_END_MARGIN = 0.5
TTS_MIN_WAIT = 1.0
TTS_MAX_WAIT = 8.0

# Valid social states for RPC
VALID_STATES = {"ss1", "ss2", "ss3"}

# LLM retry
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_DELAY = 1.0

# Greeting keywords for fast matching (no LLM needed)
GREETING_KEYWORDS = {
    "hello", "hi", "hey", "ciao", "hola", "salut", "hallo", "hej",
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok",
    "good morning", "good afternoon", "good evening", "good day",
    "howdy", "greetings", "sup", "what's up", "whats up",
    "bonjour", "buongiorno", "guten tag", "buenos dias",
}


# ═══════════════════════ Async DB Writer ═══════════════════════

class AsyncDBWriter:
    """Queue + single writer thread, WAL mode."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="im-db-writer")
        self._init_db()
        self._thread.start()

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                track_id INTEGER,
                initial_face_id TEXT,
                initial_state TEXT,
                final_state TEXT,
                success BOOLEAN,
                aborted BOOLEAN,
                abort_reason TEXT,
                extracted_name TEXT,
                transitions TEXT,
                full_result_json TEXT
            );
        """)
        conn.commit()
        conn.close()

    def log_interaction(self, track_id: int, face_id: str, initial_state: str, result: Dict):
        try:
            entry = {
                "track_id": track_id,
                "initial_face_id": face_id,
                "initial_state": initial_state,
                "final_state": result.get("social_state_out", ""),
                "success": result.get("success", False),
                "aborted": result.get("aborted", False),
                "abort_reason": result.get("abort_reason"),
                "extracted_name": result.get("extracted_name"),
                "transitions": "",
                "full_result_json": json.dumps(result, ensure_ascii=False, default=str),
            }
            self._queue.put_nowait(entry)
        except queue.Full:
            pass

    def _run(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        while not self._stop.is_set():
            try:
                data = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                cols = ", ".join(data.keys())
                ph = ", ".join("?" for _ in data)
                conn.execute(f"INSERT INTO interactions ({cols}) VALUES ({ph})",
                             list(data.values()))
                conn.commit()
            except Exception:
                pass
        conn.close()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)


# ═══════════════════════ JSON Utilities ═══════════════════════

def _load_json_safe(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save_json_atomic(path: str, data: Any) -> bool:
    try:
        parent = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(suffix=".json", dir=parent)
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


# ═══════════════════════ Shared Bbox Utility ═══════════════════════

def parse_face_bottle_to_dict(bottle: yarp.Bottle) -> Optional[Dict]:
    """Parse a single face bottle into a plain dict, handling both wire formats.

    Format A – flat key/value pairs (some vision modules):
        face_id "Alice" track_id 3 bbox [x y w h] ...

    Format B – nested sub-list for bbox (alwayson vision / faceSelector):
        face_id "Alice" track_id 3 (bbox x y w h) ...
        where bbox appears as a sub-list whose first element is the
        string "bbox" followed by four floats.

    Returns a dict with keys: face_id, track_id, zone, distance, attention,
    is_talking, time_in_view, pitch, yaw, roll, cos_angle, bbox, area.
    Returns None if the bottle is empty or unparseable.
    """
    data: Dict = {}
    try:
        i = 0
        while i < bottle.size():
            item = bottle.get(i)

            # ── Format B: nested sub-list, e.g. (bbox x y w h) ──
            if item.isList():
                nested = item.asList()
                if nested.size() >= 2 and nested.get(0).isString():
                    key = nested.get(0).asString()
                    if key == "bbox" and nested.size() >= 5:
                        w = nested.get(3).asFloat64()
                        h = nested.get(4).asFloat64()
                        data["bbox"] = [
                            nested.get(1).asFloat64(),
                            nested.get(2).asFloat64(),
                            w, h,
                        ]
                        data["area"] = w * h
                i += 1
                continue

            # ── Format A: flat key/value pairs ──
            if not item.isString() or i + 1 >= bottle.size():
                i += 1
                continue

            key = item.asString()
            val = bottle.get(i + 1)

            if key in ("face_id", "zone", "distance", "attention"):
                data[key] = val.asString()
            elif key in ("track_id", "is_talking"):
                data[key] = val.asInt32()
            elif key in ("time_in_view", "pitch", "yaw", "roll", "cos_angle"):
                data[key] = val.asFloat64()
            elif key == "bbox" and val.isList():
                # Format A bbox: key "bbox" followed by a list value
                lst = val.asList()
                vals = [lst.get(j).asFloat64() for j in range(lst.size())]
                data["bbox"] = vals
                if len(vals) >= 4:
                    data["area"] = vals[2] * vals[3]
            i += 2

        return data if data else None
    except Exception:
        return None


# ═══════════════════════ Target Monitor Thread ═══════════════════════

class TargetMonitor:
    """Monitors landmarks to check if target remains the biggest bbox.

    Sets `lost_event` if target disappears or another face becomes bigger.
    """

    def __init__(self, landmarks_port: yarp.BufferedPortBottle,
                 track_id: int, log_fn):
        self.landmarks_port = landmarks_port
        self.target_track_id = track_id
        self._log = log_fn
        self.lost_event = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="target-monitor")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def is_lost(self) -> bool:
        return self.lost_event.is_set()

    def _run(self):
        interval = 1.0 / LANDMARKS_POLL_HZ
        consecutive_lost = 0
        while not self._stop.is_set():
            try:
                faces = self._parse_latest()
                if not faces:
                    consecutive_lost += 1
                    if consecutive_lost >= 3:
                        self._log("WARNING", "Monitor: target lost (no faces)")
                        self.lost_event.set()
                        return
                else:
                    # Find biggest by area
                    biggest = max(faces, key=lambda f: f.get("area", 0))
                    if biggest.get("track_id") != self.target_track_id:
                        consecutive_lost += 1
                        if consecutive_lost >= 3:
                            self._log("WARNING",
                                      f"Monitor: target lost (biggest is track "
                                      f"{biggest.get('track_id')}, not {self.target_track_id})")
                            self.lost_event.set()
                            return
                    else:
                        consecutive_lost = 0
            except Exception:
                pass
            time.sleep(interval)

    def _parse_latest(self) -> List[Dict]:
        bottle = self.landmarks_port.read(False)
        if not bottle:
            return []
        result = []
        for i in range(bottle.size()):
            item = bottle.get(i)
            if not item.isList():
                continue
            face = self._parse_face(item.asList())
            if face:
                result.append(face)
        return result

    @staticmethod
    def _parse_face(bottle: yarp.Bottle) -> Optional[Dict]:
        """Delegate to shared module-level bbox parser."""
        return parse_face_bottle_to_dict(bottle)


# ═══════════════════════ Module ═══════════════════════

class InteractionManagerModule(yarp.RFModule):
    """
    Social interaction executor with 3 trees:
    - ss1 (unknown): hi + name extraction
    - ss2 (known, not greeted): greet by name → chain into ss3
    - ss3 (known, greeted, not talked): short 3-turn conversation

    ss4 is terminal; no tree executed.
    """

    def __init__(self):
        super().__init__()
        self.module_name = "interactionManager"
        self.period = 1.0
        self._running = True
        self.run_lock = threading.Lock()
        self.log_buffer: List[Dict] = []

        # RPC handle
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # YARP ports
        self.landmarks_port: Optional[yarp.BufferedPortBottle] = None
        self.stt_port: Optional[yarp.BufferedPortBottle] = None
        self.speech_port: Optional[yarp.Port] = None

        # Native RPC clients (lazy)
        self._interaction_rpc: Optional[yarp.RpcClient] = None
        self._vision_rpc: Optional[yarp.RpcClient] = None

        # Cached conversation starter
        self._cached_starter: Optional[str] = None

        # LLM health
        self.ollama_last_check = 0.0
        self.ollama_check_interval = 60.0

        # DB writer
        self._db: Optional[AsyncDBWriter] = None

    # ──────────────── Lifecycle ────────────────

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        try:
            if rf.check("name"):
                self.module_name = rf.find("name").asString()
            self.setName(self.module_name)

            # Open RPC handle
            self.handle_port.open("/" + self.module_name)
            self._log("INFO", f"RPC port opened at /{self.module_name}")

            # Create ports
            self.landmarks_port = yarp.BufferedPortBottle()
            self.stt_port = yarp.BufferedPortBottle()
            self.speech_port = yarp.Port()

            ports = [
                (self.landmarks_port, "landmarks:i"),
                (self.stt_port, "stt:i"),
                (self.speech_port, "speech:o"),
            ]
            for port, suffix in ports:
                pname = f"/{self.module_name}/{suffix}"
                if not port.open(pname):
                    self._log("ERROR", f"Failed to open {pname}")
                    return False
                self._log("INFO", f"Opened {pname}")

            # Init tracking file
            if not os.path.exists("last_greeted.json"):
                _save_json_atomic("last_greeted.json", [])

            # DB
            self._db = AsyncDBWriter(DB_FILE)

            # Ollama (background)
            threading.Thread(target=self._init_ollama_background, daemon=True).start()

            # Prefetch starter
            threading.Thread(target=self._generate_starter_background, daemon=True).start()

            self._log("INFO", "InteractionManagerModule configured successfully")
            return True

        except Exception as e:
            self._log("ERROR", f"Configuration failed: {e}")
            import traceback; traceback.print_exc()
            return False

    def interruptModule(self) -> bool:
        self._log("INFO", "Interrupting module...")
        self._running = False
        self.handle_port.interrupt()
        for port in [self.landmarks_port, self.stt_port, self.speech_port]:
            if port:
                port.interrupt()
        return True

    def close(self) -> bool:
        self._log("INFO", "Closing module...")
        self.handle_port.close()
        for port in [self.landmarks_port, self.stt_port, self.speech_port]:
            if port:
                port.close()
        if self._interaction_rpc:
            self._interaction_rpc.close()
            self._interaction_rpc = None
        if self._vision_rpc:
            self._vision_rpc.close()
            self._vision_rpc = None
        if self._db:
            self._db.stop()
        return True

    def getPeriod(self) -> float:
        return self.period

    def updateModule(self) -> bool:
        return self._running

    # ──────────────── RPC Handler ────────────────

    def respond(self, cmd: yarp.Bottle, reply: yarp.Bottle) -> bool:
        reply.clear()
        try:
            if cmd.size() < 1:
                return self._reply_error(reply, "Empty command")

            command = cmd.get(0).asString()
            self._log("DEBUG", f"RPC received: {command}")

            if command in ("status", "ping"):
                busy = not self.run_lock.acquire(blocking=False)
                if not busy:
                    self.run_lock.release()
                return self._reply_ok(reply, {
                    "success": True, "status": "ready",
                    "module": self.module_name, "busy": busy,
                })

            if command == "help":
                return self._reply_ok(reply, {
                    "success": True,
                    "commands": [
                        "run <track_id> <face_id> <ss1|ss2|ss3>",
                        "status - check module status",
                        "help - show commands",
                        "quit - shutdown module",
                    ],
                })

            if command == "quit":
                self._running = False
                self.stopModule()
                return self._reply_ok(reply, {"success": True, "message": "Shutting down"})

            if command != "run":
                return self._reply_error(reply, f"Unknown command: {command}")

            if cmd.size() < 4:
                return self._reply_error(reply, "Usage: run <track_id> <face_id> <ss1|ss2|ss3>")

            track_id = cmd.get(1).asInt32()
            face_id = cmd.get(2).asString()
            social_state = cmd.get(3).asString().lower()

            if social_state not in VALID_STATES:
                return self._reply_error(reply, f"Invalid state: {social_state}")

            if not self.run_lock.acquire(blocking=False):
                self._log("WARNING", "Another interaction in progress")
                return self._reply_error(reply, "Another action is running")

            try:
                self.log_buffer = []
                self._log("INFO", "=== Starting new interaction ===")
                self._log("INFO", f"Params: track_id={track_id}, face_id={face_id}, state={social_state}")

                result = self._execute_interaction(track_id, face_id, social_state)

                # Async DB save
                self._db.log_interaction(track_id, face_id, social_state, result)

                reply.addString("ok")
                reply.addString(json.dumps(result, ensure_ascii=False, default=str))
            finally:
                self.run_lock.release()

            return True

        except Exception as e:
            self._log("ERROR", f"Exception in respond: {e}")
            import traceback; traceback.print_exc()
            try:
                self.run_lock.release()
            except RuntimeError:
                pass
            return self._reply_error(reply, str(e))

    def _reply_ok(self, reply: yarp.Bottle, data: Dict) -> bool:
        reply.addString("ok")
        reply.addString(json.dumps(data, ensure_ascii=False))
        return True

    def _reply_error(self, reply: yarp.Bottle, error: str) -> bool:
        reply.addString("ok")
        reply.addString(json.dumps({
            "success": False, "error": error, "logs": self.log_buffer,
        }, ensure_ascii=False))
        return True

    # ──────────────── Interaction Execution ────────────────

    @staticmethod
    def _is_face_resolved(face_id: str) -> bool:
        return face_id.lower() not in ("recognizing", "unmatched")

    def _execute_interaction(self, track_id: int, face_id: str,
                              social_state: str) -> Dict[str, Any]:
        """Execute the appropriate tree and return the full result contract."""
        t0 = time.time()

        result = {
            "success": False,
            "aborted": False,
            "abort_reason": None,
            "track_id": track_id,
            "face_id_in": face_id,
            "face_id_out": None,
            "social_state_in": social_state,
            "social_state_out": social_state,
            "updates": {"greeted_today": False, "talked_today": False},
            "extracted_name": None,
            "dialogue": {"user_utterances": [], "robot_utterances": []},
            "steps": [],
            "timing": {},
        }

        # Resolve face_id if still recognising
        if not self._is_face_resolved(face_id):
            self._log("INFO", f"face_id unresolved, waiting up to 5s...")
            face_id = self._wait_for_resolution(track_id, face_id, timeout=5.0)
            if not self._is_face_resolved(face_id):
                self._log("WARNING", "face_id still unresolved — aborting")
                result["aborted"] = True
                result["abort_reason"] = "face_id_unresolved"
                result["timing"]["total_s"] = time.time() - t0
                return result

        # Start target monitor
        monitor = TargetMonitor(self.landmarks_port, track_id, self._log)
        monitor.start()

        try:
            # Ensure STT ready
            self._log("INFO", "STT ready")

            if social_state == "ss1":
                self._run_ss1(track_id, face_id, result, monitor)

            elif social_state == "ss2":
                self._run_ss2(track_id, face_id, result, monitor)
                # ss2 chains into ss3 if successful
                if result["steps"] and result["steps"][-1].get("status") == "success":
                    if not monitor.is_lost:
                        self._run_ss3(track_id, face_id, result, monitor)

            elif social_state == "ss3":
                self._run_ss3(track_id, face_id, result, monitor)

        finally:
            monitor.stop()

        # Determine overall success and final state
        self._finalize_result(result, social_state)

        result["timing"]["total_s"] = time.time() - t0
        result["face_id_out"] = result.get("extracted_name") or face_id
        return result

    def _finalize_result(self, result: Dict, initial_state: str):
        """Set success, social_state_out based on steps."""
        steps = result["steps"]
        updates = result["updates"]

        if not steps:
            return

        last_step = steps[-1]

        if initial_state == "ss1":
            # ss1: success if name extracted
            if result["extracted_name"]:
                result["success"] = True
                updates["greeted_today"] = True
                updates["talked_today"] = True  # hi + name exchange counts as talked
                result["social_state_out"] = "ss4"  # They become known, greeted, talked
            elif last_step.get("status") == "success":
                # hi succeeded but no name
                result["success"] = False

        elif initial_state == "ss2":
            # ss2 chains into ss3; check if ss3 completed
            ss3_steps = [s for s in steps if s.get("step") == "ss3"]
            ss2_steps = [s for s in steps if s.get("step") == "ss2"]

            if ss2_steps and ss2_steps[-1].get("status") == "success":
                updates["greeted_today"] = True

            if ss3_steps and ss3_steps[-1].get("status") in ("success", "finished"):
                updates["talked_today"] = True
                result["success"] = True
                result["social_state_out"] = "ss4"
            elif ss2_steps and ss2_steps[-1].get("status") == "success":
                result["success"] = True
                result["social_state_out"] = "ss3"

        elif initial_state == "ss3":
            if last_step.get("status") in ("success", "finished"):
                updates["talked_today"] = True
                result["success"] = True
                result["social_state_out"] = "ss4"

    # ──────────────── SS1: Unknown Person ────────────────

    def _run_ss1(self, track_id: int, face_id: str,
                  result: Dict, monitor: TargetMonitor):
        """
        SS1 tree for unknown person:
        1) ao_hi
        2) Wait for response; if none → abort
        3) Ask name; extract
        4) If no extraction → ask again once
        5) If extracted → "Nice to meet you <name>", register, update last_greeted
        6) End
        """
        t0 = time.time()
        step = {"step": "ss1", "status": "failed", "details": {}}
        result["steps"].append(step)

        self._clear_stt_buffer()

        # 1. Execute ao_hi
        self._log("INFO", "SS1: Executing ao_hi")
        threading.Thread(target=self._execute_behaviour, args=("ao_hi",), daemon=True).start()
        step["details"]["greet_attempt"] = "successful"

        # Log greeting
        threading.Thread(target=self._write_last_greeted,
                         args=(track_id, face_id, face_id), daemon=True).start()

        # 2. Wait for response
        self._log("INFO", "SS1: Waiting for hi response")
        utterance = self._wait_utterance_interruptible(SS1_STT_TIMEOUT, monitor)
        step["details"]["hi_utterance"] = utterance

        if monitor.is_lost:
            result["aborted"] = True
            result["abort_reason"] = "target_lost_during_ss1_hi"
            step["details"]["abort"] = True
            step["timing_s"] = time.time() - t0
            return

        if not utterance:
            self._log("WARNING", "SS1: No hi response, aborting")
            step["details"]["response_detected"] = False
            step["timing_s"] = time.time() - t0
            return

        # Response detected
        step["details"]["response_detected"] = True

        # 3. Ask name
        self._log("INFO", "SS1: Asking name")
        self._clear_stt_buffer()
        self._speak("We have not met, what's your name?")

        name_utterance = self._wait_utterance_interruptible(SS1_NAME_STT_TIMEOUT, monitor)
        step["details"]["name_utterance_1"] = name_utterance

        if monitor.is_lost:
            result["aborted"] = True
            result["abort_reason"] = "target_lost_during_ss1_name"
            step["details"]["abort"] = True
            step["timing_s"] = time.time() - t0
            return

        extracted_name = None
        if name_utterance:
            result["dialogue"]["user_utterances"].append(name_utterance)
            extraction = self._llm_extract_name(name_utterance)
            step["details"]["extraction_1"] = extraction
            if extraction.get("answered") and extraction.get("name"):
                extracted_name = extraction["name"]

        # 4. Retry once if no extraction
        if not extracted_name and not monitor.is_lost:
            self._log("INFO", "SS1: Name not extracted, asking again")
            self._clear_stt_buffer()
            self._speak("Sorry, I didn't catch your name. Could you tell me again?")

            name_utterance2 = self._wait_utterance_interruptible(SS1_NAME_STT_TIMEOUT, monitor)
            step["details"]["name_utterance_2"] = name_utterance2

            if monitor.is_lost:
                result["aborted"] = True
                result["abort_reason"] = "target_lost_during_ss1_name_retry"
                step["details"]["abort"] = True
                step["timing_s"] = time.time() - t0
                return

            if name_utterance2:
                result["dialogue"]["user_utterances"].append(name_utterance2)
                extraction2 = self._llm_extract_name(name_utterance2)
                step["details"]["extraction_2"] = extraction2
                if extraction2.get("answered") and extraction2.get("name"):
                    extracted_name = extraction2["name"]

        # 5. Still no name → abort
        if not extracted_name:
            self._log("WARNING", "SS1: Could not extract name, aborting")
            step["timing_s"] = time.time() - t0
            return

        # 6. Success - register
        self._log("INFO", f"SS1: Name extracted: '{extracted_name}'")
        result["extracted_name"] = extracted_name
        step["details"]["extracted_name"] = extracted_name

        # Say nice to meet you
        self._speak_and_wait(f"Nice to meet you {extracted_name}")
        result["dialogue"]["robot_utterances"].append(f"Nice to meet you {extracted_name}")

        # Register face (async)
        threading.Thread(target=self._submit_face_name,
                         args=(track_id, extracted_name), daemon=True).start()

        # Update last_greeted with name (async)
        threading.Thread(target=self._update_last_greeted_name,
                         args=(track_id, extracted_name), daemon=True).start()

        step["status"] = "success"
        step["timing_s"] = time.time() - t0

    # ──────────────── SS2: Known, Not Greeted ────────────────

    def _run_ss2(self, track_id: int, face_id: str,
                  result: Dict, monitor: TargetMonitor):
        """
        SS2 tree for known person not yet greeted:
        1) Say "Hi <name>"
        2) If responded → success (caller chains into ss3)
        3) If not → say hi again once
        4) If responded → success; else abort
        """
        t0 = time.time()
        step = {"step": "ss2", "status": "failed", "details": {"attempts": 0}}
        result["steps"].append(step)

        # Validate name
        if face_id.lower() in ("unknown", "unmatched", "recognizing"):
            self._log("WARNING", f"SS2: face_id '{face_id}' is not a valid name, aborting")
            step["details"]["error"] = "invalid_name"
            step["timing_s"] = time.time() - t0
            return

        max_attempts = 2
        for attempt in range(max_attempts):
            step["details"]["attempts"] = attempt + 1
            self._log("INFO", f"SS2: Greeting '{face_id}' (attempt {attempt + 1}/{max_attempts})")

            self._clear_stt_buffer()
            self._speak_and_wait(f"Hi {face_id}")
            result["dialogue"]["robot_utterances"].append(f"Hi {face_id}")

            if monitor.is_lost:
                result["aborted"] = True
                result["abort_reason"] = "target_lost_during_ss2"
                step["details"]["abort"] = True
                step["timing_s"] = time.time() - t0
                return

            utterance = self._wait_utterance_interruptible(SS2_STT_TIMEOUT, monitor)
            step["details"][f"utterance_{attempt + 1}"] = utterance

            if monitor.is_lost:
                result["aborted"] = True
                result["abort_reason"] = "target_lost_during_ss2_listen"
                step["details"]["abort"] = True
                step["timing_s"] = time.time() - t0
                return

            if utterance:
                result["dialogue"]["user_utterances"].append(utterance)
                detection = self._detect_greeting(utterance)
                step["details"]["response_detected"] = detection["responded"]

                if detection["responded"]:
                    self._log("INFO", "SS2: Response detected, greeting successful")
                    step["status"] = "success"
                    step["timing_s"] = time.time() - t0

                    # Log greeting
                    threading.Thread(target=self._write_last_greeted,
                                     args=(track_id, face_id, face_id), daemon=True).start()
                    return

        self._log("INFO", "SS2: No response after all attempts")
        step["timing_s"] = time.time() - t0

    # ──────────────── SS3: Known, Greeted, Not Talked ────────────────

    def _run_ss3(self, track_id: int, face_id: str,
                  result: Dict, monitor: TargetMonitor):
        """
        SS3: Short casual conversation (max 3 turns).
        Turn 1: Robot asks starter, waits user.
        Turn 2: Robot followup from LLM, waits user.
        Turn 3: Robot closing acknowledgment only (no question).

        If no user response at all → abort.
        After at least one user response → talked_today = true.
        """
        t0 = time.time()
        step = {"step": "ss3", "status": "failed", "details": {
            "turns_count": 0, "dialogue_transcript": [],
        }}
        result["steps"].append(step)

        user_responded_at_least_once = False

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as llm_pool:
            for turn in range(SS3_MAX_TURNS):
                if monitor.is_lost:
                    result["aborted"] = True
                    result["abort_reason"] = "target_lost_during_ss3"
                    step["details"]["abort"] = True
                    break

                if time.time() - t0 > SS3_MAX_TIME:
                    self._log("INFO", "SS3: Time limit reached")
                    break

                if turn == 0:
                    # First turn: conversation starter
                    if self._cached_starter:
                        starter = self._cached_starter
                        self._cached_starter = None
                        self._log("INFO", f"SS3: Using cached starter: '{starter}'")
                    else:
                        self._log("INFO", "SS3: Generating starter")
                        starter = self._llm_generate_convo_starter() or "How are you doing these days?"

                    # Refill cache in background
                    threading.Thread(target=self._generate_starter_background, daemon=True).start()

                    self._log("INFO", f"SS3 Turn 1: Robot says: '{starter}'")
                    result["dialogue"]["robot_utterances"].append(starter)
                    step["details"]["dialogue_transcript"].append(f"Robot: {starter}")
                    self._speak_and_wait(starter)

                elif turn == SS3_MAX_TURNS - 1:
                    # Last turn (turn 3): closing ack only, no question
                    # Robot already spoke the ack from previous turn's LLM, or generates one now
                    # Actually: we generate closing ack, speak it, and don't wait for user
                    last_user = result["dialogue"]["user_utterances"][-1] if result["dialogue"]["user_utterances"] else ""
                    self._log("INFO", "SS3: Generating closing acknowledgment")

                    try:
                        ack = llm_pool.submit(self._llm_generate_closing_ack, last_user).result(timeout=LLM_TIMEOUT)
                    except Exception:
                        ack = "That's great, nice talking with you!"

                    ack = ack or "That's great, nice talking with you!"
                    self._log("INFO", f"SS3 Turn {turn + 1}: Robot says (closing): '{ack}'")
                    result["dialogue"]["robot_utterances"].append(ack)
                    step["details"]["dialogue_transcript"].append(f"Robot: {ack}")
                    self._speak_and_wait(ack)
                    break  # No more waiting

                else:
                    # Middle turns: LLM followup
                    last_user = result["dialogue"]["user_utterances"][-1] if result["dialogue"]["user_utterances"] else ""
                    self._log("INFO", "SS3: Generating followup")

                    try:
                        followup = llm_pool.submit(
                            self._llm_generate_followup, last_user,
                            list(result["dialogue"]["user_utterances"])
                        ).result(timeout=LLM_TIMEOUT)
                    except Exception:
                        followup = "I see, tell me more!"

                    followup = followup or "I see, tell me more!"
                    self._log("INFO", f"SS3 Turn {turn + 1}: Robot says: '{followup}'")
                    result["dialogue"]["robot_utterances"].append(followup)
                    step["details"]["dialogue_transcript"].append(f"Robot: {followup}")
                    self._speak_and_wait(followup)

                # Wait for user (except on last turn where we already broke)
                if turn < SS3_MAX_TURNS - 1:
                    self._log("INFO", f"SS3 Turn {turn + 1}: Listening for user")
                    utterance = self._wait_utterance_interruptible(SS3_STT_TIMEOUT, monitor)

                    if monitor.is_lost:
                        result["aborted"] = True
                        result["abort_reason"] = "target_lost_during_ss3_listen"
                        step["details"]["abort"] = True
                        break

                    if utterance:
                        result["dialogue"]["user_utterances"].append(utterance)
                        step["details"]["dialogue_transcript"].append(f"User: {utterance}")
                        step["details"]["turns_count"] = turn + 1
                        user_responded_at_least_once = True
                        self._log("INFO", f"SS3 Turn {turn + 1}: User said: '{utterance}'")
                    else:
                        self._log("INFO", f"SS3 Turn {turn + 1}: No response")
                        break

        if user_responded_at_least_once:
            step["status"] = "success"
            self._log("INFO", f"SS3: Conversation successful ({step['details']['turns_count']} turns)")
        else:
            self._log("WARNING", "SS3: User never responded")

        step["timing_s"] = time.time() - t0

    # ──────────────── Interruptible Waiting ────────────────

    def _wait_utterance_interruptible(self, timeout: float,
                                        monitor: TargetMonitor) -> Optional[str]:
        """Wait for STT, checking target monitor periodically."""
        start = time.time()
        while time.time() - start < timeout:
            if monitor.is_lost:
                self._log("DEBUG", "Wait interrupted: target lost")
                return None
            bottle = self.stt_port.read(False)
            if bottle and bottle.size() > 0:
                text = self._extract_stt_text(bottle)
                if text and text.strip():
                    return text.strip()
            time.sleep(0.1)
        return None

    def _wait_for_resolution(self, track_id: int, face_id: str,
                               timeout: float = 5.0) -> str:
        """Wait for face_id to resolve from landmarks."""
        start = time.time()
        while time.time() - start < timeout:
            faces = self._parse_landmarks()
            for f in faces:
                if f.get("track_id") == track_id:
                    fid = f.get("face_id", "recognising")
                    if self._is_face_resolved(fid):
                        self._log("INFO", f"face_id resolved to '{fid}'")
                        return fid
            time.sleep(0.2)
        return face_id

    # ──────────────── Landmarks Parsing ────────────────

    def _parse_landmarks(self) -> List[Dict]:
        """Parse latest landmarks from port."""
        result = []
        try:
            bottle = self.landmarks_port.read(False)
            if bottle:
                for i in range(bottle.size()):
                    face = bottle.get(i).asList()
                    if face:
                        data = self._parse_face_bottle(face)
                        if data:
                            result.append(data)
        except Exception as e:
            self._log("WARNING", f"Landmarks parse failed: {e}")
        return result

    @staticmethod
    def _parse_face_bottle(bottle: yarp.Bottle) -> Optional[Dict]:
        """Delegate to shared module-level bbox parser."""
        return parse_face_bottle_to_dict(bottle)

    # ──────────────── STT ────────────────

    def _extract_stt_text(self, bottle: yarp.Bottle) -> Optional[str]:
        """Extract text from STT bottle format: ("text" "speaker")."""
        try:
            if bottle.size() >= 1:
                raw = bottle.get(0).toString()
                if raw.startswith('"'):
                    end_idx = raw.find('"', 1)
                    if end_idx > 1:
                        return raw[1:end_idx].strip()
                elif ' ""' in raw:
                    return raw.split(' ""')[0].strip()
                else:
                    text = bottle.get(0).asString()
                    if text and text.strip():
                        return text.strip()
        except Exception as e:
            self._log("WARNING", f"STT parse failed: {e}")
        return None

    def _clear_stt_buffer(self):
        cleared = 0
        while self.stt_port.read(False):
            cleared += 1
        if cleared > 0:
            self._log("DEBUG", f"Cleared {cleared} STT messages")

    # ──────────────── Speech Output ────────────────

    def _speak(self, text: str) -> bool:
        try:
            if not self.speech_port:
                return False
            b = yarp.Bottle()
            b.clear()
            b.addString(text)
            self.speech_port.write(b)
            return True
        except Exception as e:
            self._log("ERROR", f"Speak failed: {e}")
            return False

    def _speak_and_wait(self, text: str) -> bool:
        """Speak and wait estimated TTS duration, then clear STT buffer."""
        ok = self._speak(text)
        words = len(text.split())
        wait = words / TTS_WPS + TTS_END_MARGIN
        wait = max(TTS_MIN_WAIT, min(TTS_MAX_WAIT, wait))
        self._log("DEBUG", f"TTS wait: {words} words → {wait:.2f}s")
        time.sleep(wait)
        self._clear_stt_buffer()
        return ok

    # ──────────────── Greeting Detection ────────────────

    @staticmethod
    def _detect_greeting(utterance: str) -> Dict:
        """Fast keyword-based greeting detection."""
        lower = utterance.lower()
        for kw in GREETING_KEYWORDS:
            if kw in lower:
                return {"responded": True, "confidence": 1.0}
        # Any non-empty text is treated as some kind of response
        if lower.strip():
            return {"responded": True, "confidence": 0.5}
        return {"responded": False, "confidence": 0.0}

    # ──────────────── YARP RPC Clients ────────────────

    def _get_interaction_rpc(self) -> yarp.RpcClient:
        if self._interaction_rpc is None:
            client = yarp.RpcClient()
            port_name = f"/{self.module_name}/interactionInterface/rpc"
            if not client.open(port_name):
                raise RuntimeError(f"Failed to open {port_name}")
            if not client.addOutput("/interactionInterface"):
                client.close()
                raise RuntimeError("Failed to connect to /interactionInterface")
            self._interaction_rpc = client
            self._log("INFO", f"RPC connected: {port_name} → /interactionInterface")
        return self._interaction_rpc

    def _get_vision_rpc(self) -> yarp.RpcClient:
        if self._vision_rpc is None:
            client = yarp.RpcClient()
            port_name = f"/{self.module_name}/objectRecognition/rpc"
            if not client.open(port_name):
                raise RuntimeError(f"Failed to open {port_name}")
            if not client.addOutput("/objectRecognition"):
                client.close()
                raise RuntimeError("Failed to connect to /objectRecognition")
            self._vision_rpc = client
            self._log("INFO", f"RPC connected: {port_name} → /objectRecognition")
        return self._vision_rpc

    def _execute_behaviour(self, behaviour: str) -> bool:
        """Send 'exe <behaviour>' to /interactionInterface (fire-and-forget)."""
        try:
            rpc = self._get_interaction_rpc()
            cmd = yarp.Bottle()
            cmd.addString("exe")
            cmd.addString(behaviour)
            rpc.write(cmd)
            self._log("INFO", f"Behaviour '{behaviour}' sent")
            return True
        except Exception as e:
            self._log("ERROR", f"Behaviour failed: {e}")
            if self._interaction_rpc:
                self._interaction_rpc.close()
                self._interaction_rpc = None
            return False

    def _submit_face_name(self, track_id: int, name: str) -> bool:
        """Submit name to objectRecognition: name <name> id <track_id>."""
        try:
            rpc = self._get_vision_rpc()
            cmd = yarp.Bottle()
            cmd.addString("name")
            cmd.addString(name)
            cmd.addString("id")
            cmd.addInt32(track_id)

            reply = yarp.Bottle()
            rpc.write(cmd, reply)
            response = reply.toString()
            self._log("INFO", f"Face name '{name}' submitted for track {track_id}: {response}")
            return "ok" in response.lower()
        except Exception as e:
            self._log("ERROR", f"Face name submission failed: {e}")
            if self._vision_rpc:
                self._vision_rpc.close()
                self._vision_rpc = None
            return False

    # ──────────────── JSON Persistence ────────────────

    def _write_last_greeted(self, track_id: int, face_id: str, code: str):
        try:
            entries = _load_json_safe("last_greeted.json", [])
            if not isinstance(entries, list):
                entries = []
            entries.append({
                "timestamp": datetime.now(TIMEZONE).isoformat(),
                "track_id": track_id,
                "face_id": face_id,
                "assigned_code_or_name": code,
            })
            _save_json_atomic("last_greeted.json", entries)
        except Exception as e:
            self._log("ERROR", f"Write last_greeted failed: {e}")

    def _update_last_greeted_name(self, track_id: int, new_name: str):
        try:
            entries = _load_json_safe("last_greeted.json", [])
            if not isinstance(entries, list):
                return
            for entry in reversed(entries):
                if entry.get("track_id") == track_id:
                    entry["assigned_code_or_name"] = new_name
                    entry["name_updated"] = datetime.now(TIMEZONE).isoformat()
                    break
            _save_json_atomic("last_greeted.json", entries)
        except Exception as e:
            self._log("ERROR", f"Update last_greeted failed: {e}")

    # ──────────────── LLM Integration ────────────────

    def _init_ollama_background(self):
        """Ensure Ollama is ready (runs in background thread at startup)."""
        try:
            self.ensure_ollama_and_model()
        except Exception as e:
            self._log("WARNING", f"Ollama init failed: {e}")

    def _check_ollama_binary(self) -> bool:
        for path in ["/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/ollama/bin/ollama"]:
            if os.path.exists(path):
                return True
        return False

    def _start_ollama_server(self) -> bool:
        try:
            try:
                req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass

            self._log("INFO", "Starting Ollama server...")
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=open("/tmp/ollama_server.log", "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            for i in range(30):
                time.sleep(1)
                try:
                    req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        if resp.status == 200:
                            self._log("INFO", f"Ollama started ({i + 1}s)")
                            return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    def ensure_ollama_and_model(self) -> bool:
        try:
            if not self._check_ollama_binary():
                self._log("WARNING", "Ollama not found, attempting install...")
                result = subprocess.run(
                    "curl -fsSL https://ollama.com/install.sh | sh",
                    shell=True, capture_output=True, text=True, timeout=300,
                )
                if result.returncode != 0:
                    self._log("ERROR", "Ollama install failed")
                    return False

            if not self._start_ollama_server():
                self._log("ERROR", "Could not start Ollama")
                return False

            req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                if not any(LLM_MODEL in m for m in models):
                    self._log("INFO", f"Pulling model {LLM_MODEL}...")
                    subprocess.run(
                        ["ollama", "pull", LLM_MODEL],
                        capture_output=True, text=True, timeout=600,
                    )

            self.ollama_last_check = time.time()
            self._log("INFO", "Ollama ready")
            return True
        except Exception as e:
            self._log("ERROR", f"Ollama setup failed: {e}")
            return False

    def _check_ollama_health(self) -> bool:
        try:
            req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _llm_request(self, prompt: str, json_format: bool = False) -> str:
        """Send prompt to LLM with retry logic."""
        last_error = None
        for attempt in range(LLM_RETRY_ATTEMPTS):
            try:
                if time.time() - self.ollama_last_check > self.ollama_check_interval:
                    self._check_ollama_health()
                    self.ollama_last_check = time.time()

                payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
                if json_format:
                    payload["format"] = "json"

                data = json.dumps(payload).encode()
                req = urllib.request.Request(
                    f"{OLLAMA_URL}/api/generate",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
                    result = json.loads(resp.read().decode()).get("response", "").strip()
                    if result:
                        return result
                    last_error = "Empty LLM response"
            except Exception as e:
                last_error = str(e)
                self._log("WARNING", f"LLM attempt {attempt + 1}/{LLM_RETRY_ATTEMPTS}: {e}")
                if attempt < LLM_RETRY_ATTEMPTS - 1:
                    time.sleep(LLM_RETRY_DELAY)

        self._log("ERROR", f"LLM failed after {LLM_RETRY_ATTEMPTS} attempts: {last_error}")
        return ""

    def _llm_json(self, prompt: str) -> Dict:
        """Get JSON from LLM with robust parsing."""
        text = self._llm_request(prompt, json_format=True)
        if not text:
            return {}

        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                result = json.loads(text[start:end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _llm_extract_name(self, utterance: str) -> Dict:
        prompt = (
            f'Instruction: Extract the person\'s name from this English text. '
            f'ONLY extract if clearly stated. Do NOT invent names. If no name is found, use null.\n'
            f'Text: "{utterance}"\n'
            f'Output ONLY a raw JSON object with this exact schema: '
            f'{{"answered": true/false, "name": "extracted_name" or null, "confidence": 0.0-1.0}}\nJSON:'
        )
        result = self._llm_json(prompt)
        return {
            "answered": result.get("answered", False) is True,
            "name": result.get("name") or None,
            "confidence": float(result.get("confidence", 0) or 0),
        }

    def _llm_generate_convo_starter(self) -> str:
        text = self._llm_request(
            'You are a friendly social robot continuing a conversation with someone you just greeted. '
            'Generate a natural conversation starter. DO NOT use greetings like "hello" or "hi". '
            'Ask about their day or start a light topic. Be brief, warm, under 15 words. '
            'Examples: "How\'s your day going?", "What brings you here today?" '
            'Output ONLY the sentence, no quotes.'
        )
        return text.strip("\"'").strip() if text and len(text) < 150 else "How are you doing these days?"

    def _generate_starter_background(self):
        try:
            self._log("INFO", "Background: Prefetching conversation starter...")
            starter = self._llm_generate_convo_starter()
            if starter:
                self._cached_starter = starter
                self._log("INFO", f"Background: Starter cached: '{starter}'")
        except Exception as e:
            self._log("WARNING", f"Background starter failed: {e}")

    def _llm_generate_followup(self, last_utterance: str, history: List[str]) -> str:
        history_text = "\n".join(f"- {u}" for u in history[-3:])
        text = self._llm_request(
            f'You are a friendly social robot having a natural conversation. '
            f'Respond naturally and casually. Be empathetic and engaging. '
            f'User just said: "{last_utterance}"\n'
            f'Recent conversation: {history_text}\n'
            f'Generate a brief response (1-2 sentences, under 25 words). '
            f'Include a follow-up question to keep the conversation going. '
            f'Output ONLY your response, no quotes.'
        )
        return text.strip("\"'").strip() if text and len(text) < 200 else "That's interesting, tell me more!"

    def _llm_generate_closing_ack(self, last_utterance: str) -> str:
        text = self._llm_request(
            f'You are a friendly social robot ending a conversation. '
            f'The person just said: "{last_utterance}"\n'
            f'Generate a short, warm acknowledgment. DO NOT ask questions. '
            f'Keep it under 10 words. '
            f'Output ONLY your acknowledgment, no quotes.'
        )
        return text.strip("\"'").strip() if text and len(text) < 100 else "That's great, nice talking with you!"

    # ──────────────── Logging ────────────────

    def _log(self, level: str, message: str):
        ts = datetime.now(TIMEZONE).isoformat()
        print(f"[{ts}] [{level}] {message}")
        self.log_buffer.append({"timestamp": ts, "level": level, "message": message})


# ═══════════════════════ Main ═══════════════════════

if __name__ == "__main__":
    import sys

    yarp.Network.init()

    if not yarp.Network.checkNetwork():
        print("ERROR: YARP network not available")
        sys.exit(1)

    module = InteractionManagerModule()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.configure(sys.argv)

    print("=" * 60)
    print("InteractionManagerModule - 3 Trees (ss1/ss2/ss3)")
    print("=" * 60)
    print()
    print("YARP Connections:")
    print("  yarp connect /alwayson/vision/landmarks:o /interactionManager/landmarks:i")
    print("  yarp connect /speech2text/text:o /interactionManager/stt:i")
    print("  yarp connect /interactionManager/speech:o /acapelaSpeak/speech:i")
    print()
    print("RPC commands:")
    print("  echo 'run <track_id> <face_id> <ss1|ss2|ss3>' | yarp rpc /interactionManager")
    print("  echo 'status' | yarp rpc /interactionManager")
    print("  echo 'help' | yarp rpc /interactionManager")
    print("  echo 'quit' | yarp rpc /interactionManager")
    print()

    try:
        module.runModule(rf)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        module.interruptModule()
        module.close()
        yarp.Network.fini()
