"""Embodied Behaviour YARP RFModule with BT and Q-learning."""

import yarp
import threading
import time
import random
import sqlite3
import subprocess
import traceback
import argparse
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import py_trees

from learning import load_q_table, select_branch, update_after_action, compute_reward, clamp_reward
from behaviour_trees import build_lp_tree, build_hp_tree, ModuleAPI

CTX_UNCERTAIN = -1
CTX_CALM = 0
CTX_LIVELY = 1

INACTIVE_TIMEOUT = 60.0
ACTION_WAIT_TIMEOUT = 5.0
CONTEXT_WAIT_TIMEOUT = 5.0
PERCEPTION_LOG_INTERVAL = 1.0
MIN_CLOSE_FACE_AREA = 15000.0
ALPHA_REWARD = 1.0
BETA_REWARD = 0.3
ETA_LEARNING = 0.1

ALLOWED_ACTIONS = {
    "answer phone", "carry/hold (an object)", "drink",
    "eat", "text on/look at a cellphone", "hand wave",
}


class EmbodiedBehaviourModule(yarp.RFModule):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.module_name = "embodiedBehaviour"
        self.q_file = "./q_table.json"
        self.db_file = "./last_greeted.db"
        self.data_collection_db = "./data_collection.db"
        self._rng_seed = seed

        self.context_port_name = "/embodiedBehaviour/context:i"
        self.va_port_name = "/embodiedBehaviour/valence_arousal:i"
        self.face_id_port_name = "/embodiedBehaviour/face_id:i"
        self.action_port_name = "/embodiedBehaviour/actions:i"

        self.context_port = None
        self.va_port = None
        self.face_id_port = None
        self.action_port = None

        self._lock = threading.Lock()
        self.faces_count = 0
        self.faces_all: List[Dict] = []
        self.known_faces: List[Dict] = []
        self.context_label = CTX_UNCERTAIN
        self.detected_actions = {}

        self._va_capturing = False
        self._va_samples: List[Tuple[float, float, float]] = []

        self._running = False
        self._context_thread = None
        self._va_thread = None
        self._face_id_thread = None
        self._action_thread = None

        self._is_active = False
        self._last_face_time = 0.0
        self._transition_done = False

        self.rng = random.Random(self._rng_seed)
        if self._rng_seed is not None:
            print(f"[{self.module_name}] RNG initialized with seed: {self._rng_seed}")
        self.bt_tree = None
        self.bt_branch = None
        self.bt_context = None
        self._current_action_id = None
        self._context_wait_until = 0.0
        self._last_perception_log = 0.0
        self._db_lock = threading.Lock()

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        self.context_port = yarp.BufferedPortBottle()
        self.va_port = yarp.BufferedPortBottle()
        self.face_id_port = yarp.BufferedPortBottle()
        self.action_port = yarp.BufferedPortBottle()

        if not self.context_port.open(self.context_port_name):
            print(f"[{self.module_name}] Failed to open context port")
            return False
        if not self.va_port.open(self.va_port_name):
            print(f"[{self.module_name}] Failed to open VA port")
            return False
        if not self.face_id_port.open(self.face_id_port_name):
            print(f"[{self.module_name}] Failed to open face ID port")
            return False
        if not self.action_port.open(self.action_port_name):
            print(f"[{self.module_name}] Failed to open action port")
            return False

        self._init_last_greeted_db()
        self._init_data_collection_db()

        self._running = True
        self._context_thread = threading.Thread(target=self._context_reader, daemon=True)
        self._va_thread = threading.Thread(target=self._va_reader, daemon=True)
        self._face_id_thread = threading.Thread(target=self._face_id_reader, daemon=True)
        self._action_thread = threading.Thread(target=self._action_reader, daemon=True)
        
        self._context_thread.start()
        self._va_thread.start()
        self._face_id_thread.start()
        self._action_thread.start()

        self._last_face_time = time.time()
        self._is_active = False

        print(f"[{self.module_name}] [OK] Configured successfully!")
        print(f"[{self.module_name}]    Q-table: {self.q_file}")
        print(f"[{self.module_name}]    DB: {self.db_file}")
        print(f"[{self.module_name}]    Data DB: {self.data_collection_db}")
        print(f"[{self.module_name}]    Close face threshold: {MIN_CLOSE_FACE_AREA}")
        print(f"[{self.module_name}]    Starting INACTIVE, will activate when faces detected")
        return True

    def interruptModule(self) -> bool:
        self._running = False
        for port in [self.context_port, self.va_port, self.face_id_port, self.action_port]:
            if port:
                port.interrupt()
        return True

    def close(self) -> bool:
        self._running = False
        for t in [self._context_thread, self._va_thread, self._face_id_thread, self._action_thread]:
            if t and t.is_alive():
                t.join(timeout=1.0)
        for port in [self.context_port, self.va_port, self.face_id_port, self.action_port]:
            if port:
                port.close()
        print(f"[{self.module_name}] Closed.")
        return True

    def getPeriod(self) -> float:
        return 0.1

    def updateModule(self) -> bool:
        if not hasattr(self, '_first_update_done'):
            self._first_update_done = True
            print(f"[{self.module_name}] Main loop started - module is now running!")
        
        now = time.time()
        
        with self._lock:
            faces = self.faces_count
        
        if faces >= 1:
            self._last_face_time = now
        
        if self._is_active:
            if faces == 0 and (now - self._last_face_time) >= INACTIVE_TIMEOUT:
                self._transition_to_inactive()
        else:
            if faces >= 1:
                self._transition_to_active()
        
        if now - self._last_perception_log >= PERCEPTION_LOG_INTERVAL:
            self._log_perception()
            self._last_perception_log = now
        
        if not self._is_active:
            return True
        
        self._run_tree_selector()
        return True

    def _transition_to_inactive(self):
        if not self._is_active:
            return
        print(f"[{self.module_name}] STATE: ACTIVE -> INACTIVE (no faces for {INACTIVE_TIMEOUT}s)")
        self._log_transition("ACTIVE", "INACTIVE")
        if self.bt_tree is not None:
            print(f"[{self.module_name}] BT: Aborting {self.bt_branch} tree due to state transition")
        self._abort_bt()
        print(f"[{self.module_name}] ACTION: Executing ao_stop")
        self._execute_ao_stop()
        self._is_active = False

    def _transition_to_active(self):
        if self._is_active:
            print(f"[{self.module_name}] WARNING: Already ACTIVE, skipping transition")
            return
        with self._lock:
            faces = self.faces_count
        print(f"[{self.module_name}] STATE: INACTIVE -> ACTIVE (faces={faces})")
        self._log_transition("INACTIVE", "ACTIVE")
        print(f"[{self.module_name}] ACTION: Executing ao_start")
        self._execute_ao_start()
        self._is_active = True
        self._context_wait_until = 0.0

    def _execute_ao_start(self):
        try:
            subprocess.Popen('echo "exe ao_start" | yarp rpc /interactionInterface',
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[{self.module_name}] ao_start error: {e}")

    def _execute_ao_stop(self):
        try:
            subprocess.Popen('echo "exe ao_stop" | yarp rpc /interactionInterface',
                           shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[{self.module_name}] ao_stop error: {e}")

    def _run_tree_selector(self):
        now = time.time()
        
        if self.bt_tree is not None:
            self._tick_bt()
            return
        
        with self._lock:
            ctx = self.context_label
        
        if ctx == CTX_UNCERTAIN:
            if now < self._context_wait_until:
                return
            print(f"[{self.module_name}] SELECTOR: Context uncertain, waiting {CONTEXT_WAIT_TIMEOUT}s for stable context")
            self._context_wait_until = now + CONTEXT_WAIT_TIMEOUT
            return
        
        self._context_wait_until = 0.0
        
        q = load_q_table(self.q_file)
        branch = select_branch(ctx, q, self.rng)
        
        if branch is None:
            return
        
        ctx_str = "calm" if ctx == CTX_CALM else "lively"
        eps = q.get('epsilon', 0.8)
        q_lp = q.get(ctx_str, {}).get('LP', 0.0)
        q_hp = q.get(ctx_str, {}).get('HP', 0.0)
        print(f"[{self.module_name}] SELECTOR: Context={ctx_str}, Q[LP]={q_lp:.3f}, Q[HP]={q_hp:.3f}, eps={eps:.2f} -> Selected {branch}")
        
        self._log_selection(ctx, branch, q)
        print(f"[{self.module_name}] BT: Starting {branch} tree")
        self._start_bt(branch, ctx)

    def _start_bt(self, branch: str, context: int):
        try:
            api = self._create_api()
            root = build_lp_tree(api) if branch == "LP" else build_hp_tree(api)
            self.bt_tree = py_trees.trees.BehaviourTree(root=root)
            self.bt_tree.setup(timeout=5.0)
            self.bt_branch = branch
            self.bt_context = context
        except Exception as e:
            print(f"[{self.module_name}] ERROR: Failed to start {branch} tree - {e}")
            traceback.print_exc()
            self.bt_tree = None

    def _tick_bt(self):
        if self.bt_tree is None:
            return
        try:
            self.bt_tree.tick()
            status = self.bt_tree.root.status
            if status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE):
                result = "SUCCESS" if status == py_trees.common.Status.SUCCESS else "FAILURE"
                print(f"[{self.module_name}] BT: {self.bt_branch} tree completed with {result}")
                self._clear_bt()
                print(f"[{self.module_name}] BT: Ready for next selection")
        except Exception as e:
            print(f"[{self.module_name}] ERROR: BT tick failed - {type(e).__name__}: {e}")
            print(f"[{self.module_name}] ERROR: Branch={self.bt_branch}, Context={self.bt_context}")
            traceback.print_exc()
            print(f"[{self.module_name}] BT: Clearing tree after error")
            self._clear_bt()

    def _abort_bt(self):
        if self.bt_tree is not None:
            print(f"[{self.module_name}] BT: Aborting {self.bt_branch} tree (state change)")
            try:
                self.bt_tree.shutdown()
                print(f"[{self.module_name}] BT: Abort completed")
            except Exception as e:
                print(f"[{self.module_name}] ERROR: BT shutdown failed - {e}")
            self._clear_bt()

    def _clear_bt(self):
        self.bt_tree = None
        self.bt_branch = None
        self.bt_context = None
        with self._lock:
            self._va_capturing = False
            self._va_samples = []

    def _create_api(self) -> ModuleAPI:
        module = self
        
        class APIImpl(ModuleAPI):
            def get_all_faces(self) -> List[Dict]:
                with module._lock:
                    return list(module.faces_all)
            
            def get_known_faces(self) -> List[Dict]:
                with module._lock:
                    return list(module.known_faces)
            
            def has_close_face(self, min_area: float = MIN_CLOSE_FACE_AREA) -> bool:
                with module._lock:
                    return any(f.get("area", 0) >= min_area for f in module.faces_all)
            
            def get_max_face_area(self) -> float:
                with module._lock:
                    if not module.faces_all:
                        return 0.0
                    return max(f.get("area", 0) for f in module.faces_all)
            
            def select_target_by_biggest_box(self) -> Optional[Tuple[str, List[float]]]:
                with module._lock:
                    faces = list(module.known_faces)
                if not faces:
                    return None
                faces.sort(key=lambda f: f.get("area", 0), reverse=True)
                target = faces[0]
                return (target.get("name", ""), target.get("box", []))
            
            def get_target_action(self, target_name: str, max_age: float = 5.0) -> Optional[str]:
                now = time.time()
                with module._lock:
                    target_index = None
                    for face in module.known_faces:
                        if face.get("name") == target_name:
                            target_index = face.get("index")
                            break
                    
                    if target_index is None:
                        return None
                    
                    if target_name in module.detected_actions:
                        action, prob, ts = module.detected_actions[target_name]
                        if action in ALLOWED_ACTIONS and prob > 0.5 and (now - ts) <= max_age:
                            return action
                    
                    person_id_str = str(target_index)
                    if person_id_str in module.detected_actions:
                        action, prob, ts = module.detected_actions[person_id_str]
                        if action in ALLOWED_ACTIONS and prob > 0.5 and (now - ts) <= max_age:
                            return action
                return None
            
            def clear_detected_actions(self) -> None:
                with module._lock:
                    module.detected_actions = {}
            
            def start_va_capture(self) -> None:
                with module._lock:
                    module._va_capturing = True
                    module._va_samples = []
            
            def stop_va_capture_get_peak(self) -> Tuple[Optional[float], Optional[float]]:
                with module._lock:
                    module._va_capturing = False
                    samples = list(module._va_samples)
                    module._va_samples = []
                if not samples:
                    return (None, None)
                best = max(samples, key=lambda s: s[2])
                return (best[0], best[1])
            
            def update_learning(self, reward: float) -> None:
                ctx = module.bt_context
                branch = module.bt_branch
                if ctx is not None and branch is not None:
                    update_after_action(module.q_file, ctx, branch, reward, ETA_LEARNING)
            
            def get_context(self) -> int:
                with module._lock:
                    return module.context_label
            
            def get_branch(self) -> str:
                return module.bt_branch or ""
            
            def start_action_log(self, action_type: str, response_type: str, response_value: str,
                                trigger_reason: str, target_name: str, last_seen_today: bool) -> int:
                tree_branch = module.bt_branch if module.bt_branch else ""
                return module._start_action_log(action_type, response_type, response_value,
                                               trigger_reason, target_name, last_seen_today, tree_branch)
            
            def end_action_log(self, action_id: int, success: bool, note: str = "") -> None:
                module._end_action_log(action_id, success, note)
            
            def log_affect_summary(self, action_id: int, ts_start: float, samples: List,
                                  reward: float, used_for_learning: bool) -> None:
                module._log_affect_summary(action_id, ts_start, samples, reward, used_for_learning)
            
            def was_greeted_today(self, name: str) -> bool:
                return module._was_greeted_today(name)
            
            def mark_greeted_today(self, name: str) -> None:
                module._mark_greeted_today(name)
        
        return APIImpl()

    def _init_last_greeted_db(self):
        try:
            print(f"[{self.module_name}] Initializing last_greeted.db...")
            with self._db_lock:
                conn = sqlite3.connect(self.db_file)
                try:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS last_greeted (
                            name TEXT PRIMARY KEY,
                            timestamp REAL NOT NULL,
                            date TEXT NOT NULL
                        )
                    """)
                    conn.commit()
                    print(f"[{self.module_name}] [OK] Initialized last_greeted.db")
                finally:
                    conn.close()
        except Exception as e:
            print(f"[{self.module_name}] [ERROR] initializing last_greeted.db: {e}")
            raise

    def _init_data_collection_db(self):
        try:
            print(f"[{self.module_name}] Initializing data_collection.db...")
            with self._db_lock:
                conn = sqlite3.connect(self.data_collection_db)
                try:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS events (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            ts REAL, event_type TEXT, from_state TEXT, to_state TEXT
                        )
                    """)
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS actions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            ts_start REAL, ts_end REAL, action_type TEXT, response_type TEXT,
                            response_value TEXT, trigger_reason TEXT, target_name TEXT,
                            last_seen_today INTEGER, tree_branch TEXT, success INTEGER, note TEXT
                        )
                    """)
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS affect (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            action_id INTEGER, ts_start REAL, peak_valence REAL, peak_arousal REAL,
                            reward REAL, used_for_learning INTEGER,
                            FOREIGN KEY (action_id) REFERENCES actions(id)
                        )
                    """)
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS perception (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            ts REAL, faces_count INTEGER, context_label INTEGER
                        )
                    """)
                    conn.commit()
                    print(f"[{self.module_name}] [OK] Initialized data_collection.db")
                finally:
                    conn.close()
        except Exception as e:
            print(f"[{self.module_name}] [ERROR] initializing data_collection.db: {e}")
            raise

    def _log_transition(self, from_state: str, to_state: str):
        try:
            with self._db_lock:
                conn = sqlite3.connect(self.data_collection_db)
                try:
                    conn.execute("INSERT INTO events (ts, event_type, from_state, to_state) VALUES (?, ?, ?, ?)",
                                (time.time(), "transition", from_state, to_state))
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            print(f"[{self.module_name}] Log transition error: {e}")

    def _log_selection(self, context: int, branch: str, q: dict):
        try:
            with self._db_lock:
                conn = sqlite3.connect(self.data_collection_db)
                try:
                    ctx_str = "calm" if context == 0 else "lively"
                    conn.execute("INSERT INTO events (ts, event_type, from_state, to_state) VALUES (?, ?, ?, ?)",
                                (time.time(), "selection", ctx_str, branch))
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            print(f"[{self.module_name}] Log selection error: {e}")

    def _log_perception(self) -> None:
        try:
            with self._lock:
                faces = self.faces_count
                ctx = self.context_label
            with self._db_lock:
                conn = sqlite3.connect(self.data_collection_db)
                try:
                    conn.execute("INSERT INTO perception (ts, faces_count, context_label) VALUES (?, ?, ?)",
                                (time.time(), faces, ctx))
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            print(f"[{self.module_name}] Log perception error: {e}")

    def _start_action_log(self, action_type: str, response_type: str, response_value: str,
                          trigger_reason: str, target_name: str, last_seen_today: bool, tree_branch: str = "") -> int:
        try:
            with self._db_lock:
                conn = sqlite3.connect(self.data_collection_db)
                try:
                    cur = conn.execute(
                        """INSERT INTO actions (ts_start, action_type, response_type, response_value,
                           trigger_reason, target_name, last_seen_today, tree_branch) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (time.time(), action_type, response_type, response_value,
                         trigger_reason, target_name, 1 if last_seen_today else 0, tree_branch))
                    conn.commit()
                    return cur.lastrowid
                finally:
                    conn.close()
        except Exception as e:
            print(f"[{self.module_name}] Start action log error: {e}")
            return -1

    def _end_action_log(self, action_id: int, success: bool, note: str = ""):
        try:
            with self._db_lock:
                conn = sqlite3.connect(self.data_collection_db)
                try:
                    conn.execute("UPDATE actions SET ts_end = ?, success = ?, note = ? WHERE id = ?",
                                (time.time(), 1 if success else 0, note, action_id))
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            print(f"[{self.module_name}] End action log error: {e}")

    def _log_affect_summary(self, action_id: int, ts_start: float, samples: list,
                            reward: float, used_for_learning: bool):
        try:
            # Extract peak valence and arousal from samples (samples is list of (v, a) tuples)
            peak_valence = None
            peak_arousal = None
            if samples and len(samples) > 0:
                # samples is [(v, a), ...] - use the first/only tuple
                if len(samples[0]) >= 2:
                    peak_valence = samples[0][0]
                    peak_arousal = samples[0][1]
            
            with self._db_lock:
                conn = sqlite3.connect(self.data_collection_db)
                try:
                    conn.execute(
                        "INSERT INTO affect (action_id, ts_start, peak_valence, peak_arousal, reward, used_for_learning) VALUES (?, ?, ?, ?, ?, ?)",
                        (action_id, ts_start, peak_valence, peak_arousal, reward, 1 if used_for_learning else 0)
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            print(f"[{self.module_name}] Log affect error: {e}")

    def _was_greeted_today(self, name: str) -> bool:
        with self._db_lock:
            conn = sqlite3.connect(self.db_file)
            try:
                cur = conn.execute("SELECT timestamp FROM last_greeted WHERE name = ?", (name,))
                row = cur.fetchone()
                if row is None:
                    return False
                last_greeted_ts = row[0]
                last_greeted_date = datetime.fromtimestamp(last_greeted_ts).date()
                today = datetime.fromtimestamp(time.time()).date()
                return last_greeted_date == today
            finally:
                conn.close()

    def _mark_greeted_today(self, name: str):
        with self._db_lock:
            conn = sqlite3.connect(self.db_file)
            try:
                now = time.time()
                today_str = datetime.fromtimestamp(now).strftime('%Y-%m-%d')
                conn.execute("INSERT OR REPLACE INTO last_greeted (name, timestamp, date) VALUES (?, ?, ?)",
                            (name, now, today_str))
                conn.commit()
                print(f"[{self.module_name}] DATABASE: Marked '{name}' as greeted at {datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')}")
            finally:
                conn.close()

    def _context_reader(self):
        while self._running:
            bottle = self.context_port.read(True)
            if bottle is None:
                continue
            self._parse_context(bottle)

    def _parse_context(self, bottle: yarp.Bottle):
        if bottle.size() >= 3:
            label = bottle.get(2).asInt32()
            with self._lock:
                self.context_label = label

    def _va_reader(self):
        while self._running:
            bottle = self.va_port.read(True)
            if bottle is None:
                continue
            self._parse_va(bottle)

    def _parse_va(self, bottle: yarp.Bottle):
        best_v, best_a, best_reward = None, None, -999.0
        
        for i in range(bottle.size()):
            face_item = bottle.get(i)
            if not face_item.isList():
                continue
            face_list = face_item.asList()
            v, a, status_ok = self._extract_va(face_list)
            
            if status_ok and v is not None and a is not None:
                reward = compute_reward(v, a, ALPHA_REWARD, BETA_REWARD)
                if reward > best_reward:
                    best_reward = reward
                    best_v = v
                    best_a = a
        
        with self._lock:
            if self._va_capturing and best_v is not None and best_a is not None:
                self._va_samples.append((best_v, best_a, best_reward))

    def _extract_va(self, face_list: yarp.Bottle) -> Tuple[Optional[float], Optional[float], bool]:
        valence, arousal = None, None
        status_ok = False
        
        for j in range(face_list.size()):
            field = face_list.get(j)
            if not field.isList():
                continue
            fl = field.asList()
            if fl.size() < 2:
                continue
            key = fl.get(0).asString()
            if key == "valence":
                valence = fl.get(1).asFloat64()
            elif key == "arousal":
                arousal = fl.get(1).asFloat64()
            elif key == "status":
                status_ok = (fl.get(1).asString() == "ok")
        return valence, arousal, status_ok

    def _face_id_reader(self):
        while self._running:
            bottle = self.face_id_port.read(True)
            if bottle is None:
                continue
            self._parse_face_ids(bottle)

    def _parse_face_ids(self, bottle: yarp.Bottle):
        all_faces: List[Dict] = []
        known_faces: List[Dict] = []
        
        for i in range(bottle.size()):
            face_item = bottle.get(i)
            if not face_item.isList():
                continue
            face_list = face_item.asList()
            name, conf, box = self._extract_face_info(face_list)
            
            if box:
                area = abs((box[2] - box[0]) * (box[3] - box[1]))
                is_known = name not in ("unknown face", "recognizing", "")
                face_data = {
                    "name": name, "confidence": conf, "box": box,
                    "area": area, "index": i, "is_known": is_known
                }
                all_faces.append(face_data)
                if is_known:
                    known_faces.append(face_data)
        
        with self._lock:
            self.faces_count = len(all_faces)
            self.faces_all = all_faces
            self.known_faces = known_faces

    def _extract_face_info(self, face_list: yarp.Bottle) -> Tuple[str, float, Optional[List[float]]]:
        name = ""
        conf = 0.0
        box = None
        
        for j in range(face_list.size()):
            field = face_list.get(j)
            if not field.isList():
                continue
            fl = field.asList()
            if fl.size() < 2:
                continue
            key = fl.get(0).asString()
            if key == "class":
                name = fl.get(1).asString()
            elif key == "score":
                conf = fl.get(1).asFloat64()
            elif key == "box":
                box_data = fl.get(1)
                if box_data.isList():
                    bl = box_data.asList()
                    if bl.size() >= 4:
                        box = [bl.get(0).asFloat64(), bl.get(1).asFloat64(),
                               bl.get(2).asFloat64(), bl.get(3).asFloat64()]
                elif fl.size() >= 5:
                    box = [fl.get(1).asFloat64(), fl.get(2).asFloat64(),
                           fl.get(3).asFloat64(), fl.get(4).asFloat64()]
        return name, conf, box

    def _action_reader(self):
        while self._running:
            bottle = self.action_port.read(True)
            if bottle is None:
                continue
            self._parse_actions(bottle)

    def _parse_actions(self, bottle: yarp.Bottle):
        actions = {}
        ts = time.time()
        
        for i in range(bottle.size()):
            item = bottle.get(i)
            if not item.isList():
                continue
            lst = item.asList()
            if lst.size() < 2:
                continue
            key = lst.get(0).asString()
            if key == "stamp":
                ts = lst.get(1).asFloat64()
            elif key == "people":
                people_data = lst.get(1)
                if people_data.isList():
                    self._parse_people_actions(people_data.asList(), actions, ts)
        
        with self._lock:
            self.detected_actions = actions

    def _parse_people_actions(self, people_list: yarp.Bottle, actions: dict, ts: float):
        for i in range(people_list.size()):
            person_item = people_list.get(i)
            if not person_item.isList():
                continue
            person_list = person_item.asList()
            person_id = ""
            action = ""
            prob = 0.0
            
            for j in range(person_list.size()):
                field = person_list.get(j)
                if not field.isList():
                    continue
                fl = field.asList()
                if fl.size() < 2:
                    continue
                key = fl.get(0).asString()
                if key == "class":
                    person_id = fl.get(1).asString()
                elif key == "action":
                    action = fl.get(1).asString()
                elif key == "prob":
                    prob = fl.get(1).asFloat64()
            
            if person_id and action:
                actions[person_id] = (action, prob, ts)


def main():
    yarp.Network.init()
    if not yarp.Network.checkNetwork():
        print("YARP network not available")
        return 1
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    module = EmbodiedBehaviourModule()
    return module.runModule(rf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embodied Behaviour YARP Module")
    parser.add_argument("--q_file", default="q_table.json", help="Path to Q-table JSON file")
    parser.add_argument("--db_file", default="last_greeted.db", help="Path to last_greeted database")
    parser.add_argument("--data_db", default="data_collection.db", help="Path to data collection database")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible behavior (optional)")
    args = parser.parse_args()
    
    yarp.Network.init()
    module = EmbodiedBehaviourModule(seed=args.seed)
    module.q_file = args.q_file
    module.db_file = args.db_file
    module.data_collection_db = args.data_db
    
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    
    if module.configure(rf):
        print(f"[embodiedBehaviour] Module configured successfully")
        print(f"[embodiedBehaviour] Q-file: {module.q_file}")
        print(f"[embodiedBehaviour] Last greeted DB: {module.db_file}")
        print(f"[embodiedBehaviour] Data collection DB: {module.data_collection_db}")
        module.runModule()
    else:
        print("[embodiedBehaviour] Configuration failed")
    
    yarp.Network.fini()
