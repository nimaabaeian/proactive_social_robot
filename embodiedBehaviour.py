"""
Embodied Behaviour Module
"""
import yarp
import time
import json
import csv
import os
import random
import threading
import signal
import sys
from collections import deque


class EmbodiedBehaviour(yarp.RFModule):

    # Configuration Constants
    QTABLE_PATH = "learning_qtable.json"
    PROACTIVE_CSV = "proactive_log.csv"
    SELFADAPTOR_CSV = "selfadaptor_log.csv"
    
    PROACTIVE_ACTIONS = ["ao_greet", "ao_coffee_break", "ao_curious_lean_in"]
    SELF_ADAPTORS = ["ao_look_around", "ao_yawn", "ao_cough"]
    
    THRESH_MEAN = 0.5
    THRESH_VAR = 0.1
    WAIT_AFTER_ACTION = 3.5
    COOLDOWN = 2.0
    SELFADAPTOR_PERIOD_CALM = 240.0
    SELFADAPTOR_PERIOD_LIVELY = 120.0
    
    EPSILON = 0.8
    EPSILON_MIN = 0.2
    EPSILON_DECAY = 0.957603
    
    NO_FACES_TIMEOUT = 120.0
    WINDOW_SIZE = 60  # Rolling window for state averaging (3s at 20Hz)
    
    def __init__(self):
        super().__init__()
        
        # Resolve paths
        base_dir = os.path.dirname(__file__)
        self.QTABLE_PATH = os.path.join(base_dir, self.QTABLE_PATH)
        self.PROACTIVE_CSV = os.path.join(base_dir, self.PROACTIVE_CSV)
        self.SELFADAPTOR_CSV = os.path.join(base_dir, self.SELFADAPTOR_CSV)
        
        # Ports
        self.port_iie = yarp.BufferedPortBottle()
        self.port_context = yarp.BufferedPortBottle()
        self.port_info = yarp.BufferedPortBottle()
        self.port_learning = yarp.BufferedPortBottle()
        self.port_rpc = yarp.RpcClient()
        
        # Shared state (thread-safe)
        self._state_lock = threading.Lock()
        self.IIE_mean = 0.0
        self.IIE_var = 1.0
        self.ctx = -1
        self.num_faces = 0
        self.num_mutual_gaze = 0
        
        # Rolling window buffers for non-blocking state averaging
        self.iie_window = deque(maxlen=self.WINDOW_SIZE)
        
        # Q-table and epsilon
        self.Q = {}
        self.qtable_lock = threading.Lock()
        self.epsilon = self.EPSILON
        
        # Always-On state control
        self.alwayson_active = False
        self.alwayson_lock = threading.Lock()
        self.last_faces_seen_time = time.time()
        
        # Thread control
        self.running = False
        self.threads = []
    
    # ========================================================================
    # RFModule Interface
    # ========================================================================

    def configure(self, rf):
        """Initialize module and start all threads"""
        print("\n" + "="*70)
        print("EMBODIED BEHAVIOUR MODULE - Single Unit Architecture")
        print("="*70)
        
        # Open ports
        ports = [
            (self.port_iie, "/alwayson/embodiedbehaviour/iie:i"),
            (self.port_context, "/alwayson/embodiedbehaviour/context:i"),
            (self.port_info, "/alwayson/embodiedbehaviour/info:i"),
            (self.port_learning, "/alwayson/embodiedbehaviour/experiences:o"),
            (self.port_rpc, "/alwayson/embodiedbehaviour/rpc:o")
        ]
        
        for port, name in ports:
            if not port.open(name):
                print(f"[ERROR] Failed to open {name}")
                return False
        
        # Connect RPC port to interaction interface
        if not yarp.Network.connect("/alwayson/embodiedbehaviour/rpc:o", "/interactionInterface"):
            print("[WARNING] Could not connect to /interactionInterface, will retry dynamically")
        
        print("\n[Ports] All opened successfully")
        
        # Initialize CSV files
        self._init_csv_files()
        
        # Load Q-table
        self._load_qtable()
        
        # Start Always-On
        print("\n[Always-On] Starting...")
        self._execute_alwayson_command("ao_start")
        with self.alwayson_lock:
            self.alwayson_active = True
            self.last_faces_seen_time = time.time()
        print("[Always-On] ✓ Active")
        
        # Start threads
        self.running = True
        self.threads = [
            threading.Thread(target=self._iie_monitor_loop, daemon=True, name="IIE"),
            threading.Thread(target=self._context_monitor_loop, daemon=True, name="Context"),
            threading.Thread(target=self._info_monitor_loop, daemon=True, name="Info"),
            threading.Thread(target=self._alwayson_monitor_loop, daemon=True, name="Always-On Monitor"),
            threading.Thread(target=self._proactive_loop, daemon=True, name="Proactive"),
            threading.Thread(target=self._selfadaptor_loop, daemon=True, name="Self-Adaptor")
        ]
        
        for thread in self.threads:
            thread.start()
        
        print(f"\n[Threads] Started 6 monitoring and action threads")
        print(f"  • IIE: Monitors intention (mean, variance)")
        print(f"  • Context: Monitors calm/lively classification")
        print(f"  • Info: Monitors face count & mutual gaze")
        print(f"  • Always-On Monitor: Auto stop/start based on face presence")
        print(f"  • Proactive: Executes learned social actions")
        print(f"  • Self-Adaptor: Periodic self-regulation behaviors")
        print(f"\n[Config] Action Thresholds:")
        print(f"  • IIE Mean ≥ {self.THRESH_MEAN} (high intention)")
        print(f"  • IIE Variance < {self.THRESH_VAR} (stable/predictable)")
        print(f"  • Epsilon: {self.epsilon:.3f} (exploration rate)")
        print(f"\n[Actions] Available:")
        print(f"  • Proactive: {', '.join(self.PROACTIVE_ACTIONS)}")
        print(f"  • Self-Adaptors: {', '.join(self.SELF_ADAPTORS)}")
        print("="*70 + "\n")
        
        return True
    
    def getPeriod(self):
        return 10.0
    
    def updateModule(self):
        """Module heartbeat"""
        return True
    
    def interruptModule(self):
        """Stop all threads"""
        self.running = False
        for port in [self.port_iie, self.port_context, self.port_info, 
                     self.port_learning, self.port_rpc]:
            port.interrupt()
        return True
    
    def close(self):
        """Cleanup"""
        print("\n[Shutdown] Closing embodied behaviour module...")
        self.running = False
        
        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Close ports
        for port in [self.port_iie, self.port_context, self.port_info,
                     self.port_learning, self.port_rpc]:
            port.close()
        
        print("[Shutdown] Complete")
        return True
    
    # ========================================================================
    # State Management
    # ========================================================================
    
    def _get_state_snapshot(self):
        """Get atomic snapshot of shared state"""
        with self._state_lock:
            return {
                'IIE_mean': self.IIE_mean,
                'IIE_var': self.IIE_var,
                'ctx': self.ctx,
                'num_faces': self.num_faces,
                'num_mutual_gaze': self.num_mutual_gaze
            }

    def _update_iie(self, mean, var):
        with self._state_lock:
            self.IIE_mean = mean
            self.IIE_var = var

    def _update_context(self, ctx):
        with self._state_lock:
            self.ctx = ctx

    def _update_info(self, faces, mutual_gaze):
        with self._state_lock:
            self.num_faces = faces
            self.num_mutual_gaze = mutual_gaze
            
            # Update last faces seen time
            if faces > 0:
                with self.alwayson_lock:
                    self.last_faces_seen_time = time.time()
    
    def _is_alwayson_active(self):
        """Check if always-on is currently active"""
        with self.alwayson_lock:
            return self.alwayson_active
    
    def _windowed_snapshot(self):
        # Get current instantaneous state for non-IIE fields
        current = self._get_state_snapshot()
        
        # If window is empty or too small, return current state
        if len(self.iie_window) < 5:
            return current
        
        # Average IIE values from rolling window
        iie_means = [sample['mean'] for sample in self.iie_window]
        iie_vars = [sample['var'] for sample in self.iie_window]
        
        return {
            'IIE_mean': sum(iie_means) / len(iie_means),
            'IIE_var': sum(iie_vars) / len(iie_vars),
            'ctx': current['ctx'],
            'num_faces': current['num_faces'],
            'num_mutual_gaze': current['num_mutual_gaze']
        }
    
    def _execute_alwayson_command(self, command):
        """Execute always-on start/stop command via native YARP RPC"""
        try:
            # Ensure connection to /interactionInterface
            if not yarp.Network.isConnected("/alwayson/embodiedbehaviour/rpc:o", "/interactionInterface"):
                if yarp.Network.exists("/interactionInterface"):
                    yarp.Network.connect("/alwayson/embodiedbehaviour/rpc:o", "/interactionInterface")
                    time.sleep(0.1)  # Brief delay for connection to stabilize
            
            cmd_bottle = yarp.Bottle()
            reply_bottle = yarp.Bottle()
            
            cmd_bottle.clear()
            cmd_bottle.addString("exe")
            cmd_bottle.addString(command)
            
            self.port_rpc.write(cmd_bottle, reply_bottle)
            return True
        except Exception as e:
            print(f"[Always-On] ✗ Error executing '{command}': {e}")
            return False
    
    # ========================================================================
    # Monitor Threads
    # ========================================================================
    
    def _iie_monitor_loop(self):
        """Monitor IIE (Interaction Intention Estimation)"""
        print("[IIE Monitor] Started")
        
        while self.running:
            try:
                bottle = self.port_iie.read(False)
                if bottle and not bottle.isNull() and bottle.size() > 0:
                    # Parse face IIE distributions
                    face_data = []
                    for i in range(bottle.size()):
                        iie_dist = bottle.get(i).asList()
                        if iie_dist and iie_dist.size() >= 2:
                            iie_params = iie_dist.get(1).asList()
                            if iie_params and iie_params.size() >= 4:
                                face_data.append({
                                    'mean': iie_params.get(2).asFloat32(),
                                    'variance': iie_params.get(3).asFloat32()
                                })
                    
                    # Select most engaged person
                    if face_data:
                        best = max(face_data, key=lambda x: x['mean'])
                        old_mean = self.IIE_mean
                        
                        # Update instantaneous state
                        self._update_iie(best['mean'], best['variance'])
                        
                        # Append to rolling window for averaging
                        self.iie_window.append({'mean': best['mean'], 'var': best['variance']})
                        
                        if abs(best['mean'] - old_mean) > 0.1:
                            print(f"[IIE] {old_mean:.2f}→{best['mean']:.2f}")
                
                time.sleep(0.05)
            except Exception as e:
                print(f"[IIE Monitor] Error: {e}")
                time.sleep(1.0)
        
        print("[IIE Monitor] Stopped")

    def _context_monitor_loop(self):
        """Monitor context classification"""
        print("[Context Monitor] Started")
        
        while self.running:
            try:
                bottle = self.port_context.read(False)
                if bottle and not bottle.isNull() and bottle.size() >= 3:
                    label = bottle.get(2).asInt8()
                    old_ctx = self.ctx
                    self._update_context(label)
                    if old_ctx != label:
                        ctx_name = "Calm" if label == 0 else "Lively" if label == 1 else "Uncertain"
                        print(f"[CTX] {old_ctx}→{label} ({ctx_name})")
                
                time.sleep(0.1)
            except Exception as e:
                print(f"[Context Monitor] Error: {e}")
                time.sleep(1.0)
        
        print("[Context Monitor] Stopped")

    def _info_monitor_loop(self):
        """Monitor faces and mutual gaze"""
        print("[Info Monitor] Started")
        
        while self.running:
            try:
                bottle = self.port_info.read(False)
                if bottle and not bottle.isNull() and bottle.size() >= 3:
                    faces, mutual_gaze = 0, 0
                    
                    # Bottle format from STM:
                    # [0] = (Time t_vision t_audio t_delta)
                    # [1] = (timestamp Faces n People n Light val Motion val MutualGaze n)
                    # [2] = (timestamp Audio (right left))
                    
                    vision_list = bottle.get(1).asList()
                    
                    if vision_list and vision_list.size() > 1:
                        # Skip first element (timestamp), parse key-value pairs
                        i = 1
                        while i < vision_list.size() - 1:
                            key = vision_list.get(i).toString()
                            value = vision_list.get(i + 1)
                            
                            if key == "Faces":
                                faces = value.asInt16()
                            elif key == "MutualGaze":
                                mutual_gaze = value.asInt16()
                            
                            i += 2
                    
                    old_faces, old_gaze = self.num_faces, self.num_mutual_gaze
                    self._update_info(faces, mutual_gaze)
                    if old_faces != faces or old_gaze != mutual_gaze:
                        print(f"[INFO] Faces={faces}, Gaze={mutual_gaze}")
                
                time.sleep(0.5)
            except Exception as e:
                print(f"[Info Monitor] Error: {e}")
                time.sleep(1.0)
        
        print("[Info Monitor] Stopped")

    def _alwayson_monitor_loop(self):
        """Monitor face presence and auto-stop/start always-on"""
        print("[Always-On Monitor] Started")
        
        while self.running:
            try:
                snapshot = self._get_state_snapshot()
                current_time = time.time()
                
                with self.alwayson_lock:
                    time_since_faces = current_time - self.last_faces_seen_time
                    is_active = self.alwayson_active
                
                # Check if we should stop (no faces for 2 minutes)
                if is_active and snapshot['num_faces'] == 0 and time_since_faces >= self.NO_FACES_TIMEOUT:
                    print(f"\n[Always-On Monitor] ⏸ No faces for {self.NO_FACES_TIMEOUT:.0f}s, stopping...")
                    if self._execute_alwayson_command("ao_stop"):
                        with self.alwayson_lock:
                            self.alwayson_active = False
                        print("[Always-On Monitor] ✓ Stopped\n")
                
                # Check if we should start (faces detected while stopped)
                elif not is_active and snapshot['num_faces'] > 0:
                    print(f"\n[Always-On Monitor] ▶ Faces detected, starting...")
                    if self._execute_alwayson_command("ao_start"):
                        with self.alwayson_lock:
                            self.alwayson_active = True
                            self.last_faces_seen_time = current_time
                        print("[Always-On Monitor] ✓ Started\n")
                
                time.sleep(1.0)
            except Exception as e:
                print(f"[Always-On Monitor] Error: {e}")
                time.sleep(1.0)
        
        print("[Always-On Monitor] Stopped")

    # ========================================================================
    # Action Threads
    # ========================================================================
    
    def _proactive_loop(self):
        """Execute proactive actions with learning"""
        print("[Proactive Thread] Started")
        
        csv_file = None
        try:
            # Initialize CSV writer
            csv_file = open(self.PROACTIVE_CSV, 'a', newline='')
            csv_writer = csv.writer(csv_file)
            
            if os.path.getsize(self.PROACTIVE_CSV) == 0:
                csv_writer.writerow([
                    'timestamp', 'proactive_action',
                    'pre_IIE_mean', 'pre_IIE_var', 'pre_ctx', 'pre_num_faces', 'pre_num_mutual_gaze',
                    'post_IIE_mean', 'post_IIE_var', 'post_ctx', 'post_num_faces', 'post_num_mutual_gaze',
                    'q_value', 'epsilon'
                ])
            
            while self.running:
                try:
                    # Skip if always-on is stopped
                    if not self._is_alwayson_active():
                        time.sleep(1.0)
                        continue
                    
                    # Get windowed pre-state snapshot (instant, uses rolling buffer)
                    print(f"[Proactive] 📊 Capturing pre-state (instant from rolling window)...")
                    pre = self._windowed_snapshot()
                    
                    # Check person presence
                    if pre['num_faces'] == 0:
                        time.sleep(1.0)
                        continue
                    
                    if pre['num_mutual_gaze'] == 0:
                        print(f"[Proactive] ⏸ No mutual gaze (faces={pre['num_faces']})")
                        time.sleep(1.0)
                        continue
                    
                    # Check intention thresholds
                    if pre['IIE_mean'] < self.THRESH_MEAN:
                        print(f"[Proactive] ⏸ Low intention: μ={pre['IIE_mean']:.2f} < {self.THRESH_MEAN}")
                        time.sleep(2.0)
                        continue
                    
                    if pre['IIE_var'] >= self.THRESH_VAR:
                        print(f"[Proactive] ⏸ Unstable: σ²={pre['IIE_var']:.2f} ≥ {self.THRESH_VAR}")
                        time.sleep(2.0)
                        continue
                    
                    print(f"[Proactive] ✓ Thresholds met (averaged): μ={pre['IIE_mean']:.2f}, σ²={pre['IIE_var']:.2f}")
                    
                    # Load latest Q-table before action selection
                    self._load_qtable(verbose=False)
                    
                    # Select and execute action
                    timestamp = time.time()
                    state_key = f"CTX{pre['ctx']}"
                    action = self._select_action_epsilon_greedy(state_key)
                    
                    with self.qtable_lock:
                        q_value = self.Q.get(state_key, {}).get(action, 0.0)
                    
                    print(f"\n[ACT] {action} | CTX{pre['ctx']} IIE={pre['IIE_mean']:.2f} Q={q_value:.2f} ε={self.epsilon:.2f}")
                    
                    # Execute via native YARP RPC
                    try:
                        # Ensure connection to /interactionInterface
                        if not yarp.Network.isConnected("/alwayson/embodiedbehaviour/rpc:o", "/interactionInterface"):
                            print(f"[Proactive] Connecting to /interactionInterface...")
                            if not yarp.Network.connect("/alwayson/embodiedbehaviour/rpc:o", "/interactionInterface"):
                                print(f"[Proactive] ✗ Could not connect to /interactionInterface")
                                time.sleep(1.0)
                                continue
                        
                        cmd_bottle = yarp.Bottle()
                        reply_bottle = yarp.Bottle()
                        
                        cmd_bottle.clear()
                        cmd_bottle.addString("exe")
                        cmd_bottle.addString(action)
                        
                        self.port_rpc.write(cmd_bottle, reply_bottle)
                        print(f"[Proactive] ✓ Action sent")
                    except Exception as e:
                        print(f"[Proactive] ✗ Error: {e}")
                        continue
                    
                    # Wait for: action completion + human reaction + rolling window to refresh with new data
                    # This ensures post-state measurement doesn't contain pre-action data
                    print(f"[Proactive] ⏳ Waiting {self.WAIT_AFTER_ACTION}s for action + reaction + sensor integration...")
                    time.sleep(self.WAIT_AFTER_ACTION)
                    print(f"[Proactive] ✓ Wait completed")
                    
                    # Get windowed post-state snapshot (instant, uses rolling buffer)
                    print(f"[Proactive] 📊 Capturing post-state (instant from rolling window)...")
                    post = self._windowed_snapshot()
                    print(f"[Proactive] 📊 Post-state averaged: μ={post['IIE_mean']:.2f}, σ²={post['IIE_var']:.2f}, CTX={post['ctx']}")
                    
                    delta_iie = post['IIE_mean'] - pre['IIE_mean']
                    print(f"→ IIE {pre['IIE_mean']:.2f}→{post['IIE_mean']:.2f} ({delta_iie:+.2f}) CTX{pre['ctx']}→{post['ctx']}")
                    
                    # Send experience to learning module (13 fields)
                    bottle = self.port_learning.prepare()
                    bottle.clear()
                    bottle.addFloat64(timestamp)
                    bottle.addString(action)
                    bottle.addInt8(pre['ctx'])
                    bottle.addFloat64(pre['IIE_mean'])
                    bottle.addFloat64(pre['IIE_var'])
                    bottle.addInt8(post['ctx'])
                    bottle.addFloat64(post['IIE_mean'])
                    bottle.addFloat64(post['IIE_var'])
                    bottle.addFloat64(q_value)
                    bottle.addInt32(pre['num_faces'])
                    bottle.addInt32(pre['num_mutual_gaze'])
                    bottle.addInt32(post['num_faces'])
                    bottle.addInt32(post['num_mutual_gaze'])
                    self.port_learning.write()
                    print(f"[Proactive] 📤 Experience sent to learning module")
                    
                    # Log to CSV
                    csv_writer.writerow([
                        f"{timestamp:.2f}", action,
                        f"{pre['IIE_mean']:.4f}", f"{pre['IIE_var']:.4f}", pre['ctx'],
                        pre['num_faces'], pre['num_mutual_gaze'],
                        f"{post['IIE_mean']:.4f}", f"{post['IIE_var']:.4f}", post['ctx'],
                        post['num_faces'], post['num_mutual_gaze'],
                        f"{q_value:.4f}", f"{self.epsilon:.4f}"
                    ])
                    csv_file.flush()
                    
                    # Decay epsilon
                    old_eps = self.epsilon
                    self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)
                    if abs(old_eps - self.epsilon) > 0.001:
                        print(f"[Proactive] 📉 Epsilon decayed: {old_eps:.3f} → {self.epsilon:.3f}")
                    
                    print(f"[Proactive] 😴 Cooldown {self.COOLDOWN}s...\n")
                    time.sleep(self.COOLDOWN)
                    
                except Exception as e:
                    print(f"[Proactive Thread] Error: {e}")
                    time.sleep(1.0)
        except Exception as e:
            print(f"[Proactive Thread] Fatal error: {e}")
        finally:
            if csv_file:
                csv_file.close()
        
        print("[Proactive Thread] Stopped")

    def _selfadaptor_loop(self):
        print("[Self-Adaptor Thread] Started")
        
        csv_file = None
        try:
            # Initialize CSV writer
            csv_file = open(self.SELFADAPTOR_CSV, 'a', newline='')
            csv_writer = csv.writer(csv_file)
            
            if os.path.getsize(self.SELFADAPTOR_CSV) == 0:
                csv_writer.writerow([
                    'timestamp', 'self_adaptor_action',
                    'pre_IIE_mean', 'pre_IIE_var', 'pre_ctx',
                    'post_IIE_mean', 'post_IIE_var', 'post_ctx'
                ])
            
            while self.running:
                try:
                    # Context-dependent period (uncertain defaults to calm for conservative timing)
                    snapshot = self._get_state_snapshot()
                    if snapshot['ctx'] == 1:
                        period = self.SELFADAPTOR_PERIOD_LIVELY
                    else:  # ctx == 0 or ctx == -1 (uncertain)
                        period = self.SELFADAPTOR_PERIOD_CALM
                    
                    # Interruptible sleep for responsive shutdown
                    sleep_step = 0.5
                    elapsed = 0.0
                    while self.running and elapsed < period:
                        time.sleep(sleep_step)
                        elapsed += sleep_step
                    
                    if not self.running:
                        break
                    
                    # Execute self-adaptor
                    timestamp = time.time()
                    pre = self._get_state_snapshot()
                    action = random.choice(self.SELF_ADAPTORS)
                    ctx_name = "Calm" if pre['ctx'] == 0 else "Lively" if pre['ctx'] == 1 else "Uncertain"
                    
                    print(f"\n[Self-Adaptor] 🔄 {action}")
                    print(f"[Self-Adaptor] Pre-state: CTX={ctx_name}, μ={pre['IIE_mean']:.2f}")
                    
                    # Execute via native YARP RPC
                    try:
                        cmd_bottle = yarp.Bottle()
                        reply_bottle = yarp.Bottle()
                        
                        cmd_bottle.clear()
                        cmd_bottle.addString("exe")
                        cmd_bottle.addString(action)
                        
                        self.port_rpc.write(cmd_bottle, reply_bottle)
                        print(f"[Self-Adaptor] ✓ Action sent")
                    except Exception as e:
                        print(f"[Self-Adaptor] ✗ Error: {e}")
                        continue
                    
                    time.sleep(self.WAIT_AFTER_ACTION)
                    post = self._get_state_snapshot()
                    print(f"[Self-Adaptor] Post-state: CTX={post['ctx']}, μ={post['IIE_mean']:.2f}")
                    
                    # Log to CSV
                    csv_writer.writerow([
                        f"{timestamp:.2f}", action,
                        f"{pre['IIE_mean']:.4f}", f"{pre['IIE_var']:.4f}", pre['ctx'],
                        f"{post['IIE_mean']:.4f}", f"{post['IIE_var']:.4f}", post['ctx']
                    ])
                    csv_file.flush()
                    
                except Exception as e:
                    print(f"[Self-Adaptor Thread] Error: {e}")
                    time.sleep(1.0)
        except Exception as e:
            print(f"[Self-Adaptor Thread] Fatal error: {e}")
        finally:
            if csv_file:
                csv_file.close()
        
        print("[Self-Adaptor Thread] Stopped")

    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _select_action_epsilon_greedy(self, state_key):
        """Epsilon-greedy action selection"""
        with self.qtable_lock:
            q_values = self.Q.get(state_key, {})
            valid_q = {a: q for a, q in q_values.items() if a in self.PROACTIVE_ACTIONS}
        
        if random.random() < self.epsilon:
            return random.choice(self.PROACTIVE_ACTIONS)
        else:
            return max(valid_q, key=valid_q.get) if valid_q else random.choice(self.PROACTIVE_ACTIONS)
    
    def _load_qtable(self, verbose=True):
        """Load Q-table from JSON file"""
        if not os.path.exists(self.QTABLE_PATH):
            if verbose:
                print(f"[Q-Table] Not found, using empty")
            with self.qtable_lock:
                self.Q.clear()
            return
        
        try:
            with open(self.QTABLE_PATH, 'r') as f:
                data = json.load(f)
            
            new_Q = data.get("Q", {})
            with self.qtable_lock:
                self.Q.clear()
                self.Q.update(new_Q)
            
            if verbose:
                print(f"[Q-Table] Loaded {len(new_Q)} states")
        except Exception as e:
            if verbose:
                print(f"[Q-Table] Error: {e}")
            with self.qtable_lock:
                self.Q.clear()

    def _init_csv_files(self):
        """Initialize CSV log files"""
        for path in [self.PROACTIVE_CSV, self.SELFADAPTOR_CSV]:
            if not os.path.exists(path):
                open(path, 'w').close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    yarp.Network.init()
    
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.configure(sys.argv)
    
    module = EmbodiedBehaviour()
    
    def signal_handler(sig, frame):
        print("\n[Signal] Ctrl+C received, shutting down...")
        module.interruptModule()
        module.close()
        yarp.Network.fini()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    if not module.configure(rf):
        print("[FATAL] Configuration failed")
        sys.exit(1)
    
    try:
        module.runModule()
    except KeyboardInterrupt:
        pass
    finally:
        module.interruptModule()
        module.close()
        yarp.Network.fini()
