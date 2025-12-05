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
import subprocess
import pickle
import numpy as np


class EmbodiedBehaviour(yarp.RFModule):

    # Configuration Constants
    QTABLE_PATH = "learning_qtable.json"
    PROACTIVE_CSV = "proactive_log.csv"
    SELFADAPTOR_CSV = "selfadaptor_log.csv"
    
    PROACTIVE_ACTIONS = ["ao_greet", "ao_coffee_break", "ao_curious_lean_in"]
    SELF_ADAPTORS = ["ao_look_around", "ao_yawn", "ao_cough"]
    
    THRESH_MEAN = 0.5
    THRESH_VAR = 0.1
    WAIT_AFTER_ACTION = 3.0
    
    EPSILON = 0.8
    EPSILON_MIN = 0.2
    EPSILON_DECAY = 0.957603
    
    NO_FACES_TIMEOUT = 120.0
    
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
        
        # Shared state (thread-safe)
        self._state_lock = threading.Lock()
        self.IIE_mean = 0.0
        self.IIE_var = 1.0
        self.ctx = -1
        self.num_faces = 0
        self.num_mutual_gaze = 0
        
        # Q-table and epsilon
        self.Q = {}
        self.qtable_lock = threading.Lock()
        self.epsilon = self.EPSILON
        
        # Always-On state control
        self.alwayson_active = False
        self.alwayson_lock = threading.Lock()
        self.last_faces_seen_time = time.time()
        
        # Proactive execution state (for self-adaptor priority)
        self.proactive_active = False
        self.proactive_lock = threading.Lock()
        
        # Gatekeeper timing (Phase 3)
        # self.last_action_time = time.time()  # Uncomment for Phase 3: Track time between actions
        
        # Thread control
        self.running = False
        self.threads = []
    
    # ========================================================================
    # RFModule Interface
    # ========================================================================

    def configure(self, rf):
        """Initialize module and start all threads"""
        print("\n" + "="*70)
        print("[Actor] 🤖 EMBODIED BEHAVIOUR MODULE")
        print("="*70)
        
        # Open ports
        ports = [
            (self.port_iie, "/alwayson/embodiedbehaviour/iie:i"),
            (self.port_context, "/alwayson/embodiedbehaviour/context:i"),
            (self.port_info, "/alwayson/embodiedbehaviour/info:i"),
            (self.port_learning, "/alwayson/embodiedbehaviour/experiences:o")
        ]
        
        for port, name in ports:
            if not port.open(name):
                print(f"[Actor] ❌ Port failed: {name}")
                return False
        
        print("[Actor] ✅ Ports ready")
        
        # Initialize CSV files
        self._init_csv_files()
        
        # Load Q-table
        self._load_qtable()
        
        # Start Always-On
        self._execute_alwayson_command("ao_start")
        with self.alwayson_lock:
            self.alwayson_active = True
            self.last_faces_seen_time = time.time()
        print("[Actor] ✅ Always-On active")
        
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
        
        print(f"[Actor] ✅ 6 threads started")
        print(f"[Actor] 📊 Thresholds: μ≥{self.THRESH_MEAN}, σ²<{self.THRESH_VAR}, ε={self.epsilon:.2f}")
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
                     self.port_learning]:
            port.interrupt()
        return True
    
    def close(self):
        """Cleanup"""
        print("\n[Actor] 🛑 Shutting down...")
        self.running = False
        
        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Close ports
        for port in [self.port_iie, self.port_context, self.port_info,
                     self.port_learning]:
            port.close()
        
        print("[Actor] ✅ Shutdown complete")
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
    
    def _is_proactive_executing(self):
        """Check if proactive thread is currently executing an action"""
        with self.proactive_lock:
            return self.proactive_active
    
    def _set_proactive_active(self, active):
        """Set proactive execution state"""
        with self.proactive_lock:
            self.proactive_active = active
    
    def _windowed_snapshot(self, duration=1.0, step=0.1):
        """Get averaged state snapshot over a time window to reduce noise
        """
        samples = []
        t0 = time.time()
        while time.time() - t0 < duration and self.running:
            samples.append(self._get_state_snapshot())
            time.sleep(step)
        
        # Fallback to single snapshot if no samples collected
        if not samples:
            return self._get_state_snapshot()
        
        # Average numeric fields
        return {
            'IIE_mean': sum(s['IIE_mean'] for s in samples) / len(samples),
            'IIE_var': sum(s['IIE_var'] for s in samples) / len(samples),
            'ctx': max(set(s['ctx'] for s in samples), key=[s['ctx'] for s in samples].count),
            'num_faces': round(sum(s['num_faces'] for s in samples) / len(samples)),
            'num_mutual_gaze': round(sum(s['num_mutual_gaze'] for s in samples) / len(samples)),
        }
    
    def _execute_alwayson_command(self, command):
        """Execute always-on start/stop command"""
        try:
            result = subprocess.run(
                f'echo "exe {command}" | yarp rpc /interactionInterface',
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True
            else:
                print(f"[Always-On] ✗ Command '{command}' failed: {result.stderr.strip()}")
                return False
        except Exception as e:
            print(f"[Always-On] ✗ Error executing '{command}': {e}")
            return False
    
    # ========================================================================
    # Monitor Threads
    # ========================================================================
    
    def _iie_monitor_loop(self):
        """Monitor IIE (Interaction Intention Estimation)"""
        print("[Actor/IIE] ▶️ Monitor started")
        
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
                        self._update_iie(best['mean'], best['variance'])
                        if abs(best['mean'] - old_mean) > 0.1:
                            print(f"[Actor/IIE] 📊 Updated from {len(face_data)} face(s): μ={best['mean']:.2f}, σ²={best['variance']:.2f} (Δ{best['mean']-old_mean:+.2f})")
                
                time.sleep(0.05)
            except Exception as e:
                print(f"[Actor/IIE] ❌ {e}")
                time.sleep(1.0)
        
        print("[Actor/IIE] ⏹️ Stopped")

    def _context_monitor_loop(self):
        """Monitor context classification"""
        print("[Actor/CTX] ▶️ Monitor started")
        
        while self.running:
            try:
                bottle = self.port_context.read(False)
                if bottle and not bottle.isNull() and bottle.size() >= 3:
                    label = bottle.get(2).asInt8()
                    old_ctx = self.ctx
                    self._update_context(label)
                    if old_ctx != label:
                        ctx_name = "🔵Calm" if label == 0 else "🔴Lively" if label == 1 else "⚪Uncertain"
                        print(f"[Actor/CTX] {ctx_name}")
                
                time.sleep(0.1)
            except Exception as e:
                print(f"[Actor/CTX] ❌ {e}")
                time.sleep(1.0)
        
        print("[Actor/CTX] ⏹️ Stopped")

    def _info_monitor_loop(self):
        """Monitor faces and mutual gaze"""
        print("[Actor/INFO] ▶️ Monitor started")
        
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
                        print(f"[Actor/INFO] 👤{faces} 👁️{mutual_gaze}")
                
                time.sleep(0.5)
            except Exception as e:
                print(f"[Actor/INFO] ❌ {e}")
                time.sleep(1.0)
        
        print("[Actor/INFO] ⏹️ Stopped")

    def _alwayson_monitor_loop(self):
        """Monitor face presence and auto-stop/start always-on"""
        print("[Actor/AO] ▶️ Monitor started")
        
        while self.running:
            try:
                snapshot = self._get_state_snapshot()
                current_time = time.time()
                
                with self.alwayson_lock:
                    time_since_faces = current_time - self.last_faces_seen_time
                    is_active = self.alwayson_active
                
                # Check if we should stop (no faces for 2 minutes)
                if is_active and snapshot['num_faces'] == 0 and time_since_faces >= self.NO_FACES_TIMEOUT:
                    print(f"[Actor/AO] ⏸️ No faces for {time_since_faces:.0f}s (timeout={self.NO_FACES_TIMEOUT:.0f}s) → executing ao_stop")
                    if self._execute_alwayson_command("ao_stop"):
                        with self.alwayson_lock:
                            self.alwayson_active = False
                        print("[Actor/AO] ✅ Always-on stopped successfully")                # Check if we should start (faces detected while stopped)
                elif not is_active and snapshot['num_faces'] > 0:
                    print(f"[Actor/AO] ▶️ Faces detected ({snapshot['num_faces']}) while stopped → executing ao_start")
                    if self._execute_alwayson_command("ao_start"):
                        with self.alwayson_lock:
                            self.alwayson_active = True
                            self.last_faces_seen_time = current_time
                        print("[Actor/AO] ✅ Always-on started successfully")
                
                time.sleep(1.0)
            except Exception as e:
                print(f"[Actor/AO] ❌ {e}")
                time.sleep(1.0)
        
        print("[Actor/AO] ⏹️ Stopped")

    # ========================================================================
    # Action Threads
    # ========================================================================
    
    def _proactive_loop(self):
        """Execute proactive actions with learning"""
        print("[Actor/PRO] ▶️ Proactive thread started")
        
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
                    
                    # Get windowed pre-state snapshot (averaged over 3 seconds)
                    pre = self._windowed_snapshot(duration=3.0, step=0.1)
                    
                    # Check person presence
                    if pre['num_faces'] == 0:
                        time.sleep(1.0)
                        continue
                    
                    if pre['num_mutual_gaze'] == 0:
                        print(f"[Actor/PRO] ⏸️ Waiting for gaze: faces={pre['num_faces']}, gaze=0 (need >0)")
                        time.sleep(1.0)
                        continue
                    
                    # Check intention thresholds
                    if pre['IIE_mean'] < self.THRESH_MEAN:
                        print(f"[Actor/PRO] ⏸️ IIE too low: μ={pre['IIE_mean']:.2f} < threshold={self.THRESH_MEAN}")
                        time.sleep(2.0)
                        continue
                    
                    if pre['IIE_var'] >= self.THRESH_VAR:
                        print(f"[Actor/PRO] ⏸️ IIE unstable: σ²={pre['IIE_var']:.2f} ≥ threshold={self.THRESH_VAR}")
                        time.sleep(2.0)
                        continue
                    
                    print(f"[Actor/PRO] ✅ All checks passed: μ={pre['IIE_mean']:.2f}, σ²={pre['IIE_var']:.2f}, faces={pre['num_faces']}, gaze={pre['num_mutual_gaze']}")
                    
                    # ============================================================
                    # TODO: GATEKEEPER INTEGRATION (PHASE 3+)
                    # ============================================================
                    # Uncomment this block after 200+ training samples collected
                    # Requires: pickle, numpy imports (see top of file)
                    # Requires: self.last_action_time tracking (see __init__)
                    # ============================================================
                    #
                    # # Calculate time since last action
                    # current_time = time.time()
                    # time_delta = current_time - getattr(self, 'last_action_time', current_time - 10.0)
                    # time_delta = min(time_delta, 60.0)  # Clamp to 60s max (prevent outliers after long idle)
                    # 
                    # # Query disk-based gatekeeper model
                    # should_act = self._check_gatekeeper(pre, time_delta)
                    # 
                    # if not should_act:
                    #     print(f"[Actor/PRO] 🚫 Scene not opportune ({time_delta:.0f}s ago)")
                    #     time.sleep(2.0)
                    #     continue
                    # 
                    # print(f"[Actor/PRO] ✅ Scene opportune")
                    # 
                    # # Update timestamp for next action
                    # self.last_action_time = time.time()
                    # ============================================================
                    
                    # Load latest Q-table before action selection
                    self._load_qtable(verbose=False)
                    
                    # Select and execute action
                    timestamp = time.time()
                    state_key = f"CTX{pre['ctx']}"
                    action = self._select_action_epsilon_greedy(state_key)
                    
                    # Signal self-adaptor that proactive action is starting (AFTER action selection)
                    self._set_proactive_active(True)
                    
                    with self.qtable_lock:
                        q_value = self.Q.get(state_key, {}).get(action, 0.0)
                    
                    print(f"[Actor/PRO] ⚡ Executing: {action} | CTX{pre['ctx']} | Q={q_value:.2f} | ε={self.epsilon:.2f}")
                    
                    # Execute via RPC shell command
                    try:
                        result = subprocess.run(
                            f'echo "exe {action}" | yarp rpc /interactionInterface',
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode != 0:
                            print(f"[Actor/PRO] ❌ RPC failed: {result.stderr.strip()}")
                            self._set_proactive_active(False)
                            continue
                    except subprocess.TimeoutExpired:
                        print(f"[Actor/PRO] ❌ RPC timeout (>5s)")
                        self._set_proactive_active(False)
                        continue
                    except Exception as e:
                        print(f"[Actor/PRO] ❌ RPC error: {e}")
                        self._set_proactive_active(False)
                        continue
                    
                    # Wait for effect
                    print(f"[Actor/PRO] ⏳ Waiting {self.WAIT_AFTER_ACTION:.1f}s for action effect...")
                    time.sleep(self.WAIT_AFTER_ACTION)
                    
                    # Get windowed post-state snapshot (averaged over 3 seconds)
                    print(f"[Actor/PRO] 📸 Collecting post-state (3s window)...")
                    post = self._windowed_snapshot(duration=3.0, step=0.1)
                    
                    delta_iie = post['IIE_mean'] - pre['IIE_mean']
                    delta_var = post['IIE_var'] - pre['IIE_var']
                    print(f"[Actor/PRO] 📊 Outcome: μ {pre['IIE_mean']:.2f}→{post['IIE_mean']:.2f} ({delta_iie:+.2f}), σ² {pre['IIE_var']:.2f}→{post['IIE_var']:.2f} ({delta_var:+.2f})")
                    
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
                    print(f"[Actor/PRO] 📤 Sent to Learner")
                    
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
                    if abs(old_eps - self.epsilon) > 0.01:
                        print(f"[Actor/PRO] 🎲 ε: {self.epsilon:.2f}")
                    
                    # Signal self-adaptor that proactive action is complete
                    self._set_proactive_active(False)
                    
                    time.sleep(5.0)
                    
                except Exception as e:
                    print(f"[Actor/PRO] ❌ {e}")
                    time.sleep(1.0)
        except Exception as e:
            print(f"[Actor/PRO] ❌ Fatal: {e}")
        finally:
            if csv_file:
                csv_file.close()
        
        print("[Actor/PRO] ⏹️ Stopped")

    def _selfadaptor_loop(self):
        """Execute periodic self-regulation behaviors
        
        NOTE: Self-adaptors only execute when always-on is active.
        They represent natural background behaviors during active engagement.
        Proactive actions have priority - self-adaptor cycle aborts if proactive starts.
        """
        print("[Actor/SA] ▶️ Self-adaptor thread started")
        
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
                    # Random period between 60-120 seconds
                    period = random.uniform(60.0, 120.0)
                    print(f"[Actor/SA] ⏱️ Waiting {period:.1f}s until next self-adaptor...")
                    
                    # Interruptible sleep with proactive priority
                    sleep_step = 0.5
                    elapsed = 0.0
                    while self.running and elapsed < period:
                        # Check if proactive thread started executing
                        if self._is_proactive_executing():
                            print(f"[Actor/SA] ⏸️ Cycle aborted at {elapsed:.1f}s/{period:.1f}s (proactive action executing)")
                            # Wait for proactive to finish
                            while self.running and self._is_proactive_executing():
                                time.sleep(0.5)
                            # Restart period after proactive completes
                            new_period = random.uniform(60.0, 120.0)
                            print(f"[Actor/SA] 🔄 Proactive done, restarting with new period: {new_period:.1f}s")
                            period = new_period
                            elapsed = 0.0
                            continue
                        
                        time.sleep(sleep_step)
                        elapsed += sleep_step
                    
                    if not self.running:
                        break
                    
                    # Check if always-on is active before executing
                    if not self._is_alwayson_active():
                        print(f"[Actor/SA] ⏸️ Skipping: always-on is inactive (system stopped)")
                        continue
                    
                    # Final check: proactive might have started during last sleep
                    if self._is_proactive_executing():
                        print(f"[Actor/SA] ⏸️ Skipping: proactive action just started")
                        continue
                    
                    # Execute self-adaptor
                    timestamp = time.time()
                    pre = self._get_state_snapshot()
                    action = random.choice(self.SELF_ADAPTORS)
                    
                    print(f"[Actor/SA] 🔄 Executing self-adaptor: {action} (μ={pre['IIE_mean']:.2f}, CTX{pre['ctx']})")
                    
                    # Execute via RPC shell command
                    try:
                        result = subprocess.run(
                            f'echo "exe {action}" | yarp rpc /interactionInterface',
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode != 0:
                            print(f"[Actor/SA] ❌ {result.stderr.strip()}")
                            continue
                    except Exception as e:
                        print(f"[Actor/SA] ❌ {e}")
                        continue
                    
                    print(f"[Actor/SA] ⏳ Waiting {self.WAIT_AFTER_ACTION:.1f}s...")
                    time.sleep(self.WAIT_AFTER_ACTION)
                    post = self._get_state_snapshot()
                    
                    delta_iie = post['IIE_mean'] - pre['IIE_mean']
                    print(f"[Actor/SA] ✅ Completed: μ {pre['IIE_mean']:.2f}→{post['IIE_mean']:.2f} ({delta_iie:+.2f})")
                    
                    # Log to CSV
                    csv_writer.writerow([
                        f"{timestamp:.2f}", action,
                        f"{pre['IIE_mean']:.4f}", f"{pre['IIE_var']:.4f}", pre['ctx'],
                        f"{post['IIE_mean']:.4f}", f"{post['IIE_var']:.4f}", post['ctx']
                    ])
                    csv_file.flush()
                    
                except Exception as e:
                    print(f"[Actor/SA] ❌ {e}")
                    time.sleep(1.0)
        except Exception as e:
            print(f"[Actor/SA] ❌ Fatal: {e}")
        finally:
            if csv_file:
                csv_file.close()
        
        print("[Actor/SA] ⏹️ Stopped")

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
    
    # def _check_gatekeeper(self, pre, time_delta):
    #     """Load gatekeeper model from disk and predict 'Should I Act?'
    #     
    #     PHASE 3 INTEGRATION: Uncomment this method after 200+ training samples
    #     
    #     The model learned from RAW OUTCOMES (post-pre IIE delta > 0.02)
    #     It recognizes pre-conditions that historically led to engagement improvement
    #     
    #     Args:
    #         pre: Pre-state snapshot dict
    #         time_delta: Time since last action (seconds)
    #     
    #     Returns:
    #         bool: True = ACT (scene likely to improve), False = WAIT (scene unlikely to improve)
    #     """
    #     model_path = os.path.join(os.path.dirname(__file__), "gate_classifier.pkl")
    #     
    #     # If no model exists yet (early training), default to YES (allow exploration)
    #     if not os.path.exists(model_path):
    #         return True
    #     
    #     try:
    #         # Load the trained model from disk (read-only)
    #         with open(model_path, 'rb') as f:
    #             model = pickle.load(f)
    #         
    #         # Encode features: MUST MATCH learning.py _encode_gate_features() ORDER EXACTLY
    #         # [pre_IIE_mean, pre_IIE_var, pre_ctx, pre_num_faces, pre_num_mutual_gaze, time_delta]
    #         features = np.array([[
    #             pre['IIE_mean'],               # Interaction intention (pre)
    #             pre['IIE_var'],                # Intention stability (pre)
    #             float(pre['ctx']),             # Context (pre)
    #             float(pre['num_faces']),       # Audience size (pre)
    #             float(pre['num_mutual_gaze']), # Attention (pre)
    #             float(time_delta)              # Time since last action
    #         ]])
    #         
    #         # Predict: 1 = YES (these conditions led to improvement historically)
    #         #          0 = NO (these conditions did not improve engagement)
    #         prediction = model.predict(features)[0]
    #         return prediction == 1
    #     
    #     except Exception as e:
    #         print(f"[Gatekeeper] ⚠️ Prediction error: {e}")
    #         return True  # Default to ACT if model file is corrupt/busy
    
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
                print(f"[Actor] 💾 Q-table: {len(new_Q)} states")
        except Exception as e:
            if verbose:
                print(f"[Actor] ❌ Q-table: {e}")
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
