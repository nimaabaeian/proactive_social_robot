"""
Learning Module 
"""
import yarp
import time
import json
import csv
import os
import sys
import signal
import pickle
import numpy as np
import threading
from dataclasses import dataclass
from sklearn.ensemble import  GradientBoostingClassifier

class Learning(yarp.RFModule):

    # Configuration Constants
    QTABLE_PATH = "learning_qtable.json"
    QLEARNING_LOG = "qlearning_log.csv"
    GATE_MODEL_PATH = "gate_classifier.pkl"
    GATE_LOG = "gate_training_log.csv"
    
    ALPHA = 0.30
    GAMMA = 0.92
    W_VAR = 0.5
    W_DELTA = 1.0     # Weight for IIE change
    W_LEVEL = 0.5     # Weight for maintaining high engagement level
    
    THRESH_MEAN = 0.5  # IIE threshold (must match embodiedBehaviour.THRESH_MEAN)
    
    DELTA_EPS = 0.05  # Dead zone: minimum IIE change to be considered meaningful
    VAR_EPS = 0.02    # Dead zone: minimum variance change to be considered meaningful
    
    BUFFER_SIZE = 4
    GATE_MAX_DEPTH = 3
    GATE_N_ESTIMATORS = 100
    GATE_LEARNING_RATE = 0.1
    GATE_MAX_ESTIMATORS = 200
    
    REWARD_THRESHOLD = 0.05  # Minimum reward to classify scene as "YES"
    
    ACTION_COSTS = {
        "ao_greet": 0.08,
        "ao_coffee_break": 0.10,
        "ao_curious_lean_in": 0.06,
    }
    
    ACTION_TO_ID = {
        "ao_greet": 0,
        "ao_coffee_break": 1,
        "ao_curious_lean_in": 2,
    }
    
    def __init__(self):
        super().__init__()
        
        # Resolve paths
        base_dir = os.path.dirname(__file__)
        self.QTABLE_PATH = os.path.join(base_dir, self.QTABLE_PATH)
        self.QLEARNING_LOG = os.path.join(base_dir, self.QLEARNING_LOG)
        self.GATE_MODEL_PATH = os.path.join(base_dir, self.GATE_MODEL_PATH)
        self.GATE_LOG = os.path.join(base_dir, self.GATE_LOG)
        
        # Ports
        self.port_input = yarp.BufferedPortBottle()
        
        # Q-learning
        self.Q = {}
        self.q_lock = threading.Lock()
        
        # Gatekeeper Model (Scene Discriminator)
        self.gate_model = GradientBoostingClassifier(
            n_estimators=self.GATE_N_ESTIMATORS,
            learning_rate=self.GATE_LEARNING_RATE,
            max_depth=self.GATE_MAX_DEPTH,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42,
            warm_start=True
        )
        
        self.gate_initialized = False
        self.gate_buffer_X = []
        self.gate_buffer_y = []
        
        # Timing for scene discrimination
        self.last_exp_timestamp = None
        
        # CSV logging
        self.qlearning_csv = None
        self.gate_csv = None
        
        # Statistics
        self.qlearning_count = 0
        self.gate_count = 0
    
    # ========================================================================
    # Experience Data Structure
    # ========================================================================
    
    @dataclass
    class Experience:
        """Experience bottle structure"""
        timestamp: float
        action: str
        pre_ctx: int
        pre_IIE_mean: float
        pre_IIE_var: float
        post_ctx: int
        post_IIE_mean: float
        post_IIE_var: float
        q_value: float
        pre_num_faces: int = 0
        pre_num_mutual_gaze: int = 0
        post_num_faces: int = 0
        post_num_mutual_gaze: int = 0
        
        @staticmethod
        def from_bottle(bottle):
            """Parse experience from YARP bottle"""
            if bottle.size() < 9:
                return None
            
            return Learning.Experience(
                timestamp=bottle.get(0).asFloat64(),
                action=bottle.get(1).asString(),
                pre_ctx=bottle.get(2).asInt8(),
                pre_IIE_mean=bottle.get(3).asFloat64(),
                pre_IIE_var=bottle.get(4).asFloat64(),
                post_ctx=bottle.get(5).asInt8(),
                post_IIE_mean=bottle.get(6).asFloat64(),
                post_IIE_var=bottle.get(7).asFloat64(),
                q_value=bottle.get(8).asFloat64(),
                pre_num_faces=bottle.get(9).asInt32() if bottle.size() > 9 else 0,
                pre_num_mutual_gaze=bottle.get(10).asInt32() if bottle.size() > 10 else 0,
                post_num_faces=bottle.get(11).asInt32() if bottle.size() > 11 else 0,
                post_num_mutual_gaze=bottle.get(12).asInt32() if bottle.size() > 12 else 0
            )

    # ========================================================================
    # RFModule Interface
    # ========================================================================

    def configure(self, rf):
        """Initialize module"""
        print("\n" + "="*70)
        print("[Learner] 🧠 LEARNING MODULE")
        print("="*70)
        print(f"[Learner] 📁 Paths: Q={os.path.basename(self.QTABLE_PATH)}, Gate={os.path.basename(self.GATE_MODEL_PATH)}")
        
        # Open ports
        if not self.port_input.open("/alwayson/learning/experiences:i"):
            print("[Learner] ❌ Port failed")
            return False
        
        print("[Learner] ✅ Port ready")
        
        # Load Q-table and gatekeeper model
        self._load_qtable()
        self._load_gate_model()
        
        # Initialize CSV logs
        self._init_csv_logs()
        
        print(f"[Learner] 📊 Q-Learning: α={self.ALPHA}, γ={self.GAMMA}")
        print(f"[Learner] 🎯 Gatekeeper: {'Trained' if self.gate_initialized else 'New'} ({self.GATE_N_ESTIMATORS} trees)")
        print("="*70 + "\n")
        
        return True
    
    def getPeriod(self):
        return 0.1  # 10 Hz for experience processing
    
    def updateModule(self):
        """Process experiences"""
        # Process experience bottles
        bottle = self.port_input.read(False)
        if bottle and not bottle.isNull():
            exp = self.Experience.from_bottle(bottle)
            if exp:
                self._process_experience(exp)
        
        return True
    
    def interruptModule(self):
        """Stop module"""
        self.port_input.interrupt()
        return True
    
    def close(self):
        """Cleanup"""
        print("\n[Learner] 🛑 Shutting down...")
        
        # Save Q-table and gatekeeper model
        self._save_qtable()
        self._save_gate_model()
        
        # Close CSV files
        if self.qlearning_csv:
            self.qlearning_csv.close()
        if self.gate_csv:
            self.gate_csv.close()
        
        # Close ports
        self.port_input.close()
        
        print(f"[Learner] 💾 Saved: Q={self.qlearning_count}, Gate={self.gate_count}")
        print("[Learner] ✅ Shutdown complete")
        return True
    
    # ========================================================================
    # Experience Processing
    # ========================================================================
    
    def _process_experience(self, exp):
        """Process experience: Q-learning + gatekeeper training"""
        try:
            print(f"\n[Learner] 📥 {exp.action}")
            print(f"[Learner] Pre→Post: μ {exp.pre_IIE_mean:.2f}→{exp.post_IIE_mean:.2f}, CTX{exp.pre_ctx}→{exp.post_ctx}")
            
            # Calculate time delta since last experience
            if self.last_exp_timestamp is None:
                time_delta = 10.0  # Default for first experience
            else:
                time_delta = exp.timestamp - self.last_exp_timestamp
                time_delta = min(time_delta, 60.0)  # Clamp to 60s max (prevent outliers after long idle)
            
            # Store experience for logging (needed by _log_gate_training)
            self._last_exp = exp
            
            # Q-Learning Update
            if exp.pre_ctx != -1 and exp.post_ctx != -1:
                reward = self._compute_reward(exp)
                old_q, new_q, td_error = self._update_q(exp, reward)
                
                self._log_qlearning(exp, reward, old_q, new_q, td_error)
                self.qlearning_count += 1
                self._save_qtable()
                
                reward_emoji = "✅" if reward > 0 else "⏸️" if reward == 0 else "❌"
                print(f"[Learner/Q] {reward_emoji} R={reward:+.2f}, Q: {old_q:.2f}→{new_q:.2f}, TD={td_error:+.2f}")
                
                # Train Gatekeeper (Scene Discriminator)
                # OBJECTIVE: Learn from RAW OUTCOME (independent of Q-learning reward)
                # 
                # Calculate Raw Improvement (Physical Reality)
                raw_delta = exp.post_IIE_mean - exp.pre_IIE_mean
                
                # Assign Label based on Direct Comparison (Pre vs Post)
                # If interaction caused real engagement increase → It was a "YES" moment
                # Threshold: 0.02 (small positive change = improvement)
                if raw_delta > 0.02:
                    label = 1  # YES: This pre-state led to improvement
                else:
                    label = 0  # NO: This pre-state did not lead to improvement
                
                label_emoji = "✅YES" if label == 1 else "⏸️NO"
                print(f"[Learner/Gate] {label_emoji} Δ={raw_delta:+.2f}")
                
                # Encode pre-state features (6D - for inference)
                features = self._encode_gate_features(exp, time_delta)
                
                # Train gatekeeper model
                self._train_gate_model(features, label, reward)
                self.gate_count += 1
            
            # Update timestamp for next experience
            self.last_exp_timestamp = exp.timestamp
            
        except Exception as e:
            print(f"[Learner] ❌ {e}")

    # ========================================================================
    # Q-Learning Methods
    # ========================================================================
    
    def _compute_reward(self, exp):
        # Apply dead zone to IIE mean change
        delta_mean = exp.post_IIE_mean - exp.pre_IIE_mean
        if abs(delta_mean) < self.DELTA_EPS:
            delta_mean = 0.0
        
        # Apply dead zone to variance reduction
        var_reduction = exp.pre_IIE_var - exp.post_IIE_var
        if abs(var_reduction) < self.VAR_EPS:
            var_reduction = 0.0
        
        # Level term: reward being above threshold, penalize falling below
        # Positive when comfortably above threshold, negative when below
        level_term = exp.post_IIE_mean - self.THRESH_MEAN
        if exp.post_IIE_mean < self.THRESH_MEAN:
            # Double penalty for breaking the engagement threshold
            level_term *= 2.0
        
        action_cost = self.ACTION_COSTS.get(exp.action, 0.0)
        
        reward = (
            self.W_DELTA * delta_mean +
            self.W_VAR * var_reduction +
            self.W_LEVEL * level_term -
            action_cost
        )
        
        # Clip reward to [-1.0, 1.0] for stability (avoid outliers)
        reward = max(-1.0, min(1.0, reward))
        
        return reward
    
    def _update_q(self, exp, reward):
        """TD update: Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',·)) - Q(s,a)]"""
        pre_state = f"CTX{exp.pre_ctx}"
        post_state = f"CTX{exp.post_ctx}"
        
        with self.q_lock:
            if pre_state not in self.Q:
                self.Q[pre_state] = {}
            if exp.action not in self.Q[pre_state]:
                self.Q[pre_state][exp.action] = 0.0
            if post_state not in self.Q:
                self.Q[post_state] = {}
            
            old_q = self.Q[pre_state][exp.action]
            max_next_q = max(self.Q[post_state].values()) if self.Q[post_state] else 0.0
            td_target = reward + self.GAMMA * max_next_q
            td_error = td_target - old_q
            new_q = old_q + self.ALPHA * td_error
            
            self.Q[pre_state][exp.action] = new_q
            
            return old_q, new_q, td_error

    # ========================================================================
    # Gatekeeper (Scene Discriminator) Methods
    # ========================================================================
    
    def _encode_gate_features(self, exp, time_delta):
        """Encode pre-action state features for inference
        
        CRITICAL: Only pre-state features for prediction
        The model must decide based on what it sees BEFORE acting
        """
        return np.array([[
            exp.pre_IIE_mean,                  # Interaction intention (pre)
            exp.pre_IIE_var,                   # Intention stability (pre)
            float(exp.pre_ctx),                # Context (pre)
            float(exp.pre_num_faces),          # Audience size (pre)
            float(exp.pre_num_mutual_gaze),    # Attention (pre)
            float(time_delta)                  # Time since last action
        ]])
    
    def _train_gate_model(self, features, label, reward):
        """Train gatekeeper classifier to recognize opportune moments
        
        OBJECTIVE: Learn from raw outcomes (pre vs post), NOT reward function
        
        Args:
            features: 6D pre-state feature vector (inference input)
            label: 1 (YES - led to improvement) or 0 (NO - no improvement)
            reward: Q-learning reward (logged for reference only, NOT used in training)
        
        Labeling Logic:
            raw_delta = post_IIE - pre_IIE
            label = 1 if raw_delta > 0.02 else 0
        
        The model learns: "These pre-conditions historically led to engagement increase"
        """
        # Add to buffer
        self.gate_buffer_X.append(features[0])
        self.gate_buffer_y.append(label)
        
        # Train when buffer full
        if len(self.gate_buffer_X) >= self.BUFFER_SIZE:
            X_batch = np.array(self.gate_buffer_X)
            y_batch = np.array(self.gate_buffer_y)
            
            if not self.gate_initialized:
                self.gate_model.fit(X_batch, y_batch)
                self.gate_initialized = True
                print(f"[Learner/Gate] 🎉 Initialized ({self.gate_model.n_estimators} trees)")
            else:
                old_n = self.gate_model.n_estimators
                if self.gate_model.n_estimators < self.GATE_MAX_ESTIMATORS:
                    self.gate_model.n_estimators += 5
                self.gate_model.fit(X_batch, y_batch)
                print(f"[Learner/Gate] 📈 Trained: {old_n}→{self.gate_model.n_estimators} trees")
            
            # Log training batch
            yes_count = sum(y_batch)
            no_count = len(y_batch) - yes_count
            print(f"[Learner/Gate] Batch: {yes_count}✅ {no_count}⏸️")
            
            # Save model
            self._save_gate_model()
            
            self.gate_buffer_X.clear()
            self.gate_buffer_y.clear()
        
        # Log individual training sample
        # Note: features[0] is the 1D array from features (6 elements)
        # We need to pass exp data separately for logging post values
        self._log_gate_training(features[0], label, reward, 
                               post_IIE_mean=self._last_exp.post_IIE_mean,
                               raw_delta=(self._last_exp.post_IIE_mean - self._last_exp.pre_IIE_mean))
    
    def predict_decision(self, pre_IIE_mean, pre_IIE_var, pre_ctx, pre_num_faces, pre_num_mutual_gaze, time_delta):

        if not self.gate_initialized:
            # Model not trained yet, default to True (allow actions)
            return True
        
        try:
            features = np.array([[
                pre_IIE_mean,
                pre_IIE_var,
                float(pre_ctx),
                float(pre_num_faces),
                float(pre_num_mutual_gaze),
                float(time_delta)
            ]])
            
            prediction = self.gate_model.predict(features)[0]
            return bool(prediction == 1)
        except Exception as e:
            print(f"[Learner/Gate] ❌ Predict: {e}")
            return True  # Default to allowing action on error

    # ========================================================================
    # File I/O
    # ========================================================================
    
    def _load_qtable(self):
        """Load Q-table from JSON"""
        if not os.path.exists(self.QTABLE_PATH):
            print("[Learner] 💾 Q-table: new")
            return
        
        try:
            with open(self.QTABLE_PATH, 'r') as f:
                data = json.load(f)
            
            with self.q_lock:
                self.Q = data.get("Q", {})
            
            print(f"[Learner] 💾 Q-table: {len(self.Q)} states")
        except Exception as e:
            print(f"[Learner] ❌ Q-table: {e}")
    
    def _save_qtable(self):
        """Save Q-table to JSON with atomic write (temp + rename)"""
        try:
            with self.q_lock:
                Q_copy = dict(self.Q)
            
            data = {
                "Q": Q_copy,
                "last_update": time.time(),
                "update_count": self.qlearning_count
            }
            
            # Atomic write: write to temp file then rename
            tmp_path = self.QTABLE_PATH + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.QTABLE_PATH)  # Atomic on POSIX systems
        except Exception as e:
            print(f"[Learner] ❌ Q-save: {e}")
    
    def _load_gate_model(self):
        """Load gatekeeper model from disk"""
        if not os.path.exists(self.GATE_MODEL_PATH):
            print("[Learner/Gate] 💾 Model: new")
            return
        
        try:
            with open(self.GATE_MODEL_PATH, 'rb') as f:
                self.gate_model = pickle.load(f)
            self.gate_initialized = True
            print(f"[Learner/Gate] 💾 Model: {self.gate_model.n_estimators} trees")
        except Exception as e:
            print(f"[Learner/Gate] ❌ Load failed: {e}")
    
    def _save_gate_model(self):
        """Save gatekeeper model to disk"""
        if not self.gate_initialized:
            return
        
        try:
            with open(self.GATE_MODEL_PATH, 'wb') as f:
                pickle.dump(self.gate_model, f)
            print(f"[Learner/Gate] 💾 Saved ({self.gate_model.n_estimators} trees)")
        except Exception as e:
            print(f"[Learner/Gate] ❌ Save failed: {e}")

    # ========================================================================
    # CSV Logging
    # ========================================================================
    
    def _init_csv_logs(self):
        """Initialize CSV log files"""
        try:
            # Q-learning log
            file_exists = os.path.exists(self.QLEARNING_LOG)
            self.qlearning_csv = open(self.QLEARNING_LOG, 'a', newline='')
            writer = csv.writer(self.qlearning_csv)
            
            if not file_exists:
                writer.writerow(['timestamp', 'proactive_action', 'pre_state', 'post_state',
                               'pre_IIE_mean', 'post_IIE_mean', 'delta_mean',
                               'pre_IIE_var', 'post_IIE_var', 'var_reduction',
                               'reward', 'old_q', 'new_q', 'td_error'])
            
            # Gatekeeper training log
            file_exists = os.path.exists(self.GATE_LOG)
            self.gate_csv = open(self.GATE_LOG, 'a', newline='')
            writer = csv.writer(self.gate_csv)
            
            if not file_exists:
                writer.writerow(['timestamp', 'pre_IIE_mean', 'pre_IIE_var', 'pre_ctx',
                               'pre_num_faces', 'pre_num_mutual_gaze', 'time_delta',
                               'post_IIE_mean', 'raw_delta', 'label', 'reward_ref', 'gate_count'])
        except Exception as e:
            print(f"[Learner] ❌ CSV init: {e}")
            # Close any opened files
            if self.qlearning_csv:
                self.qlearning_csv.close()
                self.qlearning_csv = None
            if self.gate_csv:
                self.gate_csv.close()
                self.gate_csv = None
            raise
    
    def _log_qlearning(self, exp, reward, old_q, new_q, td_error):
        """Log Q-learning update"""
        if not self.qlearning_csv:
            return
        
        writer = csv.writer(self.qlearning_csv)
        writer.writerow([
            exp.timestamp, exp.action, f"CTX{exp.pre_ctx}", f"CTX{exp.post_ctx}",
            f"{exp.pre_IIE_mean:.4f}", f"{exp.post_IIE_mean:.4f}", 
            f"{exp.post_IIE_mean - exp.pre_IIE_mean:.4f}",
            f"{exp.pre_IIE_var:.4f}", f"{exp.post_IIE_var:.4f}",
            f"{exp.pre_IIE_var - exp.post_IIE_var:.4f}",
            f"{reward:.4f}", f"{old_q:.4f}", f"{new_q:.4f}", f"{td_error:.4f}"
        ])
        self.qlearning_csv.flush()
    
    def _log_gate_training(self, features, label, reward, post_IIE_mean, raw_delta):
        """Log gatekeeper training sample"""
        if not self.gate_csv:
            return
        
        writer = csv.writer(self.gate_csv)
        writer.writerow([
            time.time(),
            f"{features[0]:.4f}",  # pre_IIE_mean (feature)
            f"{features[1]:.4f}",  # pre_IIE_var (feature)
            int(features[2]),      # pre_ctx (feature)
            int(features[3]),      # pre_num_faces (feature)
            int(features[4]),      # pre_num_mutual_gaze (feature)
            f"{features[5]:.2f}",  # time_delta (feature)
            f"{post_IIE_mean:.4f}",  # post_IIE_mean (outcome, not feature)
            f"{raw_delta:.4f}",      # raw_delta (labeling criterion)
            label,                    # 1 (YES) or 0 (NO)
            f"{reward:.4f}",          # reward_ref (Q-learning only, not used here)
            self.gate_count
        ])
        self.gate_csv.flush()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    yarp.Network.init()
    
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("learning")
    rf.configure(sys.argv)
    
    module = Learning()
    
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
