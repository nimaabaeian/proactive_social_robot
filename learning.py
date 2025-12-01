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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


class Learning(yarp.RFModule):

    # Configuration Constants
    QTABLE_PATH = "learning_qtable.json"
    QLEARNING_LOG = "qlearning_log.csv"
    MODEL_TRAINING_LOG = "model_training_log.csv"
    MODEL_PREDICTION_LOG = "model_prediction_log.csv"
    MODEL_PATH = "iie_transition_model.pkl"
    SCALER_PATH = "iie_scaler.pkl"
    CTX_MODEL_PATH = "ctx_transition_model.pkl"
    
    ALPHA = 0.30
    GAMMA = 0.92
    W_VAR = 0.5
    W_DELTA = 1.0     # Weight for IIE change
    W_LEVEL = 0.5     # Weight for maintaining high engagement level
    
    THRESH_MEAN = 0.5  # IIE threshold (must match embodiedBehaviour.THRESH_MEAN)
    
    DELTA_EPS = 0.05  # Dead zone: minimum IIE change to be considered meaningful
    VAR_EPS = 0.02    # Dead zone: minimum variance change to be considered meaningful
    
    BUFFER_SIZE = 10
    MODEL_MAX_DEPTH = 2
    MODEL_N_ESTIMATORS = 50
    MODEL_LEARNING_RATE = 0.1
    MAX_ESTIMATORS = 200
    
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
        self.MODEL_TRAINING_LOG = os.path.join(base_dir, self.MODEL_TRAINING_LOG)
        self.MODEL_PATH = os.path.join(base_dir, self.MODEL_PATH)
        self.SCALER_PATH = os.path.join(base_dir, self.SCALER_PATH)
        self.CTX_MODEL_PATH = os.path.join(base_dir, self.CTX_MODEL_PATH)
        self.MODEL_PREDICTION_LOG = os.path.join(base_dir, self.MODEL_PREDICTION_LOG)
        
        # Ports
        self.port_input = yarp.BufferedPortBottle()
        
        # Q-learning
        self.Q = {}
        self.q_lock = threading.Lock()
        
        # Transition models
        self.scaler = StandardScaler()
        self.iie_model = GradientBoostingRegressor(
            n_estimators=self.MODEL_N_ESTIMATORS,
            learning_rate=self.MODEL_LEARNING_RATE,
            max_depth=self.MODEL_MAX_DEPTH,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42,
            warm_start=True
        )
        self.ctx_model = GradientBoostingClassifier(
            n_estimators=self.MODEL_N_ESTIMATORS,
            learning_rate=self.MODEL_LEARNING_RATE,
            max_depth=self.MODEL_MAX_DEPTH,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42,
            warm_start=True
        )
        
        self.iie_initialized = False
        self.ctx_initialized = False
        self.iie_buffer_X = []
        self.iie_buffer_y = []
        self.ctx_buffer_X = []
        self.ctx_buffer_y = []
        
        # CSV logging
        self.qlearning_csv = None
        self.model_csv = None
        self.prediction_csv = None
        
        # Statistics
        self.qlearning_count = 0
        self.model_count = 0
    
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
        print("LEARNING MODULE - Single Unit Architecture")
        print("="*70)
        print(f"\n[Paths]")
        print(f"  Q-table: {self.QTABLE_PATH}")
        print(f"  Models: {self.MODEL_PATH}")
        print(f"  Logs: {self.QLEARNING_LOG}, {self.MODEL_PREDICTION_LOG}")
        
        # Open ports
        if not self.port_input.open("/alwayson/learning/experiences:i"):
            print("[ERROR] Failed to open input port")
            return False
        
        print("\n[Ports] Input port opened successfully")
        
        # Load Q-table and models
        self._load_qtable()
        self._load_models()
        
        # Initialize CSV logs
        self._init_csv_logs()
        
        print(f"\n[Config] Q-Learning: α={self.ALPHA}, γ={self.GAMMA}")
        print(f"[Config] Reward Weights: Δμ={self.W_DELTA}, σ²={self.W_VAR}, level={self.W_LEVEL}")
        print(f"[Config] Threshold: μ_min={self.THRESH_MEAN} (2x penalty if dropped below)")
        print(f"[Config] Dead Zones: Δμ_min={self.DELTA_EPS}, Δσ²_min={self.VAR_EPS} (noise filtering)")
        print(f"[Config] ML Models: {self.MODEL_N_ESTIMATORS} trees, depth={self.MODEL_MAX_DEPTH}, buffer={self.BUFFER_SIZE}")
        print(f"[Status] IIE Model={'Trained' if self.iie_initialized else 'New'}, Context Model={'Trained' if self.ctx_initialized else 'New'}")
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
        print("\n[Shutdown] Closing learning module...")
        
        # Save Q-table and models
        self._save_qtable()
        self._save_models()
        
        # Close CSV files
        if self.qlearning_csv:
            self.qlearning_csv.close()
        if self.model_csv:
            self.model_csv.close()
        if self.prediction_csv:
            self.prediction_csv.close()
        
        # Close ports
        self.port_input.close()
        
        print(f"[Shutdown] Q-updates: {self.qlearning_count}, ML training samples: {self.model_count}")
        print(f"[Shutdown] Models saved: IIE={self.iie_initialized}, Context={self.ctx_initialized}")
        print("[Shutdown] Complete")
        return True
    
    # ========================================================================
    # Experience Processing
    # ========================================================================
    
    def _process_experience(self, exp):
        try:
            print(f"\n{'='*60}")
            print(f"[EXP] 📥 Received: {exp.action}")
            print(f"[EXP] Pre:  CTX={exp.pre_ctx}, μ={exp.pre_IIE_mean:.2f}, σ²={exp.pre_IIE_var:.2f}, faces={exp.pre_num_faces}, gaze={exp.pre_num_mutual_gaze}")
            print(f"[EXP] Post: CTX={exp.post_ctx}, μ={exp.post_IIE_mean:.2f}, σ²={exp.post_IIE_var:.2f}, faces={exp.post_num_faces}, gaze={exp.post_num_mutual_gaze}")
            
            # ML Model Predictions (MUST happen before any training)
            predicted_iie_delta, predicted_post_mean, predicted_post_ctx = self._get_predictions(exp)
            
            pred_error = abs(exp.post_IIE_mean - predicted_post_mean)
            ctx_match = "✓" if predicted_post_ctx == exp.post_ctx else "✗"
            print(f"\n[Prediction] IIE Model:")
            print(f"  • Predicted: {exp.pre_IIE_mean:.2f} → {predicted_post_mean:.2f} (Δ={predicted_iie_delta:+.2f})")
            print(f"  • Actual:    {exp.pre_IIE_mean:.2f} → {exp.post_IIE_mean:.2f}")
            print(f"  • Error:     {pred_error:.3f}")
            print(f"[Prediction] Context Model: {predicted_post_ctx} {ctx_match} (actual: {exp.post_ctx})")
            
            # Log predictions vs actuals
            self._log_predictions(exp, predicted_iie_delta, predicted_post_mean, predicted_post_ctx)
            
            # Q-Learning Update
            if exp.pre_ctx != -1 and exp.post_ctx != -1:
                reward = self._compute_reward(exp)
                old_q, new_q, td_error = self._update_q(exp, reward)
                
                self._log_qlearning(exp, reward, old_q, new_q, td_error)
                self.qlearning_count += 1
                self._save_qtable()
                
                print(f"\n[Q-Learning] Update #{self.qlearning_count}:")
                print(f"  • Reward: {reward:+.3f}")
                print(f"  • Q-value: {old_q:.3f} → {new_q:.3f} (TD error: {td_error:+.3f})")
                print(f"  • State: CTX{exp.pre_ctx} → CTX{exp.post_ctx}")
                print(f"  • Q-table saved")
            
            # ML Model Training (happens AFTER prediction logging)
            features = self._encode_features(exp)
            target_delta = exp.post_IIE_mean - exp.pre_IIE_mean
            
            # Train IIE Model (regression)
            self._train_iie_model_only(features, target_delta)
            self.model_count += 1
            
            # Train Context Model (classification)
            if exp.pre_ctx in (0, 1) and exp.post_ctx in (0, 1):
                self._train_ctx_model(features, exp.post_ctx)
            
            # Log model training sample
            self._log_model_training(exp, target_delta, predicted_iie_delta, 
                                    abs(target_delta - predicted_iie_delta))
            
            print(f"\n[Training] Model #{self.model_count}:")
            print(f"  • IIE Buffer: {len(self.iie_buffer_X)}/{self.BUFFER_SIZE} samples")
            print(f"  • CTX Buffer: {len(self.ctx_buffer_X)}/{self.BUFFER_SIZE} samples")
            print(f"  • Target ΔIIE: {target_delta:+.3f}")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"[ERROR] Processing experience: {e}")

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
    # Model Methods
    # ========================================================================
    
    def _encode_features(self, exp):
        """Encode 8D feature vector"""
        action_id = self.ACTION_TO_ID.get(exp.action, 0)
        action_one_hot = [0.0, 0.0, 0.0]
        action_one_hot[action_id] = 1.0
        
        return np.array([[
            exp.pre_IIE_mean, exp.pre_IIE_var, float(exp.pre_ctx),
            float(exp.pre_num_faces), float(exp.pre_num_mutual_gaze),
            action_one_hot[0], action_one_hot[1], action_one_hot[2]
        ]])
    
    def _train_iie_model_only(self, features, target_delta):
        """Train IIE regressor (without returning predictions - those are made separately)"""
        self.scaler.partial_fit(features)
        features_scaled = self.scaler.transform(features)
        
        # Add to buffer
        self.iie_buffer_X.append(features_scaled[0])
        self.iie_buffer_y.append(target_delta)
        
        # Train when buffer full
        if len(self.iie_buffer_X) >= self.BUFFER_SIZE:
            X_batch = np.array(self.iie_buffer_X)
            y_batch = np.array(self.iie_buffer_y)
            
            if not self.iie_initialized:
                print(f"[IIE Model] 🎉 Initializing with {len(y_batch)} samples...")
                self.iie_model.fit(X_batch, y_batch)
                self.iie_initialized = True
                print(f"[IIE Model] ✓ Initialized ({self.iie_model.n_estimators} trees)")
            else:
                old_n = self.iie_model.n_estimators
                if self.iie_model.n_estimators < self.MAX_ESTIMATORS:
                    self.iie_model.n_estimators += 5
                print(f"[IIE Model] 📈 Training with {len(y_batch)} samples (trees: {old_n}→{self.iie_model.n_estimators})...")
                self.iie_model.fit(X_batch, y_batch)
                print(f"[IIE Model] ✓ Updated")
            
            self.iie_buffer_X.clear()
            self.iie_buffer_y.clear()
            print(f"[IIE Model] 🗑️ Buffer cleared")
    
    def _train_ctx_model(self, features, target_ctx):

        try:
            features_scaled = self.scaler.transform(features)
        except NotFittedError:
            # Scaler not fitted yet, skip this training
            return
        
        self.ctx_buffer_X.append(features_scaled[0])
        self.ctx_buffer_y.append(target_ctx)
        
        if len(self.ctx_buffer_X) >= self.BUFFER_SIZE:
            X_batch = np.array(self.ctx_buffer_X)
            y_batch = np.array(self.ctx_buffer_y)
            
            if not self.ctx_initialized:
                print(f"[CTX Model] 🎉 Initializing with {len(y_batch)} samples...")
                self.ctx_model.fit(X_batch, y_batch)
                self.ctx_initialized = True
                print(f"[CTX Model] ✓ Initialized ({self.ctx_model.n_estimators} trees)")
            else:
                old_n = self.ctx_model.n_estimators
                if self.ctx_model.n_estimators < self.MAX_ESTIMATORS:
                    self.ctx_model.n_estimators += 5
                print(f"[CTX Model] 📈 Training with {len(y_batch)} samples (trees: {old_n}→{self.ctx_model.n_estimators})...")
                self.ctx_model.fit(X_batch, y_batch)
                print(f"[CTX Model] ✓ Updated")
            
            self.ctx_buffer_X.clear()
            self.ctx_buffer_y.clear()
            print(f"[CTX Model] 🗑️ Buffer cleared")
    
    def _predict_iie_delta(self, action, pre_mean, pre_var, pre_ctx, pre_faces, pre_mutual_gaze):
        """Predict IIE mean change"""
        if not self.iie_initialized:
            return 0.0
        
        try:
            action_id = self.ACTION_TO_ID.get(action, 0)
            action_one_hot = [0.0, 0.0, 0.0]
            action_one_hot[action_id] = 1.0
            
            features = np.array([[
                pre_mean, pre_var, float(pre_ctx),
                float(pre_faces), float(pre_mutual_gaze),
                action_one_hot[0], action_one_hot[1], action_one_hot[2]
            ]])
            
            features_scaled = self.scaler.transform(features)
            return self.iie_model.predict(features_scaled)[0]
        except (NotFittedError, Exception) as e:
            return 0.0
    
    def _predict_post_ctx(self, action, pre_mean, pre_var, pre_ctx, pre_faces, pre_mutual_gaze):
        """Predict post-action context"""
        default = int(pre_ctx) if pre_ctx in (0, 1) else 0
        
        if not self.ctx_initialized:
            return default
        
        try:
            action_id = self.ACTION_TO_ID.get(action, 0)
            action_one_hot = [0.0, 0.0, 0.0]
            action_one_hot[action_id] = 1.0
            
            features = np.array([[
                pre_mean, pre_var, float(pre_ctx),
                float(pre_faces), float(pre_mutual_gaze),
                action_one_hot[0], action_one_hot[1], action_one_hot[2]
            ]])
            
            features_scaled = self.scaler.transform(features)
            proba = self.ctx_model.predict_proba(features_scaled)[0]
            return int(proba[1] >= 0.5)
        except (NotFittedError, Exception):
            return default
    
    def _get_predictions(self, exp):
        """Get ML model predictions for experience (before training)"""
        predicted_iie_delta = self._predict_iie_delta(
            exp.action, exp.pre_IIE_mean, exp.pre_IIE_var, exp.pre_ctx,
            exp.pre_num_faces, exp.pre_num_mutual_gaze
        )
        predicted_post_mean = exp.pre_IIE_mean + predicted_iie_delta
        predicted_post_ctx = self._predict_post_ctx(
            exp.action, exp.pre_IIE_mean, exp.pre_IIE_var, exp.pre_ctx,
            exp.pre_num_faces, exp.pre_num_mutual_gaze
        )
        return predicted_iie_delta, predicted_post_mean, predicted_post_ctx

    # ========================================================================
    # File I/O
    # ========================================================================
    
    def _load_qtable(self):
        """Load Q-table from JSON"""
        if not os.path.exists(self.QTABLE_PATH):
            print("[Q-Table] Not found, initializing empty")
            return
        
        try:
            with open(self.QTABLE_PATH, 'r') as f:
                data = json.load(f)
            
            with self.q_lock:
                self.Q = data.get("Q", {})
            
            print(f"[Q-Table] Loaded: {len(self.Q)} states")
        except Exception as e:
            print(f"[Q-Table] Error loading: {e}")
    
    def _save_qtable(self):
        """Save Q-table to JSON with atomic write"""
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
            
            if self.qlearning_count % 10 == 0:
                print(f"✓ Q-table saved (n={self.qlearning_count} states={len(self.Q)})")
        except Exception as e:
            print(f"[ERROR] Saving Q-table: {e}")
    
    def _load_models(self):
        """Load transition models"""
        success = False
        
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.SCALER_PATH):
            try:
                with open(self.MODEL_PATH, 'rb') as f:
                    self.iie_model = pickle.load(f)
                with open(self.SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.iie_initialized = True
                success = True
            except Exception as e:
                print(f"[Models] Error loading IIE model: {e}")
        
        if os.path.exists(self.CTX_MODEL_PATH):
            try:
                with open(self.CTX_MODEL_PATH, 'rb') as f:
                    self.ctx_model = pickle.load(f)
                self.ctx_initialized = True
                success = True
            except Exception as e:
                print(f"[Models] Error loading context model: {e}")
        
        if success:
            print("[Models] Loaded from disk")
        else:
            print("[Models] Starting fresh")
    
    def _save_models(self):
        """Save transition models"""
        try:
            if self.iie_initialized:
                with open(self.MODEL_PATH, 'wb') as f:
                    pickle.dump(self.iie_model, f)
                with open(self.SCALER_PATH, 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            if self.ctx_initialized:
                with open(self.CTX_MODEL_PATH, 'wb') as f:
                    pickle.dump(self.ctx_model, f)
            
            print("[Models] Saved to disk")
        except Exception as e:
            print(f"[ERROR] Saving models: {e}")

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
            
            # Model training log
            file_exists = os.path.exists(self.MODEL_TRAINING_LOG)
            self.model_csv = open(self.MODEL_TRAINING_LOG, 'a', newline='')
            writer = csv.writer(self.model_csv)
            
            if not file_exists:
                writer.writerow(['timestamp', 'proactive_action', 'pre_IIE_mean', 'pre_IIE_var',
                               'pre_ctx', 'pre_num_faces', 'pre_num_mutual_gaze',
                               'target_delta_IIE', 'predicted_delta_IIE',
                               'prediction_error', 'model_training_count'])
            
            # Model prediction log
            file_exists = os.path.exists(self.MODEL_PREDICTION_LOG)
            self.prediction_csv = open(self.MODEL_PREDICTION_LOG, 'a', newline='')
            writer = csv.writer(self.prediction_csv)
            
            if not file_exists:
                writer.writerow(['timestamp', 'proactive_action',
                               'pre_IIE_mean', 'pre_IIE_var', 'pre_ctx',
                               'pre_num_faces', 'pre_num_mutual_gaze',
                               'predicted_IIE_delta', 'predicted_post_IIE_mean', 'predicted_post_ctx',
                               'actual_post_IIE_mean', 'actual_post_ctx',
                               'iie_prediction_error', 'ctx_prediction_correct'])
        except Exception as e:
            print(f"[ERROR] Initializing CSV logs: {e}")
            # Close any opened files
            if self.qlearning_csv:
                self.qlearning_csv.close()
                self.qlearning_csv = None
            if self.model_csv:
                self.model_csv.close()
                self.model_csv = None
            if self.prediction_csv:
                self.prediction_csv.close()
                self.prediction_csv = None
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
    
    def _log_model_training(self, exp, target_delta, predicted_delta, error):
        """Log model training"""
        if not self.model_csv:
            return
        
        writer = csv.writer(self.model_csv)
        writer.writerow([
            exp.timestamp, exp.action,
            f"{exp.pre_IIE_mean:.4f}", f"{exp.pre_IIE_var:.4f}", exp.pre_ctx,
            exp.pre_num_faces, exp.pre_num_mutual_gaze,
            f"{target_delta:.4f}", f"{predicted_delta:.4f}",
            f"{error:.4f}", self.model_count
        ])
        self.model_csv.flush()
    
    def _log_predictions(self, exp, predicted_iie_delta, predicted_post_mean, predicted_post_ctx):
        """Log ML model predictions vs actual outcomes"""
        if not self.prediction_csv:
            return
        
        iie_error = abs(exp.post_IIE_mean - predicted_post_mean)
        ctx_correct = 1 if predicted_post_ctx == exp.post_ctx else 0
        
        writer = csv.writer(self.prediction_csv)
        writer.writerow([
            exp.timestamp, exp.action,
            f"{exp.pre_IIE_mean:.4f}", f"{exp.pre_IIE_var:.4f}", exp.pre_ctx,
            exp.pre_num_faces, exp.pre_num_mutual_gaze,
            f"{predicted_iie_delta:.4f}", f"{predicted_post_mean:.4f}", predicted_post_ctx,
            f"{exp.post_IIE_mean:.4f}", exp.post_ctx,
            f"{iie_error:.4f}", ctx_correct
        ])
        self.prediction_csv.flush()


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
