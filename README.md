# Embodied Behaviour & Learning Modules Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Embodied Behaviour Module](#embodied-behaviour-module)
4. [Learning Module](#learning-module)
5. [Communication Protocol](#communication-protocol)
6. [Configuration & Parameters](#configuration--parameters)
7. [Data Logging](#data-logging)
8. [Reward Function Design](#reward-function-design)
9. [Learning Progress & Convergence](#learning-progress--convergence)

---

## System Overview

### Purpose
This system implements a **developmental reinforcement learning architecture** for social robot interaction. The robot learns which social behaviors (greetings, gestures, etc.) are most effective in different contexts through experience.

### Core Concept
- **Actor-Critic Architecture**: Two independent modules work together
  - **Embodied Behaviour** (Actor): Perceives, decides, and acts
  - **Learning** (Critic): Evaluates actions and improves decision-making

### Key Features
- **Real-time learning**: Updates after every interaction
- **Context-aware**: Adapts behavior to calm vs. lively environments
- **Noise-robust**: Filters out measurement jitter with dead zones and rolling window averaging (3s history, 60 samples at 20Hz)
- **Exploration-exploitation**: Balances trying new things vs. using known strategies
- **Always-on autonomy**: Auto-stops when no one is present, restarts when people return
- **Data freshness monitoring**: Detects and handles stale sensor data to prevent frozen state spam
- **High-performance**: Non-blocking state capture (0.0s latency), native YARP RPC, fire-and-forget action execution

### Performance Optimizations

**1. Rolling Window Buffers (Non-Blocking, Thread-Safe)**
- IIE Monitor continuously populates a 60-sample rolling buffer at 20Hz
- State snapshots read instantly from buffer (0.0s vs 3.0s blocking)
- Maintains 3-second noise filtering without wait time
- Pre/post state capture: **6.0s → 0.0s** (instant)
- Thread-safe: `_state_lock` protects deque from concurrent append/iteration race conditions

**2. Native YARP RPC (Fire-and-Forget)**
- Uses native `yarp.RpcClient` for action execution
- Direct bottle communication to `/interactionInterface`
- Fire-and-forget pattern: no reply checking needed
- Action execution: **faster, more reliable**

**3. Data Freshness Monitoring**
- Detects stale/frozen sensor streams
- IIE timeout: 5.0s (expected 20Hz updates)
- Info timeout: 5.0s (expected 2Hz updates)
- Auto-resets to safe defaults when data freezes:
  - IIE → `mean=0.0`, `var=1.0` (blocks actions)
  - Info → `faces=0`, `gaze=0` (triggers always-on stop)
- Prevents action spam from frozen sensor values

**4. Optimized Timing**
- `WAIT_AFTER_ACTION`: **3.5s** (ensures rolling window fully refreshes with post-action data)
- `COOLDOWN`: **5.0s** (minimum time between proactive actions)
- Total cycle time: ~9s per action (3.5s wait + 5.0s cooldown + decision overhead)

**Why 3.5s Wait?**
- Action execution time: ~0.5s
- Human reaction time: ~1.0s
- Rolling window refresh: 3.0s (60 samples at 20Hz)
- Safety margin: +0.5s for timing jitter
- The wait ensures post-state measurement contains only post-action data
- Without full refresh, post-state would be contaminated with 50%+ pre-action samples

---

## Architecture

### Module Interaction Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMBODIED BEHAVIOUR MODULE                     │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ IIE Monitor  │  │Context Mon.  │  │ Info Monitor │          │
│  │ (Intention)  │  │ (Calm/Lively)│  │ (Faces/Gaze) │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └──────────────────┴──────────────────┘                  │
│                         │                                         │
│                    Shared State                                   │
│              (IIE_mean, IIE_var, ctx,                            │
│               num_faces, num_mutual_gaze)                         │
│                         │                                         │
│         ┌───────────────┴────────────────┐                       │
│         │                                 │                       │
│  ┌──────▼──────┐              ┌──────────▼─────┐                │
│  │  Proactive  │              │ Self-Adaptor   │                │
│  │   Thread    │              │    Thread      │                │
│  │             │              │                │                │
│  │ • Load Q    │              │ • ao_yawn      │                │
│  │ • Select    │              │ • ao_cough     │                │
│  │ • Execute   │              │ • ao_look      │                │
│  │ • Measure   │              │   (periodic)   │                │
│  └──────┬──────┘              └────────────────┘                │
│         │                                                         │
│         │ Experience Bottle                                      │
│         │ (13 fields)                                            │
└─────────┼─────────────────────────────────────────────────────┘
          │
          │ YARP Port: /alwayson/embodiedbehaviour/experiences:o
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LEARNING MODULE                            │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │            Experience Processing Pipeline                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────┐         ┌──────────────┐                       │
│  │ 1. PREDICT  │────────▶│ Log Pred vs  │                       │
│  │ (ML Models) │         │   Actual     │                       │
│  └─────────────┘         └──────────────┘                       │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────┐         ┌──────────────┐                       │
│  │ 2. COMPUTE  │────────▶│ Q-Learning   │                       │
│  │   REWARD    │         │    Update    │                       │
│  └─────────────┘         └──────┬───────┘                       │
│                                  │                                │
│                                  ▼                                │
│                          ┌──────────────┐                        │
│                          │ Save Q-table │                        │
│                          │   (JSON)     │                        │
│                          └──────────────┘                        │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────┐         ┌──────────────┐                       │
│  │ 3. TRAIN ML │────────▶│ Buffer (10x) │                       │
│  │   MODELS    │         │ IIE + CTX    │                       │
│  └─────────────┘         └──────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Embodied Behaviour Module

### Overview
The actor module that perceives the environment, makes decisions, and executes actions.

### Thread Architecture (6 Threads)

#### 1. IIE Monitor Thread
**Purpose**: Tracks interaction intention estimation

**Input Port**: `/alwayson/embodiedbehaviour/iie:i`

**What it does**:
- Reads IIE distributions for all detected faces
- Each face has: `[mean, variance]`
- Selects the person with **highest mean** (most engaged)
- Updates shared state: `IIE_mean`, `IIE_var`

**Update Rate**: 20 Hz (every 0.05s)

**Example Bottle**:
```
((0 (faceID mean variance std)) (1 (faceID mean variance std)) ...)
```

#### 2. Context Monitor Thread
**Purpose**: Tracks environmental context classification

**Input Port**: `/alwayson/embodiedbehaviour/context:i`

**What it does**:
- Reads context label from STM
- `0` = Calm environment
- `1` = Lively environment
- `-1` = Uncertain
- Updates shared state: `ctx`

**Update Rate**: 10 Hz (every 0.1s)

**Example Bottle**:
```
(episode_id chunk_id label)
→ (42 5 1) means episode 42, chunk 5, lively context
```

#### 3. Info Monitor Thread
**Purpose**: Tracks face count and mutual gaze

**Input Port**: `/alwayson/embodiedbehaviour/info:i`

**What it does**:
- Parses STM info bottle
- Extracts `Faces` and `MutualGaze` counts
- Updates shared state: `num_faces`, `num_mutual_gaze`
- Updates `last_faces_seen_time` (for always-on control)

**Update Rate**: 2 Hz (every 0.5s)

**Bottle Format**:
```
[
  (Time t_vision t_audio t_delta)
  (timestamp Faces n People n Light val Motion val MutualGaze n)
  (timestamp Audio (right left))
]
```

**Parsing Logic**:
```python
vision_list = bottle.get(1).asList()
# Skip timestamp, parse key-value pairs
i = 1
while i < vision_list.size() - 1:
    key = vision_list.get(i).toString()     # "Faces", "MutualGaze", etc.
    value = vision_list.get(i + 1)          # numeric value
    i += 2
```

#### 4. Always-On Monitor Thread
**Purpose**: Auto-stop/start system based on face presence

**What it does**:
- Monitors `num_faces` continuously
- **Stop condition**: No faces for 120 seconds
  - Executes: `echo "exe ao_stop" | yarp rpc /interactionInterface`
  - Sets `alwayson_active = False`
- **Start condition**: Faces detected while stopped
  - Executes: `echo "exe ao_start" | yarp rpc /interactionInterface`
  - Sets `alwayson_active = True`

**Update Rate**: 1 Hz (every 1.0s)

**State Variables**:
```python
alwayson_active: bool          # Current state (on/off)
last_faces_seen_time: float    # Timestamp of last face detection
NO_FACES_TIMEOUT: 120.0        # 2 minutes
```

#### 5. Proactive Thread (Learning Actions)
**Purpose**: Execute learned social behaviors

**Output Port**: `/alwayson/embodiedbehaviour/experiences:o`

**Available Actions**:
```python
PROACTIVE_ACTIONS = [
    "ao_greet",           # Greeting gesture
    "ao_coffee_break",    # Suggest break
    "ao_curious_lean_in"  # Lean forward with interest
]
```

**Decision Loop**:

```
1. Check if always-on is active
   → Skip cycle if stopped

2. Capture pre-state snapshot (instant from rolling window)
   → Uses 3 seconds of buffered data (60 samples at 20Hz)

3. Check presence conditions:
   ✓ num_faces > 0
   ✓ num_mutual_gaze > 0

4. Check intention thresholds:
   ✓ IIE_mean ≥ 0.5 (THRESH_MEAN)
   ✓ IIE_var < 0.1 (THRESH_VAR)

5. Load latest Q-table from disk
   → Gets most recent learned values

6. Select action (epsilon-greedy):
   • 80% of time (initially): Exploit best Q-value
   • 20% of time: Explore random action
   • Epsilon decays: 0.8 → 0.2 over time

7. Execute action via native YARP RPC:
   yarp.Bottle → /interactionInterface (fire-and-forget)

8. Wait 3.5 seconds for action + reaction + sensor integration
   → Fixed wait ensures complete rolling window refresh

9. Capture post-state snapshot (instant from rolling window)

10. Send experience to Learning module (13 fields)

11. Log to CSV

12. Decay epsilon: ε = max(0.2, ε × 0.957603)

13. Cooldown 5 seconds
```

**Rolling Window Snapshot** (Non-Blocking, Thread-Safe Noise Filtering):
```python
def _windowed_snapshot():
    """Average state from rolling buffer (instant, 0.0s)"""
    # IIE Monitor continuously fills iie_window at 20Hz
    # Window holds last 60 samples (3 seconds of history)
    
    current = _get_state_snapshot()
    
    # Thread-safe copy (prevents race condition with IIE Monitor)
    with _state_lock:
        window_copy = list(iie_window)
    
    if len(window_copy) < 5:
        return current  # Not enough data yet
    
    # Instant averaging of buffered data (safe to iterate)
    iie_means = [s['mean'] for s in window_copy]
    iie_vars = [s['var'] for s in window_copy]
    
    return {
        'IIE_mean': mean(iie_means),    # 3s average (60 samples)
        'IIE_var': mean(iie_vars),      # 3s average (60 samples)
        'ctx': current['ctx'],          # Current
        'num_faces': current['num_faces'],
        'num_mutual_gaze': current['num_mutual_gaze']
    }
```

**Epsilon-Greedy Selection**:
```python
def _select_action_epsilon_greedy(state_key):
    Q_values = Q[state_key]  # e.g., Q["CTX1"]
    
    if random() < epsilon:
        return random_choice(PROACTIVE_ACTIONS)  # Explore
    else:
        return argmax(Q_values)                  # Exploit
```

**Experience Bottle Format** (13 fields):
```
0:  timestamp (float64)
1:  action (string)
2:  pre_ctx (int8)
3:  pre_IIE_mean (float64)
4:  pre_IIE_var (float64)
5:  post_ctx (int8)
6:  post_IIE_mean (float64)
7:  post_IIE_var (float64)
8:  q_value (float64)
9:  pre_num_faces (int32)
10: pre_num_mutual_gaze (int32)
11: post_num_faces (int32)
12: post_num_mutual_gaze (int32)
```

#### 6. Self-Adaptor Thread (Background Behaviors)
**Purpose**: Periodic natural behaviors (non-learned)

**Available Actions**:
```python
SELF_ADAPTORS = [
    "ao_look_around",  # Look around environment
    "ao_yawn",         # Yawn gesture
    "ao_cough"         # Cough gesture
]
```

**Behavior**:
- **Context-dependent timing**:
  - Calm environment: Every 240 seconds (4 minutes)
  - Lively environment: Every 120 seconds (2 minutes)
- **Random selection**: No learning involved
- **Always active**: Runs even when always-on is stopped
- **Purpose**: Make robot seem alive and natural

**Key Difference from Proactive**:
- ✗ No threshold checks
- ✗ No Q-learning
- ✗ No epsilon-greedy
- ✗ No experience sent to Learning
- ✓ Always runs (even when stopped)
- ✓ Context-adaptive timing

---

## Learning Module

### Overview
The critic module that evaluates actions and improves the policy through Q-learning and ML models.

### Core Components

#### 1. Q-Learning (Value Function)

**Algorithm**: Temporal Difference (TD) Q-Learning

**Q-Table Structure**:
```json
{
  "CTX0": {  // Calm context
    "ao_greet": 0.23,
    "ao_coffee_break": 0.15,
    "ao_curious_lean_in": 0.41
  },
  "CTX1": {  // Lively context
    "ao_greet": 0.52,
    "ao_coffee_break": 0.38,
    "ao_curious_lean_in": 0.35
  }
}
```

**TD Update Rule**:
```python
Q(s, a) ← Q(s, a) + α [r + γ · max Q(s', a') - Q(s, a)]
                      ↑       ↑           ↑
                   learn   immediate   best future
                   rate    reward      value
```

**Parameters**:
- `α = 0.30`: Learning rate (how much to update)
- `γ = 0.92`: Discount factor (how much future matters)

**Update Process**:
```python
def _update_q(exp, reward):
    pre_state = f"CTX{exp.pre_ctx}"
    post_state = f"CTX{exp.post_ctx}"
    
    old_q = Q[pre_state][exp.action]
    max_next_q = max(Q[post_state].values())
    
    td_target = reward + 0.92 * max_next_q
    td_error = td_target - old_q
    new_q = old_q + 0.30 * td_error
    
    Q[pre_state][exp.action] = new_q
    return old_q, new_q, td_error
```

**Example Update**:
```
State: CTX1 (Lively)
Action: ao_greet
Old Q: 0.32

Reward received: +0.165
Best future Q (CTX1): 0.45

TD target: 0.165 + 0.92 × 0.45 = 0.579
TD error: 0.579 - 0.32 = 0.259
New Q: 0.32 + 0.30 × 0.259 = 0.398

Result: Q(CTX1, greet) updated 0.32 → 0.40 ✓
```

**Persistence**:
- Saved after **every update** (atomic write)
- Format: JSON with temp+rename for safety
- Loaded by Embodied Behaviour before each action

#### 2. Reward Function

**Components** (4 terms + clipping):

```python
reward = (W_DELTA × delta_mean) + 
         (W_VAR × var_reduction) + 
         (W_LEVEL × level_term) - 
         action_cost

# Clip to [-1.0, 1.0] for stability
reward = max(-1.0, min(1.0, reward))
```

**1. Delta Term** (`W_DELTA = 1.0`):
```python
delta_mean = post_IIE_mean - pre_IIE_mean
if abs(delta_mean) < DELTA_EPS:  # 0.05 dead zone
    delta_mean = 0.0
```
- **Purpose**: Reward increasing engagement
- **Dead zone**: Filters noise ±0.05
- **Example**: 0.65 → 0.75 gives +0.10

**2. Variance Term** (`W_VAR = 0.5`):
```python
var_reduction = pre_IIE_var - post_IIE_var
if abs(var_reduction) < VAR_EPS:  # 0.02 dead zone
    var_reduction = 0.0
```
- **Purpose**: Reward making interaction more predictable
- **Dead zone**: Filters noise ±0.02
- **Example**: 0.08 → 0.05 gives +0.03 (reduction)

**3. Level Term** (`W_LEVEL = 0.5`):
```python
level_term = post_IIE_mean - THRESH_MEAN  # 0.5 threshold
if post_IIE_mean < THRESH_MEAN:
    level_term *= 2.0  # Double penalty!
```
- **Purpose**: Maintain engagement above threshold
- **Positive**: Comfortably above 0.5
- **Negative**: Below 0.5 (2× penalty)
- **Example**: 
  - IIE = 0.67: level_term = +0.17 (good!)
  - IIE = 0.45: level_term = -0.10 (penalty ×2)

**4. Action Cost**:
```python
ACTION_COSTS = {
    "ao_greet": 0.08,
    "ao_coffee_break": 0.10,
    "ao_curious_lean_in": 0.06,
}
```
- **Purpose**: Small penalty for energy expenditure
- Prevents action spam

**Complete Example**:
```python
# Scenario: Successful greeting in lively context
pre_IIE_mean:  0.62
post_IIE_mean: 0.78
pre_IIE_var:   0.08
post_IIE_var:  0.06
action:        "ao_greet"

# Compute components:
delta_mean = 0.78 - 0.62 = 0.16 (✓ above dead zone)
var_reduction = 0.08 - 0.06 = 0.02 (✓ just at dead zone)
level_term = 0.78 - 0.5 = 0.28 (✓ well above threshold)
action_cost = 0.08

# Final reward:
reward = 1.0×0.16 + 0.5×0.02 + 0.5×0.28 - 0.08
       = 0.16 + 0.01 + 0.14 - 0.08
       = +0.23 ✓ Strong positive!

# After clipping (already within [-1.0, 1.0]):
reward = 0.23 (no change)
```

**Reward Philosophy**:
- **Delta**: Reward improvement
- **Variance**: Reward stability
- **Level**: Maintain high engagement (don't break interaction!)
- **Cost**: Discourage excessive actions
- **Clipping**: Bound to [-1.0, 1.0] for numerical stability (prevents outliers)

#### 3. Machine Learning Models

**Purpose**: Predict action outcomes (for generalization)

**Model 1: IIE Transition Model** (Regression)
```python
GradientBoostingRegressor(
    n_estimators=50→200,  # Grows over time
    max_depth=2,
    learning_rate=0.1,
    warm_start=True
)
```

**Input Features** (8D):
```python
[
    pre_IIE_mean,      # Current engagement
    pre_IIE_var,       # Current stability
    pre_ctx,           # Context (0/1)
    pre_num_faces,     # Face count
    pre_num_mutual_gaze,  # Gaze count
    action[0],         # One-hot: greet
    action[1],         # One-hot: coffee_break
    action[2]          # One-hot: curious_lean_in
]
```

**Output**: `Δ IIE_mean` (predicted change)

**Training**:
- Buffer size: 10 samples
- When buffer full: train on batch
- Incremental: adds 5 trees per training
- Max trees: 200

**Model 2: Context Transition Model** (Classification)
```python
GradientBoostingClassifier(
    n_estimators=50→200,
    max_depth=2,
    learning_rate=0.1,
    warm_start=True
)
```

**Input**: Same 8D features

**Output**: `post_ctx` (0 or 1)

**Purpose**: Predict if context will change

#### 4. Experience Processing Pipeline

**Critical Order** (why it matters):

```python
def _process_experience(exp):
    # 1. PREDICT (before training!)
    predicted_delta, predicted_post_mean, predicted_ctx = _get_predictions(exp)
    
    # WHY: Test model on truly unseen data
    # If we trained first, we'd be testing on data we just learned from!
    
    # 2. LOG PREDICTIONS
    _log_predictions(exp, predicted_delta, predicted_post_mean, predicted_ctx)
    
    # WHY: Track model accuracy over time
    
    # 3. Q-LEARNING UPDATE
    reward = _compute_reward(exp)
    old_q, new_q, td_error = _update_q(exp, reward)
    _save_qtable()  # Save immediately!
    
    # WHY: Embodied Behaviour loads Q-table before next action
    
    # 4. TRAIN MODELS
    features = _encode_features(exp)
    target_delta = exp.post_IIE_mean - exp.pre_IIE_mean
    _train_iie_model(features, target_delta)
    _train_ctx_model(features, exp.post_ctx)
    
    # WHY: Improve predictions for future experiences
```

**Analogy**:
- ❌ Wrong: Study for test → Take test (you know answers!)
- ✓ Right: Take test → Study answers (measures real knowledge)

---

## Communication Protocol

### Port Connections

```
STM → Embodied Behaviour:
  /alwayson/stm/info:o → /alwayson/embodiedbehaviour/info:i
  /alwayson/stm/context:o → /alwayson/embodiedbehaviour/context:i
  
IIE Estimator → Embodied Behaviour:
  /iie/estimator:o → /alwayson/embodiedbehaviour/iie:i

Embodied Behaviour → Learning:
  /alwayson/embodiedbehaviour/experiences:o → /alwayson/learning/experiences:i

Robot Interface:
  RPC: /interactionInterface (for action execution)
```

### Experience Bottle (13 Fields)

**Sent by**: Embodied Behaviour  
**Received by**: Learning  
**Frequency**: After each proactive action (~10-20 seconds)

```python
bottle.addFloat64(timestamp)              # 0: When action started
bottle.addString(action)                  # 1: "ao_greet", etc.
bottle.addInt8(pre_ctx)                   # 2: Context before (-1/0/1)
bottle.addFloat64(pre_IIE_mean)           # 3: Engagement before
bottle.addFloat64(pre_IIE_var)            # 4: Variance before
bottle.addInt8(post_ctx)                  # 5: Context after
bottle.addFloat64(post_IIE_mean)          # 6: Engagement after
bottle.addFloat64(post_IIE_var)           # 7: Variance after
bottle.addFloat64(q_value)                # 8: Q-value used
bottle.addInt32(pre_num_faces)            # 9: Faces before
bottle.addInt32(pre_num_mutual_gaze)      # 10: Gaze before
bottle.addInt32(post_num_faces)           # 11: Faces after
bottle.addInt32(post_num_mutual_gaze)     # 12: Gaze after
```

---

## Configuration & Parameters

### Embodied Behaviour

**Action Thresholds**:
```python
THRESH_MEAN = 0.5     # Minimum IIE to act
THRESH_VAR = 0.1      # Maximum variance to act
```

**Timing**:
```python
WAIT_AFTER_ACTION = 3.5          # Seconds for action + reaction + rolling window refresh
COOLDOWN = 5.0                   # Seconds between actions
SELFADAPTOR_PERIOD_CALM = 240.0  # Self-adaptor: calm (4 min)
SELFADAPTOR_PERIOD_LIVELY = 120.0  # Self-adaptor: lively (2 min)
NO_FACES_TIMEOUT = 120.0         # Always-on: stop after (2 min)
```

**Exploration**:
```python
EPSILON = 0.8            # Initial exploration rate
EPSILON_MIN = 0.2        # Minimum exploration rate
EPSILON_DECAY = 0.957603 # Decay per action
```

**Rolling Window** (Non-Blocking Noise Filtering):
```python
WINDOW_SIZE = 60         # Buffer size (3s at 20Hz)
# IIE Monitor feeds buffer continuously
# Snapshot reads instantly from buffer (0.0s latency)
```

**Data Freshness** (Stale Sensor Detection):
```python
IIE_TIMEOUT = 5.0   # Reset to safe defaults after 5s of no updates
INFO_TIMEOUT = 5.0  # Reset to safe defaults after 5s of no updates
# Context has no timeout (deliberate state, not continuous stream)
```

### Learning

**Q-Learning**:
```python
ALPHA = 0.30    # Learning rate
GAMMA = 0.92    # Discount factor
```

**Reward Weights**:
```python
W_DELTA = 1.0   # IIE change weight
W_VAR = 0.5     # Variance reduction weight
W_LEVEL = 0.5   # Engagement level weight
THRESH_MEAN = 0.5  # Threshold for level term
```

**Dead Zones** (Noise Filtering):
```python
DELTA_EPS = 0.05  # Minimum IIE change (±0.05)
VAR_EPS = 0.02    # Minimum variance change (±0.02)
```

**ML Models**:
```python
BUFFER_SIZE = 10           # Training batch size
MODEL_MAX_DEPTH = 2        # Tree depth
MODEL_N_ESTIMATORS = 50    # Initial trees
MAX_ESTIMATORS = 200       # Maximum trees
MODEL_LEARNING_RATE = 0.1  # Boosting learning rate
```

**Action Costs**:
```python
ACTION_COSTS = {
    "ao_greet": 0.08,
    "ao_coffee_break": 0.10,
    "ao_curious_lean_in": 0.06,
}
```

---

## Data Logging

### Embodied Behaviour Logs

#### 1. Proactive Log (`proactive_log.csv`)
**Frequency**: Every proactive action  
**Fields** (14 columns):
```csv
timestamp, proactive_action,
pre_IIE_mean, pre_IIE_var, pre_ctx, pre_num_faces, pre_num_mutual_gaze,
post_IIE_mean, post_IIE_var, post_ctx, post_num_faces, post_num_mutual_gaze,
q_value, epsilon
```

**Purpose**: Track action selection and outcomes

#### 2. Self-Adaptor Log (`selfadaptor_log.csv`)
**Frequency**: Every self-adaptor action  
**Fields** (8 columns):
```csv
timestamp, self_adaptor_action,
pre_IIE_mean, pre_IIE_var, pre_ctx,
post_IIE_mean, post_IIE_var, post_ctx
```

**Purpose**: Track background behaviors

### Learning Logs

#### 1. Q-Learning Log (`qlearning_log.csv`)
**Frequency**: Every Q-update  
**Fields** (14 columns):
```csv
timestamp, proactive_action, pre_state, post_state,
pre_IIE_mean, post_IIE_mean, delta_mean,
pre_IIE_var, post_IIE_var, var_reduction,
reward, old_q, new_q, td_error
```

**Purpose**: Track reward and Q-value evolution

**Key Metrics**:
- `reward`: Total reward received
- `td_error`: Prediction error (how surprising was outcome?)
- `new_q - old_q`: How much Q-value changed

#### 2. Model Prediction Log (`model_prediction_log.csv`)
**Frequency**: Every experience  
**Fields** (14 columns):
```csv
timestamp, proactive_action,
pre_IIE_mean, pre_IIE_var, pre_ctx, pre_num_faces, pre_num_mutual_gaze,
predicted_IIE_delta, predicted_post_IIE_mean, predicted_post_ctx,
actual_post_IIE_mean, actual_post_ctx,
iie_prediction_error, ctx_prediction_correct
```

**Purpose**: Track ML model accuracy

**Key Metrics**:
- `iie_prediction_error`: |predicted - actual|
- `ctx_prediction_correct`: 1 if correct, 0 if wrong

#### 3. Model Training Log (`model_training_log.csv`)
**Frequency**: Every experience  
**Fields** (11 columns):
```csv
timestamp, proactive_action,
pre_IIE_mean, pre_IIE_var, pre_ctx, pre_num_faces, pre_num_mutual_gaze,
target_delta_IIE, predicted_delta_IIE,
prediction_error, model_training_count
```

**Purpose**: Track training samples

---

## Reward Function Design

### Design Philosophy

**Goal**: Learn to maintain high engagement while avoiding interaction breakdowns

**Challenges**:
1. **Noise**: IIE measurements flicker ±0.03
2. **Threshold**: Must stay above 0.5 to act
3. **Sparse signal**: Changes are often small

**Solutions**:

#### 1. Dead Zones (Noise Filtering)
```python
if abs(delta_mean) < 0.05:
    delta_mean = 0.0  # Ignore tiny changes
```

**Benefit**: Random jitter doesn't create false rewards

#### 2. Windowed Averaging (Noise Filtering)
```python
pre = average_over_3_seconds()   # 30 samples
post = average_over_3_seconds()  # 30 samples
```

**Benefit**: Pre/post measurements are highly stable (~30× noise reduction)

#### 3. Level Term (Threshold Awareness)
```python
level_term = post_IIE_mean - 0.5
if post_IIE_mean < 0.5:
    level_term *= 2.0  # Penalty!
```

**Benefit**: 
- Don't break engagement (strong penalty)
- Maintain high IIE (mild reward)

### Reward Examples

**Scenario 1: Maintain engagement**
```python
Pre:  IIE = 0.65, var = 0.08  # From 3s rolling buffer average
Post: IIE = 0.67, var = 0.06  # From 3s rolling buffer average
Action: ao_greet

delta_mean = 0.02 → 0.0 (dead zone)
var_reduction = 0.02 → 0.02 (at threshold)
level_term = 0.67 - 0.5 = 0.17

reward = 0.0 + 0.5×0.02 + 0.5×0.17 - 0.08
       = 0.0 + 0.01 + 0.085 - 0.08
       = +0.015 (slightly positive!)
# Clipped: 0.015 (within bounds)
```
✓ Maintaining high engagement is rewarded

**Scenario 2: Break engagement**
```python
Pre:  IIE = 0.55, var = 0.07
Post: IIE = 0.45, var = 0.09
Action: ao_coffee_break

delta_mean = -0.10
var_reduction = -0.02
level_term = (0.45 - 0.5) × 2.0 = -0.10 (×2 penalty!)

reward = 1.0×(-0.10) + 0.5×(-0.02) + 0.5×(-0.10) - 0.10
       = -0.10 - 0.01 - 0.05 - 0.10
       = -0.26 (strong penalty!)
# Clipped: -0.26 (within bounds)
```
✗ Breaking threshold is heavily penalized

**Scenario 3: Boost engagement**
```python
Pre:  IIE = 0.60, var = 0.08
Post: IIE = 0.78, var = 0.05
Action: ao_curious_lean_in

delta_mean = 0.18
var_reduction = 0.03
level_term = 0.78 - 0.5 = 0.28

reward = 1.0×0.18 + 0.5×0.03 + 0.5×0.28 - 0.06
       = 0.18 + 0.015 + 0.14 - 0.06
       = +0.275 (excellent!)
# Clipped: 0.275 (within bounds)
```
✓ Large improvement + high level = big reward

---

## Learning Progress & Convergence

### Phases of Learning

#### Phase 1: Exploration (Episodes 1-50)
**Epsilon**: 0.8 → 0.6

**Behavior**:
- 80% random actions (exploring)
- 20% best known actions (exploiting)
- Building initial Q-table
- High variance in outcomes

**Q-Table Example** (after 20 actions):
```json
{
  "CTX0": {
    "ao_greet": 0.12,
    "ao_coffee_break": -0.05,  // Doesn't work well in calm
    "ao_curious_lean_in": 0.28
  },
  "CTX1": {
    "ao_greet": 0.35,
    "ao_coffee_break": 0.18,
    "ao_curious_lean_in": 0.22
  }
}
```

**ML Models**:
- IIE Model: Not trained yet (need 10 samples)
- Context Model: Not trained yet
- Prediction error: N/A

#### Phase 2: Refinement (Episodes 51-150)
**Epsilon**: 0.6 → 0.3

**Behavior**:
- 40% random (still exploring)
- 60% best known (mostly exploiting)
- Q-values converging
- Patterns emerging

**Q-Table Example** (after 100 actions):
```json
{
  "CTX0": {
    "ao_greet": 0.23,
    "ao_coffee_break": 0.15,
    "ao_curious_lean_in": 0.41  // Clear winner in calm
  },
  "CTX1": {
    "ao_greet": 0.52,           // Clear winner in lively
    "ao_coffee_break": 0.38,
    "ao_curious_lean_in": 0.35
  }
}
```

**ML Models**:
- IIE Model: Trained 10+ times, 100+ trees
- Prediction error: 0.15 → 0.08 (improving!)
- Context Model: 60% → 75% accuracy

#### Phase 3: Convergence (Episodes 150+)
**Epsilon**: 0.3 → 0.2

**Behavior**:
- 20% random (minimal exploration)
- 80% best known (mostly exploiting)
- Q-values stable
- Confident action selection

**Q-Table Example** (after 300 actions):
```json
{
  "CTX0": {
    "ao_greet": 0.25,
    "ao_coffee_break": 0.18,
    "ao_curious_lean_in": 0.47  // Stable best
  },
  "CTX1": {
    "ao_greet": 0.56,           // Stable best
    "ao_coffee_break": 0.41,
    "ao_curious_lean_in": 0.38
  }
}
```

**ML Models**:
- IIE Model: 200 trees, fully trained
- Prediction error: 0.04 (excellent!)
- Context Model: 85% accuracy

### Convergence Indicators

**Q-values stabilizing**:
```python
# Watch in qlearning_log.csv
td_error → 0 (predictions match reality)
new_q - old_q → 0 (no more updates)
```

**Prediction accuracy improving**:
```python
# Watch in model_prediction_log.csv
iie_prediction_error: 0.20 → 0.15 → 0.08 → 0.04
ctx_prediction_correct: 60% → 75% → 85%
```

**Reward increasing**:
```python
# Watch in qlearning_log.csv
mean(reward) increasing over time
variance(reward) decreasing over time
```

**Epsilon decaying**:
```python
# Watch in proactive_log.csv
epsilon: 0.8 → 0.6 → 0.4 → 0.2
```

### Expected Timeline

**Quick learning**: 20-30 actions
- Initial patterns emerge
- Q-values start differentiating

**Competent**: 50-100 actions
- Reliable action selection
- Context-appropriate behaviors

**Expert**: 150-300 actions
- Near-optimal policy
- Confident predictions

---

## Troubleshooting

### Common Issues

#### 1. No Actions Executed
**Symptoms**: Proactive thread never acts

**Checks**:
```python
# Check thresholds:
IIE_mean ≥ 0.5?
IIE_var < 0.1?
num_faces > 0?
num_mutual_gaze > 0?
alwayson_active == True?
```

**Solutions**:
- Lower thresholds temporarily
- Check IIE estimator output
- Verify STM info bottle format

#### 2. Q-Values Not Updating
**Symptoms**: Q-table stays at 0.0

**Checks**:
```python
# In Learning module:
exp.pre_ctx != -1?
exp.post_ctx != -1?
reward != 0.0?
```

**Solutions**:
- Check experience bottle format
- Verify context is being detected
- Check reward components

#### 3. High Prediction Errors
**Symptoms**: ML models not learning

**Checks**:
```python
# In model_prediction_log.csv:
prediction_error > 0.20 after 100 samples?
```

**Solutions**:
- Check feature encoding
- Verify StandardScaler is fitted
- Increase buffer size or tree count

#### 4. Always-On Stuck
**Symptoms**: Won't start/stop

**Checks**:
```python
# Check RPC interface:
yarp rpc /interactionInterface
exe ao_start  # Manual test
```

**Solutions**:
- Verify interaction interface running
- Check RPC command format
- Review always-on monitor logs

---

## Appendix: Mathematical Notation

### Q-Learning
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                      a'

Where:
  Q(s,a): Value of action a in state s
  α: Learning rate (0.30)
  r: Reward received
  γ: Discount factor (0.92)
  s': Next state
  max Q(s',a'): Best future value
     a'
```

### Reward Function
```
R_raw = W_Δ·Δμ + W_σ·Δσ² + W_L·L - C

R = clip(R_raw, -1.0, 1.0)  # Bounded for stability

Where:
  Δμ = {μ_post - μ_pre  if |Δμ| ≥ 0.05
       {0                otherwise
  
  Δσ² = {σ²_pre - σ²_post  if |Δσ²| ≥ 0.02
        {0                  otherwise
  
  L = {(μ_post - 0.5)      if μ_post ≥ 0.5
      {2(μ_post - 0.5)      if μ_post < 0.5
  
  C = action cost
  
  W_Δ = 1.0, W_σ = 0.5, W_L = 0.5
```

### Epsilon-Greedy
```
π(s) = {argmax Q(s,a)    with probability 1-ε
       {random action     with probability ε
        a

ε_t = max(ε_min, ε_0 · λ^t)

Where:
  ε_0 = 0.8 (initial)
  ε_min = 0.2 (minimum)
  λ = 0.957603 (decay rate)
  t = action count
```
