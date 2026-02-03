# Proactive Social Robot Test Harness

A comprehensive testing framework for a proactive social robot behavior system that uses behavior trees and reinforcement learning to interact naturally with people.

## Overview

This project implements a YARP RFModule (`embodied_behaviour.py`) that switches between **INACTIVE** and **ACTIVE** states based on face detection, and when active, uses context-aware Q-learning to dynamically choose and execute one of two behavior branches:

- **Low Proactivity (LP)** - Engages when mutual gaze detected (close face)
- **High Proactivity (HP)** - More proactive, greets even without mutual gaze

The system:
- Detects and recognizes people through face detection
- Responds to human actions (waving, eating, phone use, etc.)
- Adapts behavior based on social context (calm vs. lively environments)
- Learns optimal interaction strategies through Q-learning with valence/arousal feedback
- Manages greetings and maintains interaction history
- Logs all interactions for analysis and reproducibility

## System Components

- **`embodied_behaviour.py`** - Main YARP RFModule with state machine, port readers, and BT orchestration
- **`behaviour_trees.py`** - Low Proactivity (LP) and High Proactivity (HP) behavior tree implementations using py_trees
- **`learning.py`** - Q-learning library for epsilon-greedy branch selection and reward-based updates
- **`test_harness.py`** - Comprehensive test suite that simulates YARP perception ports and validates all scenarios
- **`mock_context_publisher.py`** - Standalone mock for publishing context messages during testing
- **`run_complete_test.sh`** - Automated test execution script with multiple modes

## How It Works
### State Machine: ACTIVE ↔ INACTIVE

The module operates in two states:

- **INACTIVE**: No faces detected for 60 seconds
  - Executes `ao_stop` command
  - No behavior trees running
  - Continuously monitors for faces

- **ACTIVE**: At least one face detected
  - Executes `ao_start` command on entry
  - Enables Q-learning tree selection
  - Runs behavior trees based on context

Transition logic runs in the main loop (0.1s period):
- ACTIVE → INACTIVE: When `faces_count == 0` for ≥ 60 seconds
- INACTIVE → ACTIVE: When `faces_count >= 1`

### Behavior Trees

The system dynamically selects between two behavior modes:

#### Low Proactivity (LP) Tree
**Activation Requirements:**
- At least one **close face** (bounding box area ≥ 15,000 pixels)
- At least one **known person** visible

**Behavior:**
1. Selects target by biggest face box among known persons
2. If target **not greeted today** → Execute greet+wave ("Ciao {name}" + `ao_wave`)
3. If target **already greeted** → Wait up to 5 seconds for an action:
   - Action detected → Execute mapped response (see Action Responses)
   - No action → Enter cooldown (3s for calm, 1.5s for lively)

#### High Proactivity (HP) Tree
**Activation Requirements:**
- At least one known person visible (no close face required)

**Behavior:**
1. If **no known person** → Execute wave only (`ao_wave`)
2. If **known person exists**:
   - Select target by biggest box
   - If not greeted today → Execute greet+wave
   - If already greeted → Wait 5s for action, respond or cooldown

### Action → Response Mapping

When a person's action is detected, the robot responds appropriately:

| Detected Action | Response Type | Response Value |
|----------------|---------------|----------------|
| answer phone | speak | "Salutalo da parte mia" |
| carry/hold (an object) | speak | "Tieni forte" |
| drink | speak | "Salute" |
| eat | speak | "Buon appetito" |
| text on/look at a cellphone | animation | `ao_yawn_phone` |
| hand wave | animation | `ao_wave` |

### Q-Learning System

The system uses **epsilon-greedy Q-learning** to decide which behavior tree (LP or HP) to run based on social context and interaction outcomes.

**Q-Table Structure:**
### YARP Ports and Threading

The module opens **four input ports** with dedicated daemon threads for concurrent perception processing:

| Port Name | Data Format | Thread | Purpose |
|-----------|-------------|--------|---------|
| `/embodiedBehaviour/context:i` | Bottle: [label, confidence, timestamp] | `_context_thread` | Social context: -1=uncertain, 0=calm, 1=lively |
| `/embodiedBehaviour/valence_arousal:i` | Nested face bottles with V/A/box | `_va_thread` | Valence/arousal per detected face |
| `/embodiedBehaviour/face_id:i` | Nested face bottles with name/box | `_face_id_thread` | Face recognition (name, confidence, bounding box) |
| `/embodiedBehaviour/actions:i` | Nested person bottles with actions | `_action_thread` | Action recognition events with timestamps |

All threads update **shared state variables** protected by a `threading.Lock`:
- `context_label` - Current social context
- `faces_all`, `known_faces`, `faces_count` - Face tracking
- `detected_actions` - Recent action detections by person
- `_va_samples` - Valence/arousal buffer during action execution

### VA Capture and Reward Computation

When a behavior tree action executes:

1. **Start Capture**: `start_va_capture()` enables buffering of incoming VA samples
2. **Execute Action**: Run subprocess commands (YARP write/rpc for speech/animations)
3. **Stop Capture**: `stop_va_capture_get_peak()` returns the VA sample with **highest reward** during execution
4. **Compute Reward**: 
   ```python
   reward = alpha * valence + beta * arousal
   reward = clamp(reward, -1.0, 1.0)
   ```
5. **Update Q-table**: Apply learning update and decay epsilon

The VA parser selects the "best face" per message by computing the reward for each face and choosing the maximum, ensuring learning focuses on the most emotionally expressive person.

### Mock Services (Test Harness)

Mock RPC services simulate robot outputs:
- `/interactionInterface` - Animation commands (`ao_wave`, `ao_yawn_phone`, `ao_start`, `ao_stop`)
- `/acapelaSpeak/speech:i` - Speech synthesis (Italian phrases)
1. Wait if context is uncertain (`-1`)
2. Load Q-table from `q_table.json`
3. With probability `epsilon`: choose randomly (exploration)
4. With probability `1-epsilon`: choose branch with highest Q-value (exploitation)
5. On ties: random selection for fair exploration

**Learning Update:**
After each action completes:
```
reward = alpha * valence + beta * arousal  (clamped to [-1, 1])
Q ← Q + eta * (reward - Q)
epsilon ← max(0.2, epsilon - 0.01)
```

Default hyperparameters:
- `alpha = 1.0` (valence weight)
- `beta = 0.3` (arousal weight)
- `eta = 0.1` (learning rate)
- `epsilon` decays from 0.8 to 0.2 minimum
The system uses Q-learning to decide which behavior tree to activate based on the social context and interaction outcomes (measured by valence and arousal).

## Quick Start

### Prerequisites
- Python 3
- YARP (Yet Another Robot Platform)
- py_trees library

### Running Tests

**Standard 2-minute test:**
```bash
./run_complete_test.sh
```maintains comprehensive logs for experiment analysis:

### `q_table.json`
Q-values for LP/HP branches in calm/lively contexts, plus current epsilon value.

### `last_greeted.db`
**Schema:**
```sql
CREATE TABLE last_greeted (
    name TEXT PRIMARY KEY,
    timestamp REAL,
    date TEXT
);
```
- Tracks when each person was last greeted
- Used to determine "greeted today" status
- Updated after successful greet actions

### `data_collection.db`
**Four tables for comprehensive logging:**

#### 1. `events` - State Transitions & Tree Selections
```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    event_type TEXT,  -- 'transition' or 'selection'
    from_state TEXT,
    to_state TEXT,
    context_str TEXT,
    branch TEXT,
    q_lp REAL,
    q_hp REAL,
    epsilon REAL
);
```

#### 2. `actions` - Executed Actions
```sql
CREATE TABLE actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start REAL,
    ts_end REAL,
    action_type TEXT,        -- 'greet_wave', 'response', 'wave_only', etc.
    response_type TEXT,      -- 'speak', 'ao', 'speak+ao'
    response_value TEXT,     -- Actual phrase or command
    trigger_reason TEXT,     -- 'not_seen_today', 'action_detected', etc.
    target_name TEXT,
### Key Parameters in `embodied_behaviour.py`

**State Machine:**
- `INACTIVE_TIMEOUT = 60.0` - Seconds with no faces before entering INACTIVE state
- `ACTION_WAIT_TIMEOUT = 5.0` - Seconds to wait for person's action in behavior trees
- `CONTEXT_WAIT_TIMEOUT = 5.0` - Seconds to wait when context is uncertain

**Perception:**
- `MIN_CLOSE_FACE_AREA = 15000.0` - Minimum bounding box area (pixels²) for "close face" detection
- `PERCEPTION_LOG_INTERVAL = 1.0` - Seconds between perception snapshots logged to database

**Learning:**
- `ALPHA_REWARD = 1.0` - Valence weight in reward computation
- `BETA_REWARD = 0.3` - Arousal weight in reward computation
- `ETA_LEARNING = 0.1` - Learning rate (η) for Q-table updates

**Action Whitelist:**
```python
ALLOWED_ACTIONS = {
    "answer phone", "carry/hold (an object)", "drink",
    "eat", "text on/look at a cellphone", "hand wave"
}
```

### Command-line Arguments

```bash
python embodied_behaviour.py [OPTIONS]

Options:
  --q_file PATH      Path to Q-table JSON file (default: q_table.json)
  --db_file PATH     Path to last_greeted database (default: last_greeted.db)
  --data_db PATH     Path to data collection database (default: data_collection.db)
  --seed INT         RNG seed for reproducible behavior (optional)
```

### Behavior Tree Cooldowns

After executing an action, each tree enters a cooldown before restarting:
- **LP Tree**: 3.0s in calm context, 1.5s in lively context
- **HP Tree**: 3.0s in calm context, 1.5s in lively context

This prevents rapid repeated interactions with the same person.
```

#### 3. `affect` - Valence/Arousal Summary
```sql
CREATE TABLE affect (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id INTEGER,
    ts_start REAL,
    num_samples INTEGER,
    peak_valence REAL,
    peak_arousal REAL,
    reward REAL,
    used_for_learning INTEGER  -- 0 or 1
);
```

#### 4. `perception` - Periodic Snapshots
```sql
CREATE TABLE perception (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    faces_count INTEGER,
    context_label INTEGER  -- -1, 0, or 1
);
```
Logged every 1 second to track environment state.
./run_complete_test.sh --clean 5
```

**Interactive mode:**
```bash
./run_complete_test.sh --interactive
```

**Run module in separate terminal for visibility:**
```bash
./run_complete_test.sh --separate 10
```

## Test Scenarios

The test harness validates:
1. New person greetings with mutual gaze
2. Action-based responses (waving, eating, etc.)
3. Context-aware behavior selection
4. State transitions (ACTIVE ↔ INACTIVE)
5. Q-learning updates and convergence
6. Multi-person target selection

## Data Collection

The system logs all interactions to SQLite databases:
- `q_table.json` - Learned action values
- `last_greeted.db` - Greeting history
- `data_collection.db` - Detailed interaction logs

## Architecture

The system uses YARP ports for perception data:
- `/embodiedBehaviour/context:i` - Social context (calm/lively)
- `/embodiedBehaviour/valence_arousal:i` - Emotional feedback
- `/embodiedBehaviour/face_id:i` - Face detection and recognition
- `/embodiedBehaviour/actions:i` - Detected human actions

Mock RPC services simulate robot outputs:
- `/interactionInterface` - Animation commands
- `/acapelaSpeak/speech:i` - Speech synthesis

## Configuration

Key parameters in `embodied_behaviour.py`:
- `INACTIVE_TIMEOUT = 60.0` - Seconds before entering inactive state
- `ALPHA_REWARD = 1.0` - Valence weight in reward function
- `BETA_REWARD = 0.3` - Arousal weight in reward function
- `ETA_LEARNING = 0.1` - Q-learning rate

---

**Note**: This is a test harness for a YARP-based social robot system. The actual robot hardware/simulation is not included in this repository.
