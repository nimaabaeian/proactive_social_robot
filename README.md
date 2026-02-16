# Proactive Social Robot - Architecture Summary

## System Overview

This system implements a **proactive social robot** using the iCub humanoid robot platform. The robot autonomously detects people, analyzes their social context, and initiates appropriate interactions based on a sophisticated state machine that tracks social relationships and learning progress.

### Core Philosophy

- **Proactive Engagement**: Robot initiates interactions rather than waiting for user commands
- **Social State Tracking**: Maintains memory of past interactions (greeted, talked, known/unknown)
- **Adaptive Learning**: Progressive spatial constraints (Learning States LS1-LS4) ensure quality interactions
- **Context-Aware**: Uses environmental context (calm/lively) to inform interaction strategies

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         YARP Network                             │
└─────────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌────────────┐      ┌──────────────┐    ┌─────────────────┐
    │ perception │      │ faceSelector │    │ interaction     │
    │  (Vision)  │─────▶│  (Decision)  │───▶│   Manager       │
    └────────────┘      └──────────────┘    │  (Execution)    │
           │                    │            └─────────────────┘
           │                    │                     │
           ▼                    ▼                     ▼
    ┌────────────┐      ┌──────────────┐    ┌─────────────────┐
    │ MediaPipe  │      │ JSON State   │    │  LLM (Ollama)   │
    │ Face Mesh  │      │   Storage    │    │   Phi-3 mini    │
    └────────────┘      └──────────────┘    └─────────────────┘
```

---

## Component Breakdown

### 1. **perception.py** - Vision Analyzer Module

> **Note**: This module is part of the external vision system and is not included in this repository. It provides the landmark data consumed by faceSelector.

**Purpose**: Real-time face detection, tracking, landmark extraction, and gaze analysis.

**Key Technologies**:
- MediaPipe Face Mesh (468 landmarks per face)
- OpenCV for image processing
- YARP for robotics middleware

**Responsibilities**:
- Read RGB images from robot cameras
- Receive face detections from object recognition system (bbox, face_id, track_id)
- Extract 3D head pose (pitch, yaw, roll) using PnP algorithm
- Compute attention state: `MUTUAL_GAZE`, `NEAR_GAZE`, `AWAY`
- Detect talking behavior via mouth motion analysis (lip distance variance)
- Compute spatial zones: `FAR_LEFT`, `LEFT`, `CENTER`, `RIGHT`, `FAR_RIGHT`
- Compute distance categories: `SO_CLOSE`, `CLOSE`, `FAR`, `VERY_FAR`
- Track time-in-view for each person

**Output Format** (per face):
```python
{
    "face_id": "PersonName" or "12345",  # Known name or temporary 5-digit code
    "track_id": 42,                       # Temporal tracking ID
    "bbox": (x, y, w, h),                 # Bounding box
    "zone": "CENTER",                     # Spatial zone
    "distance": "CLOSE",                  # Distance category
    "gaze_direction": (dx, dy, dz),       # 3D gaze vector
    "pitch": 10.5, "yaw": -5.2, "roll": 0.3,
    "cos_angle": 0.92,                    # Face-camera alignment
    "attention": "MUTUAL_GAZE",           # Attention state
    "is_talking": 1,                      # Boolean (0 or 1)
    "time_in_view": 12.5                  # Seconds since first seen
}
```

**YARP Ports**:
- Input: `/alwayson/vision/img:i` - Raw camera feed
- Input: `/alwayson/vision/recognition:i` - Object recognition results
- Output: `/alwayson/vision/landmarks:o` - Detailed face analysis

> **Note**: The actual input port used by faceSelector is `/alwayson/vision/img:o` (output from vision system)

---

### 2. **faceSelector.py** - Face Selection & Decision Module

**Purpose**: Compute social/learning states for all detected faces, select the best interaction candidate, and trigger interactions.

**State Machine - Social States (SS)**:

| State | Description | Conditions |
|-------|-------------|------------|
| **SS1** | Unknown, Not Greeted | First encounter, no greeting yet |
| **SS2** | Unknown, Greeted | Temporary face code assigned, greeted but no name |
| **SS3** | Known, Not Greeted | Person recognized (has name), not greeted today |
| **SS4** | Known, Greeted, Not Talked | Greeted today but no conversation yet |
| **SS5** | Known, Greeted, Talked | Full interaction completed today |

**Learning States (LS)** - Progressive Spatial Constraints:

| State | Zone | Distance | Attention | Purpose |
|-------|------|----------|-----------|---------|
| **LS1** | Any | Any | Any | Initial exploration |
| **LS2** | L/C/R | SO_CLOSE/CLOSE/FAR | Any | Zone filtering |
| **LS3** | L/C/R | SO_CLOSE/CLOSE | MUTUAL/NEAR | Proximity focus |
| **LS4** | L/C/R | SO_CLOSE/CLOSE | MUTUAL | Full engagement |

**Selection Algorithm**:
1. Filter to eligible faces (spatial constraints met, not SS5*)
2. Sort by priority:
   - **Social State** (SS1 > SS2 > SS3 > SS4 > SS5)
   - **Attention** (MUTUAL_GAZE > NEAR_GAZE > AWAY)
   - **Distance** (SO_CLOSE > CLOSE > FAR > VERY_FAR)
   - **Time in view** (longer = better, tie-breaker)
3. Select best candidate and trigger interaction

**Persistent Storage** (JSON files):
- `learning.json`: `{person_id: {ls: 2, updated_at: "2026-02-16T10:30:00"}}`
- `greeted_today.json`: `{person_id: "2026-02-16T09:15:00"}`
- `talked_today.json`: `{person_id: "2026-02-16T10:45:00"}`

**YARP Ports**:
- Input: `/faceSelector/landmarks:i` - From perception module
- Input: `/faceSelector/img:i` - For annotated visualization
- Output: `/faceSelector/img:o` - Annotated video with state labels
- Output: `/faceSelector/debug:o` - Debug information bottle
- RPC Client: `/faceSelector/interactionManager:rpc` - Trigger interactions
- RPC Client: `/faceSelector/interactionInterface:rpc` - Robot behaviors

**Interaction Trigger**:
- Runs in background thread (non-blocking)
- Sends RPC: `run <track_id> <face_id> <ss1|ss2|ss3|ss4>`
- Processes result to update learning state based on interaction quality
- RPC ports auto-connect during module configuration

**Configuration Flags**:
- `--allow_ss5 <true/false>` - Allow SS5 faces to be selected (default: false)
- `--verbose <true/false>` - Enable verbose DEBUG logging (default: false)
- `--rate <seconds>` - Update period in seconds (default: 0.05 = 20 Hz)

**Thread Safety**:
- Uses `state_lock` for thread-safe access to shared state
- Interaction runs in background thread without blocking face detection

**Timezone Configuration**:
- Default timezone: `Europe/Rome`
- Automatic daily reset at midnight (timezone-aware)
- "Today" computed using configured timezone

---

### 3. **interactionManager.py** - Interaction Execution Module

**Purpose**: Execute social interaction state trees using natural language understanding and generation.

**State Trees**:

```
SS1 (Initial Greeting)
├─ Register face with 5-digit code
├─ Execute greeting behavior
├─ Say greeting phrase
├─ Wait for response (LLM detection)
└─ → SS2 on success

SS2 (Name Acquisition)  
├─ Ask "What is your name?" (LLM generated)
├─ Wait for speech-to-text response
├─ Extract name using LLM
├─ Rename face file from code to name
└─ → SS4 on success

SS3 (Known Person Greeting)
├─ Say "Hello, [Name]!"
├─ Wait for response
└─ → SS4 on response

SS4 (Conversation)
├─ Generate conversation starter (LLM)
├─ Loop: Listen → Generate response (LLM) → Speak
├─ Max 5 turns or 120 seconds
└─ → SS5 on successful exchange
```

**Key Technologies**:
- **Ollama LLM** (Phi-3 mini, quantized): Local inference for NLU/NLG
- **SQLite**: Persistent logging of all interactions
- **YARP RPC**: Integration with robot modules

**LLM Prompts**:
- Greeting response detection: `{responded: bool, confidence: float}`
- Name extraction: `{answered: bool, name: str|null, confidence: float}`
- Conversation generation: Context-aware followups

**Interaction Scoring** (for learning state updates):
- **SS1**: response_detected=+2, greet_successful=+1, else=-2
- **SS2**: name extracted (confidence≥0.7)=+3, name only=+1, attempts≥2=-2, else=-3
- **SS3**: response_detected=+2, else=-2
- **SS4**: turns≥5=+4, turns≥3=+2, turns≥1=0, else=-4
- **Global failure penalty**: -3
- **Thresholds**: delta ≥+4 → upgrade LS, delta ≤-4 → downgrade LS

**Error Recovery**:
- LLM requests: 3 retry attempts with 1-second delay
- Fallback JSON logging if SQLite database fails
- TTS bookmark fallback: tries both `mkr` and `mrk` tags

**Thread Safety**:
- Uses `run_lock` to prevent concurrent interaction execution
- Only one interaction runs at a time

**YARP Ports**:
- Input: `/interactionManager/context:i` - Environment context (calm/lively)
- Input: `/interactionManager/landmarks:i` - Real-time face tracking
- Input: `/interactionManager/stt:i` - Speech recognition results
- Input: `/interactionManager/acapela_bookmark:i` - TTS synchronization
- Output: `/interactionManager/speech:o` - Text-to-speech commands
- RPC Server: `/interactionManager` - Receives interaction requests

**Database Schema**:
```sql
CREATE TABLE interactions (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    track_id INTEGER,
    face_id TEXT,
    initial_state TEXT,
    final_state TEXT,
    success INTEGER,
    context_label TEXT,
    result_json TEXT
)
```
---

### 4. **mock_context_publisher.py** - Context Simulation Module

**Purpose**: Simulates Short-Term Memory (STM) context for testing without full cognitive architecture.

**Context Labels**:
- `-1` = UNCERTAIN: Ambiguous environment
- `0` = CALM: Low activity, suitable for extended interactions
- `1` = LIVELY: High activity, keep interactions brief

**Behavior**:
- Publishes every 5 seconds
- Randomly changes context every 3 minutes
- Output format: `(episode_id, chunk_id, context_label)`

---

## Data Flow

### 1. Perception Pipeline
```
Camera → perception.py → MediaPipe Face Mesh
                       → Object Recognition matching
                       → Spatial/Attention/Talking analysis
                       → YARP Bottle output
```

### 2. Decision Pipeline
```
Landmarks → faceSelector.py → Social State computation
                             → Learning State lookup
                             → Eligibility filtering
                             → Priority sorting
                             → Best candidate selection
                             → RPC trigger (if not busy)
```

### 3. Interaction Pipeline
```
RPC Request → interactionManager.py → State tree execution
                                     → LLM inference
                                     → YARP speech/behavior commands
                                     → Result processing
                                     → Database logging
                                     → RPC response
```

### 4. Feedback Loop
```
Interaction Result → faceSelector.py → Score computation
                                      → Learning state update
                                      → JSON file persistence
```

---

## Key Design Patterns

### 1. **Stateful Session Management**
- Track IDs map to stable person IDs
- Daily reset of greeting/talking flags (timezone-aware)
- Persistent learning progression across sessions

### 2. **Non-Blocking Interaction**
- faceSelector continues running during interactions
- Background thread for interaction execution
- Visual feedback (green box) maintained for selected target

### 3. **Graceful Degradation**
- Timeouts for all blocking operations (TTS, STT, LLM)
- Fallback mechanisms (e.g., bookmark waiting)
- Error handling with detailed logging

### 4. **One-to-One Matching**
- Each MediaPipe face matched to at most one object recognition bbox
- Distance-based matching with area tie-breaking
- Prevents duplicate processing

### 5. **Prompt Engineering for Reliability**
- Structured JSON outputs from LLM
- Explicit schemas in prompts
- Robust parsing with fallbacks

---

## Configuration & Dependencies

### Python Dependencies
```
yarp-python         # Robotics middleware
opencv-python       # Computer vision
mediapipe          # Face landmark detection
numpy              # Numerical operations
ollama             # LLM inference
sqlite3            # Database (stdlib)
```

### External Services
- **YARP Network**: Must be running (`yarp server`)
- **Ollama**: Must have `phi3:mini` model downloaded
- **Object Recognition Module**: Provides face detection/tracking
- **Acapela TTS**: Text-to-speech synthesis
- **Speech2Text**: Speech recognition module

### Hardcoded System Paths
- **Face storage directory**: `/usr/local/src/robot/cognitiveInteraction/objectRecognition/modules/objectRecognition/faces`
- **Ollama URL**: `http://localhost:11434`
- Ensure these paths exist or update code accordingly

### File System
```
proactive_social_robot/
├── faceSelector.py            # Face selection and decision logic
├── interactionManager.py      # Interaction state machine execution
├── mock_context_publisher.py  # Context simulation for testing
├── README.md                  # This document
├── learning.json              # Generated at runtime (learning states)
├── greeted_today.json         # Generated at runtime (daily greetings)
├── talked_today.json          # Generated at runtime (daily conversations)
├── interaction_data.db        # Generated at runtime (SQLite log)
├── interaction_fallback.json  # Generated on DB failure (fallback log)
└── last_greeted.json          # Generated at runtime (track→face mapping)
```

> **Note**: `perception.py` is part of the external vision system and not included in this repository.

---

## Port Connection Diagram

```
Camera Feed
    └─▶ /alwayson/vision/img:i (perception.py - external)

Object Recognition
    └─▶ /alwayson/vision/recognition:i (perception.py - external)

perception.py (external)
    ├─▶ /alwayson/vision/landmarks:o
    │    ├─▶ /faceSelector/landmarks:i (faceSelector.py)
    │    └─▶ /interactionManager/landmarks:i (interactionManager.py)
    └─▶ /alwayson/vision/img:o
         └─▶ /faceSelector/img:i (for annotation)

Context Publisher
    └─▶ /alwayson/stm/context:o
         └─▶ /interactionManager/context:i

faceSelector.py
    ├─▶ /faceSelector/interactionManager:rpc ─▶ /interactionManager (RPC)
    └─▶ /faceSelector/interactionInterface:rpc ─▶ /interactionInterface (RPC)

Speech Recognition
    └─▶ /speech2text/text:o
         └─▶ /interactionManager/stt:i

interactionManager.py
    ├─▶ /interactionManager/speech:o ─▶ /acapelaSpeak/speech:i
    └─▶ /acapelaSpeak/bookmark:o ─▶ /interactionManager/acapela_bookmark:i
```
---

## Performance Characteristics

- **Perception**: ~20 Hz (50ms period) - external module
- **Face Selection**: ~20 Hz (50ms period, configurable via `--rate`)
- **Interaction Manager**: 1 Hz (1s period)
- **LLM Inference**: Variable, 60s timeout with 3 retry attempts
- **STT Timeout**: 10 seconds
- **TTS Timeout**: 30 seconds with bookmark fallback
- **File Verification Timeout**: 5 seconds (face registration)
- **SS1 Duration**: ~15-30 seconds
- **SS2 Duration**: ~30-60 seconds (2 attempts max in lively context, 1 in calm)
- **SS3 Duration**: ~15-30 seconds (2 attempts max in lively context, 1 in calm)
- **SS4 Duration**: 60-120 seconds (max 5 turns or 120s timeout)

## Error Handling & Recovery

- **Consecutive errors**: faceSelector stops after 10 consecutive errors
- **Database failure**: Automatic fallback to JSON file logging
- **LLM failure**: 3 retry attempts with exponential backoff
- **Port disconnection**: Graceful handling with status logging
- **TTS bookmark timeout**: Fallback sleep timers (1.4s or 1.0s)
- **Thread safety**: Lock-based synchronization prevents race conditions
