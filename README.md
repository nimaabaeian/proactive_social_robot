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
- Base score from social state transition success
- Bonuses: Context relevance, multiple turns, quick responses
- Penalties: Timeouts, no responses, errors
- Thresholds: +15 → upgrade LS, -10 → downgrade LS

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
- **Ollama**: Must have `phi3:mini-q4` model downloaded
- **Object Recognition Module**: Provides face detection/tracking
- **Acapela TTS**: Text-to-speech synthesis
- **Speech2Text**: Speech recognition module

### File System
```
proactive_social_robot/
├── perception.py
├── faceSelector.py
├── interactionManager.py
├── mock_context_publisher.py
├── learning.json              # Generated at runtime
├── greeted_today.json         # Generated at runtime
├── talked_today.json          # Generated at runtime
├── interaction_data.db        # Generated at runtime
└── face_landmarker.task       # MediaPipe model file
```

---

## Port Connection Diagram

```
Camera Feed
    └─▶ /alwayson/vision/img:i (perception.py)

Object Recognition
    └─▶ /alwayson/vision/recognition:i (perception.py)

perception.py
    └─▶ /alwayson/vision/landmarks:o
         ├─▶ /faceSelector/landmarks:i (faceSelector.py)
         └─▶ /interactionManager/landmarks:i (interactionManager.py)

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

- **Perception**: ~20 Hz (50ms period)
- **Face Selection**: ~20 Hz (50ms period)
- **LLM Inference**: ~5-30 seconds per query (local CPU)
- **STT Timeout**: 10 seconds
- **TTS Duration**: Variable (5-15 seconds typical)
- **SS1 Duration**: ~15-30 seconds
- **SS2 Duration**: ~30-60 seconds
- **SS4 Duration**: ~60-120 seconds (max)
