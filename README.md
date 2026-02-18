# Proactive Social Robot - Architecture Summary

## System Overview

This system implements a **proactive social robot** using the iCub
humanoid robot platform. The robot autonomously detects people, analyzes
their social context, and initiates appropriate interactions based on a
state machine that tracks social relationships and learning progress.

### Core Philosophy

-   **Proactive Engagement**: Robot initiates interactions rather than
    waiting for user commands
-   **Social State Tracking**: Maintains memory of past interactions
    (known/unknown, greeted today, talked today)
-   **Adaptive Learning**: Progressive spatial constraints (Learning
    States LS1--LS4) ensure quality interactions
-   **Real-Time Robustness**: Non-blocking architecture with monitored
    interaction execution

------------------------------------------------------------------------

## System Architecture

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

------------------------------------------------------------------------

## Component Breakdown

### 1. perception.py - Vision Analyzer Module

**Purpose**: Real-time face detection, tracking, landmark extraction,
and gaze analysis.

**Responsibilities**: - Detect faces and assign `track_id` - Resolve
`face_id` (known name or temporary code) - Compute bounding boxes
`(x, y, w, h)` - Estimate head pose (pitch, yaw, roll) - Compute
attention (`MUTUAL_GAZE`, `NEAR_GAZE`, `AWAY`) - Detect talking
behavior - Estimate zone and distance categories - Publish structured
YARP bottle

Output is consumed by both `faceSelector` and `interactionManager`.

------------------------------------------------------------------------

### 2. faceSelector.py - Face Selection & Decision Module

**Purpose**: Select target face (biggest bounding box), compute Social
State (SS) and Learning State (LS), manage AO start/stop, and trigger
interactions.

#### Social States (SS)

  State   Meaning
  ------- -----------------------------------------
  SS1     Unknown
  SS2     Known, not greeted today
  SS3     Known, greeted today, not talked
  SS4     Known, greeted today, talked (terminal)

"Greeted today" and "Talked today" apply only to known persons.

#### Learning States (LS)

LS1--LS4 progressively restrict zone, distance, and attention
requirements.\
Eligibility is enforced before triggering interaction.

#### Selection Logic

1.  Parse all faces.
2.  Compute SS and LS.
3.  Select **biggest bounding box face** (with small stability guard).
4.  Trigger interaction only if:
    -   Not busy
    -   Not in cooldown
    -   LS spatial gate satisfied
    -   Social state not SS4

#### AO Behaviour

-   If at least one face exists → `ao_start` (edge-triggered)
-   If no faces → `ao_stop` (edge-triggered)

#### Persistence

-   `learning.json`
-   `greeted_today.json`
-   `talked_today.json`
-   `last_greeted.json`
-   `faceSelector.db` (async logging)

------------------------------------------------------------------------

### 3. interactionManager.py - Interaction Execution Module

**Purpose**: Execute interaction trees in real time using LLM +
monitored abort logic.

#### Interaction Trees

SS1 (Unknown) - Execute `ao_hi` - Wait response - Ask name (retry
once) - Extract name (LLM) - Say "Nice to meet you `<name>`{=html}" -
Register via objectRecognition RPC - Update `last_greeted`

SS2 (Known, Not Greeted) - Say "Hi `<name>`{=html}\` - Wait response
(retry once) - On success → chain into SS3

SS3 (Known, Greeted, Not Talked) - Max 3-turn short conversation - Turn
1: Starter - Turn 2: Follow-up - Turn 3: Closing acknowledgment (no
question) - At least one user response → success

SS4 is terminal and does not execute a tree.

#### Real-Time Safeguards

-   TargetMonitor thread aborts if:
    -   Target disappears
    -   Another face becomes biggest
-   STT waiting is interruptible
-   LLM calls include retry logic
-   Only one interaction runs at a time (run_lock)

#### Logging

-   `interaction_data.db` (async WAL logging)
-   Full JSON result contract returned to faceSelector

------------------------------------------------------------------------

## Data Flow

Perception → faceSelector → interactionManager → faceSelector → Learning
update

------------------------------------------------------------------------

## Performance

-   faceSelector: \~20 Hz
-   interactionManager: 1 Hz control loop
-   SS3: max 3 turns, max 120 seconds
-   All DB writes async
-   RPC calls protected with timeout

------------------------------------------------------------------------

## Dependencies

-   yarp-python
-   ollama (Phi-3 mini)
-   sqlite3 (stdlib)

------------------------------------------------------------------------

## File Structure

    proactive_social_robot/
    ├── faceSelector.py
    ├── interactionManager.py
    ├── README.md
    ├── learning.json
    ├── greeted_today.json
    ├── talked_today.json
    ├── faceSelector.db
    ├── interaction_data.db
    └── last_greeted.json
