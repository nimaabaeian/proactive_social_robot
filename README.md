# Embodied Behaviour

> **Robot:** iCub  
> **Platform:** YARP  
> **Author:** Nima Abaeian

This document explains the runtime architecture in plain language and diagrams.
It is intended to be the single technical reference for how the system senses, decides, interacts, and remembers.

---

## 1) Big Picture (Intuition First)

Think of the system as four always-on layers:

1. **See people** (`vision.py`)
2. **Choose who matters now** (`salienceNetwork.py`)
3. **Run interaction behavior** (`executiveControl.py`)
4. **Maintain long-term relationship over Telegram** (`chatBot.py`)

### System map

```text
           CAMERA
             |
             v
 +---------------------------+
 | vision.py                 |
 | detect / track / identify |
 | gaze / pose / talking /QR |
 +---------------------------+
             |
             | /alwayson/vision/landmarks:o
             v
 +------------------------------+
 | salienceNetwork.py           |
 | social state + adaptive IPS  |
 | target arbitration + gating  |
 +------------------------------+
             |
             | RPC run(track, face, ss)
             v
 +------------------------------+
 | executiveControl.py          |
 | SS trees + hunger tree       |
 | TTS/STT + responsive behavior|
 +------------------------------+
             |
             | /alwayson/executiveControl/hunger:o
             v
 +------------------------------+
 | chatBot.py                   |
 | Telegram LLM + user memory   |
 | hunger-aware persona         |
 +------------------------------+
```

Supporting utilities:

- `generateQr.py`: creates feed QR codes (`SMALL_MEAL`, `MEDIUM_MEAL`, `LARGE_MEAL`)
- `mockPublisher.py`: publishes mock STM context and mock hunger transitions

---

## 2) Dataflow and Controlflow

### End-to-end runtime sequence

```text
[Frame arrives]
   -> vision.py publishes per-face landmarks
   -> salienceNetwork.py computes SS + IPS, selects target
   -> salienceNetwork.py checks cooldown/eligibility/executive availability
   -> executiveControl.py executes tree (social or hunger)
   -> executiveControl.py updates JSON/DB and publishes hunger
   -> chatBot.py adapts Telegram behavior from hunger stream
```

### Fast mental model

```text
Perception stream (high rate) ---> Selection gate ---> Interaction transaction ---> Memory update
         continuous                    opportunistic           bounded/abortable        asynchronous
```

---

## 3) Module Details

## 3.1 `vision.py` — Perception Front-End

### What it does

- Reads image stream and runs:
  - YOLO face detection
  - ByteTrack identity continuity (`track_id`)
  - optional `face_recognition` matching
  - MediaPipe Face Landmarker (pose/gaze)
  - QR decoding
- Publishes one rich landmark bottle per face.
- Publishes scene-level compact features.
- Receives selected target (`track_id`, `ips`) and sends FaceTracker-compatible bbox command.
- Supports runtime naming over RPC: `name <person_name> id <track_id>`.

### Per-face output model

```text
face_id, track_id,
bbox(x,y,w,h), zone, distance,
gaze_direction, pitch/yaw/roll, cos_angle,
attention, is_talking, time_in_view
```

### Internal perception pipeline

```text
RGB frame
   |
   +--> YOLO --> face boxes -------------------+
   |                                           |
   +--> ByteTrack --> stable track_id ---------+--> fused face objects
   |                                           |
   +--> MediaPipe landmarks --> head pose/gaze-+
   |
   +--> lip landmarks buffer --> is_talking
   |
   +--> QR detector --> /alwayson/vision/qr:o
```

### Ports and RPC

- Input:
  - `/alwayson/vision/img:i`
  - `/alwayson/vision/targetCmd:i`
- Outputs:
  - `/alwayson/vision/landmarks:o`
  - `/alwayson/vision/features:o`
  - `/alwayson/vision/targetBox:o`
  - `/alwayson/vision/faces_view:o`
  - `/alwayson/vision/qr:o`
- RPC:
  - `/alwayson/vision/rpc` (`name`, `help`, `process`, `quit`)

RPC endpoint note:

- The RPC port name is configurable via `rpc_name` in `vision.py` ResourceFinder params.
- Default effective endpoint is `/alwayson/vision/rpc`.

---

## 3.2 `salienceNetwork.py` — Target Selection and Gating

### What it does

- Consumes face landmarks from `vision.py`.
- Computes social state `ss1..ss4` per face.
- Computes adaptive IPS (interest priority score).
- Decides:
  - who the robot should **look at**
  - who the robot should **talk to**
- Triggers `executiveControl` only when a candidate passes gates.
- Logs events and updates social-learning memory.

### Two-layer arbitration

```text
LAYER A: ATTENTION (head/eyes target)
  priority:
    1) executive override track_id
    2) active interaction track lock
    3) best IPS face

LAYER B: DIALOGUE (start proactive interaction)
  requires:
    - not interaction_busy
    - no override active
    - candidate eligible
    - cooldown passed
    - executiveControl status = not busy
```

### IPS intuition

```text
IPS = weighted_sum(proximity, centricity, approach_velocity, gaze)
      * habituation_decay(time_idle)
      + hysteresis_bonus(if same target)
```

### IPS algorithm (implemented)

The score used in `salienceNetwork.py` is computed per face as follows.

1) **Normalize input variables**

```text
Given bbox = (x, y, w, h), image size = (W=640, H=480)

s_prox = clamp(h / H, 0, 1)

cx = x + w/2
cy = y + h/2
max_dist = sqrt((W/2)^2 + (H/2)^2)
dist_center = sqrt((cx - W/2)^2 + (cy - H/2)^2)
s_cent = clamp(1 - dist_center/max_dist, 0, 1)

area_now  = w*h
area_prev = previous area for same track_id (or area_now if first sight)
raw_vel   = (area_now - area_prev) / (W*H)
s_vel     = clamp(raw_vel * 10.0, 0, 1)

s_gaze = clamp(cos_angle, 0, 1)
```

2) **Apply person-specific weights**

If the person exists in `learning.json`, use learned weights; otherwise use baseline:

```text
w_prox=0.5, w_cent=0.15, w_vel=0.3, w_gaze=0.5

base_ips = w_prox*s_prox + w_cent*s_cent + w_vel*s_vel + w_gaze*s_gaze
```

3) **Apply habituation decay**

```text
time_since_last_interaction = now - last_interaction_time(cooldown_key)
t_idle = min(time_in_view, time_since_last_interaction)
habituation = exp(-lambda * t_idle), lambda = 0.05

ips = base_ips * habituation
```

4) **Apply hysteresis bonus**

```text
if face.track_id == current_target_track_id and track_id != -1:
    ips += 0.3
```

5) **Use IPS for behavior decisions**

```text
Target selection: max IPS face

Eligibility thresholds by social state:
  ss1 >= 1.0
  ss2 >= 0.8
  ss3 >= 1.2
  ss4 >= 99.0  (effectively never proactive)

Tracking gate (look-only): target IPS must also pass min_track_ips (default 0.6)
```

### How it is adaptive

`salienceNetwork.py` adapts online in two ways:

1. **Short-term adaptation (within session)**
  - Habituation decay lowers IPS while a person stays in view without interaction.
  - Hysteresis bonus stabilizes the current target and prevents noisy switching.

2. **Long-term adaptation (across sessions)**
  - Per-person IPS weights are updated after each interaction outcome.
  - Success shifts behavior toward proactive selection.
  - Failure shifts behavior toward conservative/reactive selection.

```text
interaction result
  |
  +--> success? yes --> increase prox/vel, decrease gaze
  |
  +--> success? no  --> decrease prox/vel, increase gaze
  |
  +--> save weights in learning.json
  |
  +--> future IPS uses updated personal weights
```

### Social-state assignment

```text
is_known = false                     -> ss1
is_known = true, greeted_today = no  -> ss2
is_known = true, greeted=yes, talked=no -> ss3
is_known = true, greeted=yes, talked=yes -> ss4
```

### Context-aware cooldown

```text
STM label = 1  -> lively  -> short cooldown
STM label = 0  -> calm    -> long cooldown
STM missing    -> default cooldown
```

### Ports and RPC

- Input:
  - `/alwayson/salienceNetwork/landmarks:i`
  - `/alwayson/salienceNetwork/context:i` (optional)
- Outputs:
  - `/alwayson/salienceNetwork/targetCmd:o`
  - `/alwayson/salienceNetwork/debug:o`
- RPC server:
  - `/salienceNetwork` (`set_track_id`, `reset_cooldown`)
- RPC clients:
  - `/salienceNetwork/executiveControl:rpc`
  - `/salienceNetwork/faceTracker:rpc`

FaceTracker lifecycle note:

- `salienceNetwork.py` sends `run` to FaceTracker during startup.
- `salienceNetwork.py` sends `sus` to FaceTracker during module close.

---

## 3.3 `executiveControl.py` — Interaction Engine

### What it does

- Receives proactive run requests (`run track_id face_id ss`).
- Runs social interaction trees (`ss1`, `ss2`, `ss3`, `ss4`).
- Runs hunger feeding tree when hunger policy requires it.
- Publishes hunger state continuously.
- Handles responsive interactions when no proactive interaction is running.
- Uses Azure OpenAI for name extraction + short conversational turns.

### Proactive interaction flow

```text
run(track, face, ss)
   |
   +--> start target monitor (abort if target lost too long)
   |
   +--> choose behavior path:
         - hunger path (HS3 always, HS2+ss3)
         - social tree path (ss1/ss2/ss3/ss4)
   |
   +--> return compact result JSON
```

### Social trees (intuitive)

```text
SS1 unknown:
  greet -> wait -> ask name -> (retry if needed) -> extract name -> register -> close

SS2 known/not greeted:
  greet attempt #1 -> if response go SS3
  else greet attempt #2 -> if response go SS3 else abort

SS3 known/greeted/not talked:
  starter -> up to max turns follow-up/closing
  if user speaks at least once => talked=true => ss4 path

SS4:
  no-op success
```

### Hunger subsystem

```text
stomach level drains over time (HungerModel)

HS1: normal
HS2: hungry
HS3: starving

if hunger tree active:
  ask for food -> wait QR -> apply delta -> thank -> repeat until satisfied or timeout
```

Feed deltas:

- `SMALL_MEAL` = +10
- `MEDIUM_MEAL` = +25
- `LARGE_MEAL` = +45

### Responsive interactions

```text
responsive loop:
  - if proactive busy: drop responsive events immediately
  - greeting detected in STT -> choose largest face -> greet/intro path
  - QR feed outside proactive -> short acknowledgment
```

### LLM execution model

```text
main thread submits request --> async worker --> result map
                                   |
                                   +--> timeout/cancel fallback if overdue
```

### Ports and RPC

- RPC server: `/executiveControl`
  - `status`, `ping`, `help`, `run`, `hunger`, `quit`
- Inputs:
  - `/alwayson/executiveControl/landmarks:i`
  - `/alwayson/executiveControl/stt:i`
  - `/alwayson/executiveControl/qr:i`
- Outputs:
  - `/alwayson/executiveControl/speech:o`
  - `/alwayson/executiveControl/hunger:o`

---

## 3.4 `chatBot.py` — Telegram Relationship Layer

### What it does

- Polls Telegram updates in background.
- Consumes hunger state from executive.
- Generates hunger-aware replies using prompt overlays.
- Stores:
  - per-chat memory summary/history
  - per-user profile memory
- Broadcasts starvation messages in HS3 using cooldown rules.

### Behavior by hunger state

```text
HS1: normal social texting
HS2: normal + occasional hunger leakage
HS3: strict starving mode, in-person feeding request focus
```

### HS3 broadcast logic

```text
on HS3 entry:
  broadcast to all subscribers

while staying HS3:
  periodic rebroadcast candidates by cooldown
  skip users who chatted very recently
```

### User memory extraction model

From message metadata and text patterns, stores compact user profile:

- name / nickname / age
- likes / dislikes / topics
- recent life update
- conversation style
- inside jokes (with confidence through repetition)

### RPC

- `/chatBot/rpc`
  - `status`
  - `set_hs HS1|HS2|HS3`
  - `reload_prompts`

Prompt path behavior:

- Default prompt file path resolves to parent `alwaysOn` directory: `../prompts.json`.
- Runtime override is supported through ResourceFinder `prompts` parameter.

---

## 3.5 `mockPublisher.py` — Test Stream Generator

Publishes synthetic channels for integration testing without full upstream stack:

- `/alwayson/stm/context:o` → `(episode_id, chunk, label)`
- `/interactionManager/hunger:o` → hunger transitions over `{HS1, HS2, HS3}`
  - compatibility-only mock stream using historical naming
  - main live architecture stream is `/alwayson/executiveControl/hunger:o`

---

## 3.6 `generateQr.py` — Feed QR Generator

Generates QR PNGs used by hunger feed path:

- `small_meal.png`
- `medium_meal.png`
- `large_meal.png`

Optional decode validation with `--verify`.

---

## 3.7 `prompts.json` — Prompt Surface

Contains prompt sets for:

- `chat_bot`
- `executiveControl`

Includes system prompts, extraction prompts, starter/follow-up/closing templates, hunger overlays, and fallbacks.

---

## 4) State Machines

## 4.1 Social state (`salienceNetwork.py`)

```text
             +------------------+
             | unknown identity |
             +--------+---------+
                      |
                      v
                     ss1
                      |
              (name known & not greeted)
                      v
                     ss2
                      |
             (greeted today = yes)
                      v
                     ss3
                      |
             (talked today = yes)
                      v
                     ss4
```

## 4.2 Hunger state (`executiveControl.py`)

```text
level >= hungry_threshold                     -> HS1
starving_threshold <= level < hungry_threshold -> HS2
level < starving_threshold                    -> HS3
```

## 4.3 Chatbot hunger fallback

```text
hunger stale and no manual override -> effective HS1
manual override active              -> use overridden HS directly
```

---

## 5) Failure and Abort Paths

This section describes what happens when interactions fail, inputs disappear, or services are temporarily unavailable.

## 5.1 Proactive interaction abort path (`executiveControl.py`)

```text
run(track, face, ss)
  |
  +--> start target monitor
  |
  +--> execute tree
        |
        +--> target present? yes --> continue
        |
        +--> target absent > TARGET_LOST_TIMEOUT
             -> set abort_event
             -> stop current waits (STT/LLM/speech wait)
             -> return result with abort_reason
```

## 5.2 No-response path (social trees)

```text
SS1:
  greeting -> no user response -> abort no_response_greeting
  ask name -> no response      -> abort no_response_name
  name extraction fail (incl retry) -> abort name_extraction_failed

SS2:
  greet attempt #1 no response -> attempt #2
  greet attempt #2 no response -> abort no_response_greeting

SS3:
  starter sent
  no user utterance in timeout -> abort no_response_conversation
```

## 5.3 Responsive gating path (drop, don’t defer)

```text
responsive event arrives
  |
  +--> proactive interaction running?
        |
        +--> yes: drop event immediately
        +--> no : execute responsive action now
```

## 5.4 Salience → executive call protection

```text
candidate selected in salience
  |
  +--> cooldown passed?
  +--> eligible?
  +--> executive RPC reachable?
  +--> executive status busy?

if any gate fails: skip run now, continue tracking
if all pass     : start interaction thread
```

## 5.5 Queue pressure and backpressure behavior

```text
queue full (DB/IO/responsive/telegram updates)
  -> drop oldest
  -> enqueue newest if possible
  -> continue module loop (non-blocking)
```

## 5.6 Chatbot stale hunger safety

```text
no fresh hunger update for HS_STALE_SEC
  -> effective hunger forced to HS1
  -> avoids stale HS2/HS3 behavior
manual override active
  -> stale protection bypassed
```

---

## 6) YARP Interfaces

## 6.1 Core stream wiring

```bash
yarp connect /alwayson/vision/landmarks:o /alwayson/salienceNetwork/landmarks:i
yarp connect /alwayson/salienceNetwork/targetCmd:o /alwayson/vision/targetCmd:i
yarp connect /alwayson/vision/targetBox:o /faceTracker/target:i

yarp connect /alwayson/vision/qr:o /alwayson/executiveControl/qr:i
yarp connect /speech2text/text:o /alwayson/executiveControl/stt:i
yarp connect /alwayson/executiveControl/speech:o /acapelaSpeak/speech:i

yarp connect /alwayson/executiveControl/hunger:o /alwayson/chatBot/hunger:i
```

Optional:

```bash
yarp connect /alwayson/stm/context:o /alwayson/salienceNetwork/context:i
```

## 6.2 RPC endpoints

- `salienceNetwork`
  - `set_track_id <int>`
  - `reset_cooldown <face_id> <track_id>`
- `executiveControl`
  - `status`, `ping`, `help`
  - `run <track_id> <face_id> <ss1|ss2|ss3|ss4>`
  - `hunger <hs1|hs2|hs3>`
  - `quit`
- `chatBot`
  - `status`
  - `set_hs HS1|HS2|HS3`
  - `reload_prompts`
- `vision`
  - `name <person_name> id <track_id>`
  - `help`, `process on/off`, `quit`

---

## 7) Persistence and Logging

## 7.0 Memory architecture (what is remembered)

The system memory is split into **episodic interaction memory**, **daily social memory**,
**person-level adaptive memory**, and **chat relationship memory**:

```text
            +----------------------------+
            |  Episodic interaction logs |
            |  (SQLite rows, timestamps) |
            +-------------+--------------+
                          |
                          v
            +----------------------------+
            | Daily social memory        |
            | greeted_today / talked_today|
            +-------------+--------------+
                          |
                          v
            +----------------------------+
            | Person adaptive memory     |
            | learning.json weights      |
            +-------------+--------------+
                          |
                          v
            +----------------------------+
            | Chat relationship memory   |
            | chat_bot.db user/chat mem  |
            +----------------------------+
```

### Who writes what

- `salienceNetwork.py`
  - reads/writes: `greeted_today.json`, `talked_today.json`, `learning.json`
  - logs to: `salience_network.db`
- `executiveControl.py`
  - writes: `last_greeted.json`, `greeted_today.json`, `hunger_state.json`
  - logs to: `executive_control.db`
- `chatBot.py`
  - writes: `chat_bot.db` (`meta`, `subscribers`, `chat_memory`, `user_memory`)

Runtime path resolution note:

- `chatBot.py` and `executiveControl.py` resolve `chat_bot.db` from parent `alwaysOn/memory` at runtime.
- `hunger_state.json` is also resolved from parent `alwaysOn/memory` at runtime.
- Paths shown below as `memory/...` are logical storage names; deployment may map them outside this workspace folder.

### Memory write safety model

```text
producer module
   -> enqueue write/log event
   -> background worker drains queue
   -> atomic file replace / sqlite commit
   -> main loop keeps running (non-blocking)
```

### SQLite

- `memory/chat_bot.db` (logical name; runtime-resolved under parent `alwaysOn/memory`)
  - `meta`, `subscribers`, `chat_memory`, `user_memory`
- `.../data_collection/executive_control.db`
  - proactive + responsive interaction records
- `.../data_collection/salience_network.db`
  - target selections, SS changes, learning changes, interaction attempts

### JSON

- `.../memory/greeted_today.json`
- `.../memory/talked_today.json`
- `.../memory/last_greeted.json`
- `.../memory/learning.json`
- `memory/hunger_state.json` (logical name; runtime-resolved under parent `alwaysOn/memory`)

### Persistence strategy

```text
runtime updates -> queue -> background worker -> atomic write/commit
```

## 7.4 Memory semantics by timescale

- **Frame scale (milliseconds):** face geometry, gaze, speaking signal (`vision.py`)
- **Interaction scale (seconds/minutes):** run outcome, abort reason, transcript snippets
- **Day scale:** greeted/talked flags drive `ss1..ss4`
- **Long-term person scale:** adaptive IPS weights in `learning.json`
- **Relationship scale:** chatbot profile + chat summaries in `chat_bot.db`

---

## 8) Concurrency Model

```text
vision.py
  - main RFModule loop
  - lock-protected face identity maps

salienceNetwork.py
  - main loop
  - interaction thread
  - IO worker + DB worker + last_greeted refresh thread

executiveControl.py
  - main loop
  - target monitor thread
  - QR reader thread
  - responsive loop thread
  - async LLM worker thread
  - DB worker thread

chatBot.py
  - main loop
  - Telegram polling thread + update queue
```

---

## 9) Startup Order

1. Start `yarpserver`.
2. Start `vision.py`.
3. Start `salienceNetwork.py`.
4. Start `executiveControl.py`.
5. Start `chatBot.py`.
6. Connect YARP ports.
7. Optional: run `mockPublisher.py` for synthetic testing.

---

## 10) Operational Behavior Guarantees

- `executiveControl` proactive and responsive paths are mutually exclusive.
- `salienceNetwork` can keep running even if STM context is not connected yet.
- `chatBot` protects against stale hunger streams via effective-state fallback.
