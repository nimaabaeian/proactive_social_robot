# Always On Cognitive Architecture — `Embodied Behaviour` 
> **Modules:** `faceSelector.py` · `interactionManager.py` · `telegram_bot.py`
> **Platform:** YARP (Yet Another Robot Platform)  
> **Robot:** iCub
> **Author:** Nima Abaeian
--

## Table of Contents

1. [Overview](#1-overview)
2. [System Block Diagram](#2-system-block-diagram)
3. [Module: `faceSelector`](#3-module-faceselector)
   - [Purpose](#31-purpose)
   - [YARP Ports](#32-yarp-ports)
   - [State Model](#33-state-model)
   - [Main Loop — updateModule()](#34-main-loop--updatemodule)
   - [Face Selection Policy & Context-Aware Cooldown](#35-face-selection-policy--context-aware-cooldown)
   - [Interaction Trigger Flow](#36-interaction-trigger-flow)
   - [Reward & Learning State Updates](#37-reward--learning-state-updates)
   - [Background Threads](#38-background-threads)
   - [Persistent Data Files](#39-persistent-data-files)
   - [SQLite Logging](#310-sqlite-logging)
   - [Image Annotation & Visualization](#311-image-annotation--visualization)
4. [Module: `interactionManager`](#4-module-interactionmanager)
   - [Purpose](#41-purpose)
   - [YARP Ports](#42-yarp-ports)
   - [State Trees (Interaction Flows)](#43-state-trees-interaction-flows)
   - [SS1 — Unknown Person](#44-ss1--unknown-person)
   - [SS2 — Known, Not Greeted](#45-ss2--known-not-greeted)
   - [SS3 — Known, Greeted, Not Talked](#46-ss3--known-greeted-not-talked)
   - [Telegram User Lookup & Personalization](#47-telegram-user-lookup--personalization)
   - [Hunger / QR Feeding Tree](#48-hunger--qr-feeding-tree)
   - [Target Monitor](#49-target-monitor)
   - [Responsive Interaction Path](#410-responsive-interaction-path)
   - [LLM Integration (Azure OpenAI)](#411-llm-integration-azure-openai)
   - [Speech Output (TTS)](#412-speech-output-tts)
   - [STT (Speech-to-Text) Input](#413-stt-speech-to-text-input)
   - [HungerModel](#414-hungermodel)
   - [RPC Interface](#415-rpc-interface)
   - [Database](#416-database)
5. [Cross-Module Data Flow](#5-cross-module-data-flow)
6. [State Transition Diagrams](#6-state-transition-diagrams)
   - [Social State Machine](#61-social-state-machine)
   - [Learning State Machine](#62-learning-state-machine)
7. [Threading Architecture](#7-threading-architecture)
8. [Memory Files Reference](#8-memory-files-reference)
9. [Key Constants Reference](#9-key-constants-reference)
10. [YARP Connection Commands](#10-yarp-connection-commands)
11. [Module: `telegram_bot`](#11-module-telegram_bot)
    - [Purpose](#111-purpose)
    - [YARP Ports & RPC Interface](#112-yarp-ports--rpc-interface)
    - [Message Handling](#113-message-handling)
    - [Hunger-Aware Personality](#114-hunger-aware-personality)
    - [HS3 Broadcast System](#115-hs3-broadcast-system)
    - [User Memory & Personalization](#116-user-memory--personalization)
    - [Conversation Memory](#117-conversation-memory)
    - [LLM Integration](#118-llm-integration)
    - [Database Schema](#119-database-schema)
    - [Threading Architecture](#1110-threading-architecture)
    - [Prompts (`prompts.json`)](#1111-prompts-promptsjson)
    - [Key Constants](#1112-key-constants)

---

## 1. Overview

The`Embodied Behaviour` Module in the `AlwaysOn Cognitive Architecture` system implements a **adaptive social interaction architecture** for the iCub robot. 

The Embodied Behaviour has three tightly coupled modules:

| **`faceSelector`** |
| **`interactionManager`** |
| **`telegram_bot`** |
---

## 2. System Block Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                      alwaysOn Architecture                           │
└──────────────────────────────────────────────────────────────────────┘

      Perception (faces, image)                     User inputs
                 │                                  (speech, QR)
                 │                                        │
                 v                                        │
┌──────────────────────────────────────────────────────────────────────┐
│ faceSelector                                                         │
│ - Parse face observations (id, bbox, attention, distance)            │
│ - Compute Social State (SS) and Learning State (LS)                  │
│ - Apply eligibility + cooldown rules                                 │
│ - Select biggest-bbox face as the active target                      │
│ - Trigger proactive interaction and update learning/memory           │
└──────────────────────────────────────────────────────────────────────┘
                 │
                 │ RPC call: run(track_id, face_id, ss)
                 v
┌──────────────────────────────────────────────────────────────────────┐
│ interactionManager                                                   │
│ - Run interaction trees (SS1/SS2/SS3; SS4 no-op)                     │
│ - Run hunger feed tree (HS override) with QR feeding events          │
│ - Monitor target continuity (still visible + still biggest)          │
│ - Use LLM for name extraction and conversation generation            │
│ - Coordinate STT/TTS + behavior execution                            │
│ - Handle responsive greetings/QR acknowledgments when proactive idle │
│ - Drive robot actions (speech + behaviors)                           │
└──────────────────────────────────────────────────────────────────────┘
                 │
                 │ compact JSON result (success, final_state, abort...)
                 └──────────────────────────────► back to faceSelector

                 │ hunger state (HS1/HS2/HS3) at 1 Hz
                 │ /interactionManager/hunger:o
                 v
┌──────────────────────────────────────────────────────────────────────┐
│ telegram_bot                                                         │
│ - Chat with registered users via Telegram (Azure LLM replies)        │
│ - Adapt personality to hunger state (HS1/HS2/HS3 overlays)          │
│ - Broadcast HS3 starvation alerts to all subscribers                 │
│ - Build per-user long-term memory (name, likes, inside jokes, etc.)  │
│ - Persist user_memory → telegram_bot.db (read by interactionManager) │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ State Machines                                                       │
│ - SS (Social): ss1 / ss2 / ss3 / ss4                                 │
│   computed in faceSelector, executed in interactionManager trees     │
│ - LS (Learning): LS1 <-> LS2 <-> LS3                                 │
│   maintained in faceSelector via reward updates                      │
│ - HS (Hunger): HS1 <-> HS2 <-> HS3                                   │
│   maintained in interactionManager (can override social tree)        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ Shared Memory + Logs                                                 │
│ - JSON memory (learning, greeted/talked, last_greeted)               │
│ - SQLite logs (faceSelector + interactionManager records)            │
│ - SQLite user/chat memory (telegram_bot.db)                          │
└──────────────────────────────────────────────────────────────────────┘
faceSelector ─────────────── read/write ───────────────┐
                                                        ├── memory
interactionManager ───────── read/write ───────────────┘
telegramBot ──────────────── read (user_memory) ───────┘

```
---

## 3. Module: `faceSelector`

### 3.1 Purpose

`faceSelector` is the **perception and selection layer**. It:
- Reads raw face landmark data from the vision system
- Parses Social State (SS), Learning State (LS), distance, attention, `time_in_view`, and bounding box per face
- Selects **one target**: always the face with the **biggest bounding box**
- Waits for that face's identity (`face_id`) to be resolved before proceeding
- Checks eligibility gates (cooldown, LS constraints, SS state)
- Fires off an interaction thread that calls `interactionManager` via RPC
- Processes the interaction result to update learning states and social memory

### 3.2 YARP Ports

| Port | Type | Direction | Purpose |
|---|---|---|---|
| `/faceSelector/landmarks:i` | BufferedPortBottle | IN | Face landmark data from vision |
| `/faceSelector/img:i` | BufferedPortImageRgb | IN | Camera image for annotation |
| `/faceSelector/img:o` | BufferedPortImageRgb | OUT | Annotated camera image |
| `/faceSelector/debug:o` | Port | OUT | Debug status bottle |
| `/faceSelector/interactionManager:rpc` | RpcClient | OUT | Trigger interactions |
| `/faceSelector/interactionInterface:rpc` | RpcClient | OUT | Send `ao_start`/`ao_stop` signals |
| `/faceSelector/context:i` | BufferedPortBottle | IN | STM scene-context cluster label (auto-connects to `/alwayson/stm/context:o`; non-critical — module runs normally if STM is absent) |

### 3.3 State Model

#### Social States (SS)

| State | Meaning | Condition |
|---|---|---|
| `ss1` | **Unknown** | `face_id` not recognized |
| `ss2` | **Known, Not Greeted** | Known person, not greeted today |
| `ss3` | **Known, Greeted, No Talk** | Known, greeted today, no conversation yet |
| `ss4` | **Known, Greeted, Talked** | Full interaction completed today (no further action) |

```
is_known? ─── NO ──────────────────► ss1
              │
             YES
              ├── greeted_today? ─── NO ──► ss2
              │
             YES
              ├── talked_today? ─── NO ──► ss3
              │
             YES ────────────────────────► ss4  (no-op)
```

#### Learning States (LS)

| State | Description | Distance allowed | Attention required | Minimum `time_in_view` |
|---|---|---|---|---|
| `LS1` | Early — strict constraints | `SO_CLOSE`, `CLOSE` | `MUTUAL_GAZE` only | `>= 3.0s` |
| `LS2` | Developing — relaxed | `SO_CLOSE`, `CLOSE`, `FAR` | `MUTUAL_GAZE`, `NEAR_GAZE` | `>= 1.0s` |
| `LS3` | Advanced — no constraints | Any | Any | `>= 0.0s` |

LS values are **per-person** and stored in `learning.json`. They evolve via **reward shaping** after each interaction.

#### Eligibility Check (`_is_eligible`)

A face is eligible for interaction if:
- It is **not** `ss4`
- If `LS3` → always eligible
- If `LS1` or `LS2` → must satisfy distance, attention, and minimum `time_in_view` constraints for that LS level

### 3.4 Main Loop — `updateModule()`

Runs at **20 Hz** (period = 0.05 s). Steps each cycle:

```
0. Read STM context (non-blocking) → update current_context_label (0=calm, 1=lively, −1=unknown/none)
1. Check ports connected (landmarks + img). Wait if not.
2. Day-change check → reload memory JSON from disk, then prune greeted_today / talked_today if new day
3. _read_landmarks()     → parse face bottles from YARP
4. _read_image()         → get camera frame (with frame skip)
5. _compute_face_states()→ enrich each face with SS, LS, eligibility, last_greeted_ts
6. _select_biggest_face()→ find face with max bbox area

   IF face_id NOT resolved (still "recognizing"/"unmatched") → WAIT, do not fall back

   IF resolved AND not in cooldown:
     └─ IF eligible AND ss != ss4 AND interactionManager status is available+idle:
          → set interaction_busy = True
          → start _run_interaction_thread(candidate)
        ELSE:
          → skip proactive spawn this cycle

7. Annotate & publish image
8. Publish debug bottle
```

`_read_landmarks()` can now receive both:
- Faces matched with MediaPipe landmarks (full pose/gaze fields)
- Faces detected by object recognition but not landmark-matched (e.g., very small/far faces). These are published with neutral pose values and `attention="UNKNOWN"` while still carrying `time_in_view`.

### 3.5 Face Selection Policy & Context-Aware Cooldown

> **Rule:** Always pick the biggest bbox. Never fall back to a smaller resolved face.

```python
biggest = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])

if not _is_face_id_resolved(biggest['face_id']):
    # wait — do not switch to a different face
    return  

# Proceed with the biggest resolved face
```

**Cooldown key logic:**
- Known person → key = `person_id` (e.g., `"Alice"`)
- Unknown person → key = `f"unknown:{track_id}"`
- Cooldown duration: **context-aware** via `_effective_cooldown()`

#### Context-Aware Cooldown (`_effective_cooldown`)

The per-person cooldown adapts to the scene cluster label received from `/alwayson/stm/context:o`:

| STM label | Scene interpretation | Cooldown |
|---|---|---|
| `1` | Lively / active | `cooldown_lively = 3.0 s` |
| `0` | Calm / quiet | `cooldown_calm = 15.0 s` |
| `-1` | Not yet received or noise cluster | `cooldown_default = 5.0 s` |

The label is read once per `updateModule()` cycle (non-blocking). If the STM module is not running, `current_context_label` stays at `−1` and the default 5 s cooldown applies.

### 3.6 Interaction Trigger Flow

```
updateModule proactive trigger
  ├─ Select biggest resolved face (cooldown + eligibility checks)
  ├─ Pre-spawn check interactionManager status
  │     └─ If unavailable or busy → skip spawn
  └─ Spawn _run_interaction_thread(target)

_run_interaction_thread(target)
  ├─ Re-check interactionManager status → skip if unavailable/busy
  ├─ Skip if ss4
  ├─ _execute_interaction_interface("ao_start") → signal the robot body
  ├─ _run_interaction_manager(track_id, face_id, ss)
  │     └─ RPC: "run <track_id> <face_id> <ss>"
  │     └─ Returns compact JSON result
  ├─ _process_interaction_result(result, target)
  │     ├─ Update greeted_today / talked_today
  │     ├─ Update track_to_person mapping
  │     ├─ Compute reward delta
  │     └─ Update learning state (LS)
  └─ _execute_interaction_interface("ao_stop")
  
  [finally]
  └─ interaction_busy = False
  └─ Clear selected target/bbox and update cooldown timestamp
```

### 3.7 Reward & Learning State Updates

#### Reward Computation (`_compute_reward`)

| Scenario | Reward |
|---|---|
| Success + name extracted | `+2` |
| Success (no name) | `+1` |
| Failure: `not_responded` | `-1` |
| Failure: `face_disappeared` (first time in 30s window) | `-1` |
| Failure: `face_disappeared` (repeated ≥ 2 times in 30s) | `-2` |
| Other failure | `-1` |

#### Learning State Update (`_update_learning_state`)

```
delta > 0  →  new_ls = min(3, current_ls + 1)   (advance)
delta < 0  →  new_ls = max(1, current_ls - 1)   (regress)
delta = 0  →  no change
```

Changes are logged to SQLite (`ls_changes` table) and persisted to `learning.json`.

#### Face Disappear Penalty Tracking

Uses a **sliding window** per person:
- Window: 30 seconds
- Threshold: 2 events
- Below threshold → mild penalty (`-1`)
- At/above threshold → harsh penalty (`-2`)

### 3.8 Background Threads

| Thread | Function | Purpose |
|---|---|---|
| `_io_thread` | `_io_worker()` | Drains `_io_queue` → saves JSON files asynchronously |
| `_db_thread` | `_db_worker()` | Drains `_db_queue` → writes SQLite records |
| `_lg_refresh_thread` | `_last_greeted_refresh_loop()` | Re-reads `last_greeted.json` every 0.2s (5 Hz) |
| `_prewarm_thread` | `_prewarm_rpc_connections()` | Pre-warms RPC connections at startup (one-time, daemon) |
| `interaction_thread` | `_run_interaction_thread()` | Spawned per interaction, runs the RPC call |

### 3.9 Persistent Data Files

| File | Content | Format |
|---|---|---|
| `memory/learning.json` | Per-person LS values + updated_at | `{"people": {"Alice": {"ls": 2, "updated_at": "..."}}}` |
| `memory/greeted_today.json` | ISO timestamps of today's greetings | `{"Alice": "2026-02-26T09:30:00+01:00"}` |
| `memory/talked_today.json` | ISO timestamps of today's conversations | `{"Alice": "2026-02-26T09:35:00+01:00"}` |
| `memory/last_greeted.json` | Latest greeted entry per person | `{"Alice": {"timestamp": "...", "track_id": 3, ...}}` |

All writes are **atomic** (written to a temp file then `os.replace()`).

### 3.10 SQLite Logging

**Database:** `data_collection/face_selector.db`

| Table | Logged event | Key columns |
|---|---|---|
| `target_selections` | Biggest-bbox candidate after face_id resolves and cooldown passes (logged before eligibility/ss4 gate) | `track_id`, `face_id`, `person_id`, `bbox_area`, `ss`, `ls`, `eligible` |
| `ss_changes` | Social state transitions | `person_id`, `old_ss`, `new_ss` |
| `ls_changes` | Learning state transitions | `person_id`, `old_ls`, `new_ls`, `reward_delta` |

### 3.11 Image Annotation & Visualization

Every face is drawn with:
- **Green box** → currently active interaction target
- **Yellow box** → eligible (ready for interaction)
- **White box** → present but not eligible

Labels drawn above each box:
```
Alice (T:3)                    ← person_id + track_id
ss2 | LS2 | LG:09:30           ← social state, learning state, last greeted time
CLOSE/MUT                      ← distance / attention (3-char)
area=12400                     ← bbox area in pixels²
```

Status overlay (top-left): `Status: BUSY | Faces: 2`

---

## 4. Module: `interactionManager`

### 4.1 Purpose

`interactionManager` is the **dialogue and behavior execution layer**. It:
- Receives RPC commands from `faceSelector` (`run <track_id> <face_id> <state>`)
- Executes the appropriate **social state tree** (SS1/SS2/SS3)
- Uses **Azure OpenAI (via LangChain)** for natural language generation and name extraction
- **Personalizes SS3 conversations** by looking up the recognized face in the Telegram bot's `telegram_bot.db` user-memory table
- Listens to **STT** for user responses
- Sends **TTS speech** through YARP
- Continuously monitors if the target face is still the biggest (abort if not)
- Handles **responsive interactions** (user-initiated greetings, QR feeding)
- Manages the robot's **hunger model**

### 4.2 YARP Ports

| Port | Type | Direction | Purpose |
|---|---|---|---|
| `/interactionManager` | Port (RPC) | IN | Main RPC handle (run / status / quit) |
| `/interactionManager/landmarks:i` | BufferedPortBottle | IN | Face data for target monitoring |
| `/interactionManager/stt:i` | BufferedPortBottle | IN | Speech-to-text transcripts |
| `/interactionManager/speech:o` | Port | OUT | TTS text → Acapela speaker |
| `/interactionManager/camLeft:i` | BufferedPortImageRgb | IN | Camera for QR code reading |
| `/interactionManager/hunger:o` | BufferedPortBottle | OUT | Current hunger state string (`HS1`/`HS2`/`HS3`) published at 1 Hz |

**Lazy RPC Clients (created on first use):**

| Client connects to | Purpose |
|---|---|
| `/interactionInterface` | Send `exe <behaviour>` commands (ao_start, ao_stop, ao_hi, ...) |
| `/objectRecognition` | Submit `name <name> id <track_id>` for face labeling |

### 4.3 State Trees (Interaction Flows)

The module supports 4 social states. Only SS1–SS3 have active trees:

| State | Tree | What happens |
|---|---|---|
| `ss1` | `_run_ss1_tree` | Greet unknown → ask name → extract name → register |
| `ss2` | `_run_ss2_tree` | Greet by name → wait response → chain to SS3 |
| `ss3` | `_run_ss3_tree` | Proactive conversation starter → up to 3 turns |
| `ss4` | no-op | Immediately returns success |

**Hunger override:** If hunger state is `HS3` (starving), or `HS2` + `ss3`, the **hunger feed tree** runs instead of the social tree.

### 4.4 SS1 — Unknown Person

```
┌─────────────────────────────────────────────────────┐
│ SS1: Unknown Person                                 │
│                                                     │
│  ① Run behaviour: ao_hi                            │
│  ② Wait STT response (10s)                         │
│      └─ No response → ABORT: no_response_greeting   │
│  ③ Say "We have not met, what's your name?"        │
│  ④ Wait STT response (10s)                         │
│      └─ No response → ABORT: no_response_name       │
│  ⑤ Extract name (regex + LLM fallback)             │
│      └─ Fail → Say "Sorry, I didn't catch that"     │
│            └─ Retry once                            │
│            └─ Fail → ABORT: name_extraction_failed  │
│  ⑥ Register name via /objectRecognition RPC        │
│  ⑦ Write last_greeted.json                         │
│  ⑧ Say "Nice to meet you"                          │
│                                                     │
│  Result: success=True, final_state=ss3              │
└─────────────────────────────────────────────────────┘
```

**Name extraction pipeline:**
1. **Fast regex:** patterns like `"My name is X"`, `"I'm X"`, `"Call me X"` (supports apostrophes/hyphens in names)
2. **LLM fallback (Azure GPT-5 nano):** strict JSON extraction with schema validation and confidence clamped to `[0.0, 1.0]`

### 4.5 SS2 — Known, Not Greeted

```
┌─────────────────────────────────────────────────────┐
│ SS2: Known, Not Greeted                             │
│                                                     │
│  ① Say "Hello <name>"    (attempt 1)               │
│  ② Wait STT response (10s)                         │
│      └─ Responded → write last_greeted              │
│              → final_state=ss3                      │
│              → chain to _run_ss3_tree()             │
│      └─ No response → retry once                    │
│  ③ Say "Hello <name>"    (attempt 2)               │
│  ④ Wait STT response (10s)                         │
│      └─ Responded → chain to ss3                    │
│      └─ No response → ABORT: no_response_greeting   │
└─────────────────────────────────────────────────────┘
```

Validates `face_id` is a real name (not `"unknown"`, `"unmatched"`, or a digit).

### 4.6 SS3 — Known, Greeted, Not Talked

```
┌─────────────────────────────────────────────────────┐
│ SS3: Short Conversation (max 3 turns)               │
│                                                     │
│  ① Telegram user lookup (best-effort)              │
│     └─ Match found → build user_context string      │
│     └─ No match   → user_context = ""              │
│                                                     │
│  ② Choose opening starter:                         │
│     ├─ user_context set → submit personalised LLM   │
│     │    starter to pool (abort-aware via           │
│     │    _await_future_abortable); falls back to     │
│     │    cached generic starter on abort/failure    │
│     └─ no context     → use cached generic starter  │
│          (pre-fetched in background)                │
│                                                     │
│  ③ Say the starter                                 │
│  ④ Schedule next background starter prefetch       │
│                                                     │
│  Loop (up to 3 turns):                              │
│    ├─ Wait STT response (12s)                       │
│    │    └─ No response → end loop                   │
│    ├─ Turn 1 or 2: LLM generate follow-up           │
│    │    (passes user_context for personalisation)   │
│    ├─ Turn 3 (last): LLM generate closing ack       │
│    │    (passes user_context for personalisation)   │
│    └─ Say robot's reply                             │
│                                                     │
│  ≥1 response → talked=True, final_state=ss4         │
│  0 responses → ABORT: no_response_conversation      │
└─────────────────────────────────────────────────────┘
```

**Personalization:** When a Telegram DB record is found for the current `face_id`, `result["telegram_user_matched"]` is set to `True`. The matched user's profile (name, age, interests, recent update, etc.) is injected into the LLM prompts for the opening question, follow-up turns, and closing acknowledgment.

Note: `SS3_MAX_TIME = 120.0s` is defined in code but currently not enforced in the SS3 loop.

### 4.7 Telegram User Lookup & Personalization

Added in commit `721a58f`. Provides per-person context to the SS3 LLM calls by reading from the Telegram bot's local database.

#### `_lookup_telegram_user(face_name) → Optional[Dict]`

Looks up the recognized `face_id` in `memory/telegram_bot.db` (`user_memory` table).

**Name normalization (`_normalize_name`):** Before any comparison, both the lookup key and DB values are normalized: lowercased, stripped, and Unicode-decomposed (NFD) with combining diacritics removed. This makes matching accent-insensitive — e.g., `André` matches `andre`, `Ñoño` matches `nono`.

**Matching priority (accent-insensitive):**
1. Exact match on `name` field → immediate return
2. Exact match on `nickname` field → immediate return
3. Partial: `face_name` appears as a word in the DB `name` → candidate (keeps searching for exact)
4. Partial: `face_name` appears as a word in the DB `nickname` → weaker candidate

Returns `None` if the DB file is missing, unreadable, or no match is found. Never raises.

#### `_build_face_user_context(record) → str`

Converts a Telegram user record dict into a concise plain-text context string (max 500 chars) suitable for injection into LLM prompts. Includes:

| Field extracted | Example output |
|---|---|
| `name` + `nickname` + `age` | `Their name is Alice. They go by Ali. They are 28 years old.` |
| `favorite_topics` + `likes` (deduplicated, max 5) | `They like: robotics, music, cycling.` |
| `dislikes` (max 3) | `They dislike: loud noise.` |
| `relationship_style` | `Relationship style: playful.` |
| `last_personal_update` (max 80 chars) | `Recent life update: starting a PhD next month.` |
| `inside_jokes` (last one, max 3) | `Inside joke: the watermelon incident.` |
| `trust_level == "close_friend"` | `They consider iCub a close friend.` |

Returns an empty string if the record has no useful data (safe to pass to LLM as empty `user_context`).

---

### 4.8 Hunger / QR Feeding Tree

Triggered when `HungerModel` reports the robot is hungry/starving. Overrides the social tree.

```
Hunger States:
  HS1: level ≥ 60%   (satisfied)
  HS2: 25% ≤ level < 60%  (hungry)
  HS3: level < 25%   (starving)

Trigger conditions:
  HS3 → always replaces SS1/SS2/SS3
  HS2 + ss3 → replaces SS3

Flow:
  ① Say "I'm so hungry, would you feed me please?"
  Loop:
    ├─ Wait for QR scan event (8s timeout)
    ├─ Fed → say "Yummy, thank you so much."
    │        → if HS1 → break (satisfied)
    │        → else → say "I'm still hungry. Give me more."
    └─ Timeout handling → prompt once, then abort on next consecutive timeout
  
  QR Mapping:
    SMALL_MEAL  → +10 hunger
    MEDIUM_MEAL → +25 hunger
    LARGE_MEAL  → +45 hunger

  Timeout policy:
    1st timeout → say "Take a look around, you will find some food for me."
    2nd consecutive timeout → ABORT: no_food_qr
    (timeout counter resets after any successful feed)

  Result: success if ≥1 meal eaten; final_state unchanged (no ss promotion)
```

The **QR reader** runs in its own daemon thread (`_qr_reader_loop`), reading from `/interactionManager/camLeft:i` at ~50fps using `cv2.QRCodeDetector`.

### 4.9 Target Monitor

A dedicated thread runs **alongside every interaction** (at 15 Hz) checking that the interaction target remains **visible** in the landmarks stream. Displacement logic has been removed — the only abort criterion is whether the face with the given `track_id` is present.

```
_target_monitor_loop(track_id, result):
  Loop at 15 Hz:
    ├─ Parse latest landmarks (staleness_sec=5.0 — brief port hiccups ignored)
    ├─ Find face with track_id
    ├─ If found:
    │    └─ Reset last_seen timer
    └─ If not found:
         └─ Wait TARGET_LOST_TIMEOUT (12.0s)
         └─ Still missing → ABORT: target_lost
```

Abort reasons cascade: the monitor sets `abort_event`, which every STT wait loop, speak-and-wait, and LLM future poll checks.

### 4.10 Responsive Interaction Path

The responsive path handles **user-initiated events** that arise independently of the proactive cycle:

#### Responsive Greeting
- **Trigger:** User says `"hello"`, `"hi"`, `"ciao"`, `"good morning"` (matched by regex word boundary search)
- **Condition:** Biggest-bbox known face (no gaze requirement—utterance itself is sufficient signal)
- **Cooldown:** 10 seconds per name
- **Action:** Say `"Hi <name>"` + write `last_greeted`; then wait `SS3_STT_TIMEOUT` for a follow-up utterance. If one is received, enter a full **SS3-style conversation loop** (up to `SS3_MAX_TURNS` turns, with optional Telegram personalisation). The `abort_event` is cleared before the follow-up wait so landmark-monitor leftovers from a prior proactive interaction do not cut the conversation short.

#### Responsive QR Acknowledgment
- **Trigger:** QR scan detected (outside of a proactive interaction)
- **Action:** Say `"yummy, thank you"`

**Safety:** Responsive interactions are **dropped** (not deferred) if a proactive interaction is running. The `run_lock` and `_responsive_active` event prevent any concurrency conflicts.

### 4.11 LLM Integration (Azure OpenAI)

**Backend:** Azure OpenAI via `langchain_openai.AzureChatOpenAI`  
**Env loading order:** `load_dotenv()` then `memory/llm.env` (`override=False`)  
**Request timeout:** `LLM_TIMEOUT = 60.0s`  
**Retries:** 3 attempts with 1s delay

**Prompt externalisation (`prompts.json`):** All LLM prompt templates _and_ fixed speech strings used by `interactionManager` are stored in `prompts.json` under the `"interactionManager"` key and loaded at `configure()` time via `_load_im_prompts()`. Each value has an in-code fallback so the module still works if the file is missing or a key is absent. The two system-prompt constants (`LLM_SYSTEM_DEFAULT`, `LLM_SYSTEM_JSON`) are also overridden from `prompts.json` when the corresponding keys are present.

| `prompts.json` key | Used by |
|---|---|
| `system_default` | Default LLM system prompt for conversational calls |
| `system_json` | System prompt for JSON extraction calls |
| `extract_name_prompt` | LLM name-extraction prompt template (`{utterance}`) |
| `convo_starter_prompt` | LLM conversation-starter generation prompt (generic, no user context) |
| `convo_starter_personalized_prompt` | LLM conversation-starter prompt when Telegram user context is available (`{user_context}`) |
| `followup_prompt` | LLM follow-up reply prompt template (`{last_utterance}`) |
| `followup_personalized_prompt` | LLM follow-up prompt when Telegram user context is available (`{user_context}`, `{last_utterance}`) |
| `closing_ack_prompt` | LLM closing acknowledgment prompt template (`{last_utterance}`) |
| `closing_ack_personalized_prompt` | LLM closing acknowledgment when Telegram user context is available (`{user_context}`, `{last_utterance}`) |
| `convo_starter_fallback` | Fallback starter when LLM fails |
| `followup_fallback` | Fallback follow-up text |
| `closing_ack_fallback` | Fallback closing text |
| `ss3_mid_turn_fallback` | Mid-turn fallback text |
| `ss1_ask_name` | SS1 ask-name utterance |
| `ss1_ask_name_retry` | SS1 retry ask-name utterance |
| `ss1_nice_to_meet` | SS1 closing "Nice to meet you" utterance |
| `ss2_greeting` | SS2 greeting template (`{name}`) |
| `hunger_ask_feed` | Hunger feed request utterance |
| `hunger_thank_feed` | Post-feed thank-you utterance |
| `hunger_still_hungry` | Still-hungry prompt |
| `hunger_look_around` | Look-around prompt (first QR timeout) |
| `responsive_qr_ack_text` | Responsive QR acknowledgment text |
| `responsive_greeting` | Responsive greeting template (`{name}`) |

**Required environment variables (validated at startup):**
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `OPENAI_API_VERSION` (or `AZURE_OPENAI_API_VERSION`)

**Deployments (with defaults):**
- `AZURE_DEPLOYMENT_GPT5_NANO` → `contact-Yogaexperiment_gpt5nano` (all tasks — JSON extraction, name extraction, and conversation)

> `AZURE_DEPLOYMENT_GPT5_MINI` is no longer used; both `llm_extract` and `llm_chat` clients now point to the same `gpt5-nano` deployment.

**Routing and options:**
- `_llm_json(...)` requests route to the `llm_extract` client (nano).
- Conversational generation routes to the `llm_chat` client (also nano).
- Routing logic is preserved for forward-compatibility (can be re-split later).
- `options["num_predict"]` is mapped to `max_completion_tokens`.
- Temperature is not passed (GPT-5 deployment constraint in code comments).

| LLM Function | Purpose | Key params |
|---|---|---|
| `_llm_generate_convo_starter(user_context="")` | One short wellbeing/day question; personalised if `user_context` set | `num_predict`→`max_completion_tokens` (2000) |
| `_llm_generate_followup(last_utterance, history, user_context="")` | Short sentiment-aware follow-up (≤22 words); personalised if `user_context` set | `num_predict`→`max_completion_tokens` (2000) |
| `_llm_generate_closing_acknowledgment(last_utterance, user_context="")` | Warm short closing (4–8 words, no question); personalised if `user_context` set | `num_predict`→`max_completion_tokens` (2000) |
| `_llm_extract_name` | JSON name extraction with schema | `num_predict`→`max_completion_tokens` (2000), strict JSON system prompt |

**LLM thread pool:** Single-worker `ThreadPoolExecutor` — one LLM call at a time.  
Futures are polled with `_await_future_abortable()` which checks `abort_event` every 100ms and can cancel the future early.

**Startup:** On `configure()`, the module:
1. Loads `prompts.json` via `_load_im_prompts()` (overrides hardcoded prompt constants)
2. Creates Azure clients via `setup_azure_llms()` (both pointing to nano)
3. Pre-fetches a conversation starter in the background
4. Pre-warms RPC connections (background thread sends `status` pings to avoid TCP setup latency on first interaction)

### 4.12 Speech Output (TTS)

```python
_speak(text)          → writes Bottle to /interactionManager/speech:o
_speak_and_wait(text) → speak() + estimated wait based on word count
```

**Wait estimation:**
```
wait = word_count / 3.0 + 0.5   (words_per_second=3.0, end_margin=0.5)
wait = clamp(wait, 1.0, 8.0)
```

During `speak_and_wait`, the abort event is checked every 100ms so the robot can be interrupted mid-speech.

### 4.13 STT (Speech-to-Text) Input

Reads from `/interactionManager/stt:i` (connected to `/speech2text/text:o`).

```python
_wait_user_utterance_abortable(timeout):
  Loop until timeout:
    ├─ Check abort_event → return None if set
    ├─ Read stt_port (non-blocking)
    ├─ If text → return stripped text
    └─ sleep 0.1s
    
    GIL compensation: if loop body took >0.5s, extend timeout by that amount
```

**Buffer clearing** (`_clear_stt_buffer`): Done before each expected utterance to discard stale transcripts.

### 4.14 HungerModel

Simulates the robot's "hunger" as a level from 0–100:

```python
HungerModel(drain_hours=5.0, hungry_threshold=60.0, starving_threshold=25.0,
            log_callback=None)

update()  → decrements level based on elapsed time (drains to 0 in drain_hours)
          → calls log_callback("DEBUG", "Hunger: <N>%") each time level drops by 1%
feed(delta, payload) → increments level (capped at 100)
get_state() → "HS1" (≥60), "HS2" (≥25), "HS3" (<25)
```

Thread-safe via internal `_lock`. Updated every `updateModule()` cycle (1 Hz).

**Logging:** The new optional `log_callback` parameter accepts a callable `(level: str, msg: str) → None`. When provided, `HungerModel.update()` fires a `DEBUG` log entry each time the integer percentage drops by 1 point, enabling fine-grained hunger tracking in the module log. `InteractionManagerModule` wires its own `_log` method as the callback.

**Hunger broadcast port:** Each `updateModule()` cycle also writes the current hunger state string to `/interactionManager/hunger:o` (`BufferedPortBottle`). Any external module (e.g. a Telegram bot) can connect and read the hunger level in real time.

### 4.15 RPC Interface

The module exposes an RPC handle at `/interactionManager`.

**Supported commands:**

| Command | Arguments | Returns |
|---|---|---|
| `run` | `<track_id> <face_id> <ss1\|ss2\|ss3\|ss4>` | Compact JSON result |
| `hunger` | `<hs1\|hs2\|hs3>` | Manually set hunger level (hs1=100 %, hs2=59 %, hs3=24 %) |
| `status` / `ping` | — | `{"success":true, "busy":<bool>, ...}` |
| `help` | — | Plain-text command list (one line per command, not JSON) |
| `quit` | — | Shutdown |

**Concurrency:** `run_lock` (non-blocking acquire) ensures only one interaction runs at a time. If busy, returns `{"error": "Another action is running"}`.

**Compact result format (returned to faceSelector):**
```json
{
  "success": true,
  "track_id": 3,
  "name": "Alice",
  "name_extracted": true,
  "abort_reason": null,
  "initial_state": "ss1",
  "final_state": "ss3",
  "interaction_tag": "SS1HS1",
  "hunger_state_start": "HS1",
  "hunger_state_end": "HS1",
  "stomach_level_start": 85.2,
  "stomach_level_end": 85.1,
  "telegram_user_matched": true
}
```

`telegram_user_matched` is set to `true` only when an SS3 interaction successfully looked up and used a Telegram user profile for personalization.

**Abort reason compaction (applied only when the user never spoke):**
- `target_lost` / `target_not_biggest` / `target_monitor_abort` → `"face_disappeared"` (only if `result["talked"]` is falsy)
- Anything else → `"not_responded"` (only if `result["talked"]` is falsy)
- If the user *did* speak (`result["talked"] = True`), no negative abort reason is recorded.

**Success compaction:** `compact_success` is `True` when the interaction explicitly succeeded **or** when the user talked and no abort reason was recorded.

### 4.16 Database

**File:** `data_collection/interaction_manager.db`

| Table | What it stores |
|---|---|
| `interactions` | Full record of every proactive interaction: states, success, abort, transcript |
| `responsive_interactions` | Responsive greeting and QR feed events |

A background `_db_thread` drains the `_db_queue` to avoid blocking the interaction thread.

---

## 5. Cross-Module Data Flow

```
┌───────────────────────────────────────────────────────────────────────┐
│              Complete Interaction Cycle (Proactive + Responsive)      │
│                                                                       │
│  A) Proactive Path (faceSelector-driven)                              │
│  1. Vision landmarks/image → faceSelector                             │
│  2. faceSelector parses faces + computes SS/LS/eligibility            │
│  3. faceSelector selects biggest resolved face (cooldown-aware)       │
│  4. faceSelector → RPC run(track_id, face_id, ss) → interactionMgr    │
│  5. interactionManager executes tree (SS or hunger-feed override),    │
│     with monitor + STT/TTS + behaviors + optional name registration   │
│  6. interactionManager returns compact JSON result                    │
│  7. faceSelector updates memory/LS/cooldown + DB logs                 │
│                                                                       │
│  B) Responsive Path (interactionManager internal, event-driven)       │
│  R1. STT greeting or QR feed event detected                           │
│  R2. If proactive interaction is running (run_lock busy) → DROP event │
│  R3. If idle → run responsive greeting or responsive QR acknowledgment│
│  R4. Execute behaviors + speech, update greeting memory when needed   │
│  R5. Log event in responsive_interactions DB table                    │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 6. State Transition Diagrams

### 6.1 Social State Machine

```
         [ss1: Unknown]
              │
      Greet + Ask Name + Extract Name + Register
              │   Success
              ▼
         [ss3: Known, Greeted, No Talk]
              │
         Conversation (≥1 user turn)
              │   Success
              ▼
         [ss4: Known, Greeted, Talked]    ← TERMINAL (today)
              │
         (next day reset → back to ss2)
              │
              ▼
         [ss2: Known, Not Greeted]
              │
         Say "Hi <name>" + Response received
              │   Success
              └────────────────────────────► [ss3]
```

### 6.2 Learning State Machine

```
         [LS1: Strict]
         ·  SO_CLOSE or CLOSE
         ·  MUTUAL_GAZE only
         ·  time_in_view >= 3.0s
              │ reward +1 or +2
              ▼
         [LS2: Relaxed]
         ·  SO_CLOSE or CLOSE or FAR
         ·  MUTUAL_GAZE or NEAR_GAZE
         ·  time_in_view >= 1.0s
              │ reward +1 or +2
              ▼
         [LS3: Advanced]
         ·  No constraints
         ·  Always eligible

         Any state: reward -1 or -2 → regress one level (min LS1)
         Any state: reward +1 or +2 → advance one level (max LS3)
```

---

## 7. Threading Architecture

### `faceSelector` Threads

```
Main thread (updateModule @ 20Hz)
├── _io_thread           → JSON file saves (queue-driven)
├── _db_thread           → SQLite writes (queue-driven)
├── _lg_refresh_thread   → last_greeted.json re-read (5Hz loop)
├── _prewarm_thread      → pre-warm RPC connections at startup (one-time)
└── interaction_thread   → spawned per interaction
      └── (wait for interactionManager RPC, then process result)
```

### `interactionManager` Threads

```
Main thread (updateModule @ 1Hz) → hunger.update()
RPC handle thread                → respond() (YARP managed)
├── _landmarks_reader_thread     → continuously parses /landmarks:i
├── _db_thread                   → async SQLite writes
├── _qr_reader_thread            → camera QR scanning (50fps)
├── _responsive_thread           → watches STT for user-initiated greetings
├── _prewarm_thread              → pre-warm RPC connections at startup (one-time)
└── [per interaction]:
      ├── _monitor_thread        → target monitor (15Hz)
      ├── LLM future             → single-slot ThreadPoolExecutor
      └── behaviour thread       → ao_hi in SS1 (fire-and-forget)
```

**Locks & Events:**

| Primitive | Purpose |
|---|---|
| `state_lock` (faceSelector) | Protect shared runtime state snapshots (`current_faces`, target metadata, cooldown map, etc.) |
| `_interaction_lock` (faceSelector) | Serialize `interaction_busy` transitions and spawn/finalize interaction decisions |
| `_memory_lock` (faceSelector) | Protect memory dicts (`greeted_today`, `talked_today`, `learning_data`) and JSON I/O snapshots |
| `_last_greeted_lock` (faceSelector) | Protect `_last_greeted_snapshot` for background refresh |
| `run_lock` (interactionManager) | Mutual exclusion for interaction execution |
| `abort_event` (interactionManager) | Signal abort to all interaction sub-steps |
| `_responsive_active` (interactionManager) | Prevent overlap of responsive + proactive |
| `_feed_condition` (interactionManager) | Condition variable for QR feed notification |
| `_faces_lock` (interactionManager) | Protect `_latest_faces` shared with monitor |

---

## 8. Memory Files Reference

All files under `modules/alwaysOn/memory/` (plus `prompts.json` at the module root):

| File | R/W | Owner | Description |
|---|---|---|---|
| `learning.json` | R+W | faceSelector | LS per person, `{"people": {"Alice": {"ls": 2, "updated_at": "..."}}}` |
| `greeted_today.json` | R+W | faceSelector + interactionManager | ISO timestamps of today's greetings |
| `talked_today.json` | R+W | faceSelector | ISO timestamps of today's talks |
| `last_greeted.json` | R+W | interactionManager (write) / faceSelector (read) | Last greeting record per person |
| `prompts.json` | R | interactionManager + telegram_bot | All speech strings and LLM prompt templates. Section `"interactionManager"` used by interactionManager (falls back to hardcoded defaults if absent). Section `"telegram_bot"` used by telegram_bot (hot-reloadable via `reload_prompts` RPC). |
| `memory/telegram_bot.db` | R+W | telegram_bot (write) / interactionManager (read) | Telegram bot's SQLite DB. Contains `subscribers`, `chat_memory`, `user_memory`, and `meta` tables. `user_memory` is queried by interactionManager during SS3 for LLM personalization. Optional for interactionManager — interaction continues normally if absent. |

---

## 9. Key Constants Reference

### `faceSelector`

| Constant | Value | Purpose |
|---|---|---|
| `period` | 0.05s | Main loop rate (20 Hz) |
| `interaction_cooldown` | 5.0s | Legacy fallback (superseded by `_effective_cooldown()`) |
| `cooldown_lively` | 3.0s | Cooldown when STM context label = 1 (active scene) |
| `cooldown_calm` | 15.0s | Cooldown when STM context label = 0 (quiet scene) |
| `cooldown_default` | 5.0s | Cooldown when STM context label = −1 (unknown) |
| `DISAPPEAR_WINDOW_SEC` | 30.0s | Window for counting face_disappeared events |
| `DISAPPEAR_THRESHOLD` | 2 | Events before harsh penalty kicks in |
| `frame_skip_rate` | 0 | Process every frame (0 = no skip) |
| `LS_VALID_DISTANCES (LS1)` | `SO_CLOSE`, `CLOSE` | Allowed distance classes for LS1 eligibility |
| `LS_VALID_DISTANCES (LS2)` | `SO_CLOSE`, `CLOSE`, `FAR` | Allowed distance classes for LS2 eligibility |
| `LS_MIN_TIME_IN_VIEW (LS1)` | 3.0s | Minimum observed dwell time before LS1 interactions |
| `LS_MIN_TIME_IN_VIEW (LS2)` | 1.0s | Minimum observed dwell time before LS2 interactions |

### Vision Feed Assumptions (from `perception.py`)

| Rule | Value |
|---|---|
| Unmatched detections (no MediaPipe landmarks) | Still published to landmarks stream with `attention="UNKNOWN"` and `time_in_view` |
| `SO_CLOSE` threshold | `h_norm > 0.4` |
| `CLOSE` threshold | `0.2 < h_norm <= 0.4` |
| `FAR` threshold | `0.1 < h_norm <= 0.2` |
| `VERY_FAR` threshold | `h_norm <= 0.1` |

### `interactionManager`

| Constant | Value | Purpose |
|---|---|---|
| `period` | 1.0s | Main loop (hunger update) |
| `SS1_STT_TIMEOUT` | 10.0s | Wait for greeting response (SS1) |
| `SS2_STT_TIMEOUT` | 10.0s | Wait for name response (SS1) |
| `SS2_GREET_TIMEOUT` | 10.0s | Wait for greeting response (SS2) |
| `SS3_STT_TIMEOUT` | 12.0s | Wait per conversation turn (SS3) |
| `SS3_MAX_TURNS` | 3 | Maximum conversation turns |
| `SS3_MAX_TIME` | 120.0s | Defined SS3 total-time cap (currently not enforced in loop) |
| `LLM_TIMEOUT` | 60.0s | Maximum LLM wait |
| `MONITOR_HZ` | 15.0 | Target monitor polling rate |
| `TARGET_LOST_TIMEOUT` | 12.0s | Seconds track_id must be continuously absent before declaring target lost |
| `RESPONSIVE_GREET_COOLDOWN_SEC` | 10.0s | Per-name cooldown for reactive greetings |
| `TTS_WORDS_PER_SECOND` | 3.0 | Used to estimate speech duration |

---

## 10. YARP Connection Commands

```bash
# faceSelector
yarp connect /alwayson/vision/landmarks:o  /faceSelector/landmarks:i
yarp connect /icub/camcalib/left/out       /faceSelector/img:i
# (faceSelector auto-connects its RPC ports to interactionManager)
# (faceSelector auto-connects /faceSelector/context:i ← /alwayson/stm/context:o)

# interactionManager
yarp connect /alwayson/vision/landmarks:o  /interactionManager/landmarks:i
yarp connect /speech2text/text:o           /interactionManager/stt:i
yarp connect /icub/cam/left                /interactionManager/camLeft:i
yarp connect /interactionManager/speech:o  /acapelaSpeak/speech:i

# telegramBot
yarp connect /interactionManager/hunger:o  /telegramBot/hunger:i

# RPC test
echo "status" | yarp rpc /interactionManager
echo "run 3 Alice ss2" | yarp rpc /interactionManager

# Manually override hunger level (for testing)
echo "hunger hs3" | yarp rpc /interactionManager

# telegramBot RPC
echo "status"      | yarp rpc /telegramBot/rpc
echo "set_hs HS3"  | yarp rpc /telegramBot/rpc
echo "reload_prompts" | yarp rpc /telegramBot/rpc
```

---

## 11. Module: `telegram_bot`

### 11.1 Purpose

`telegram_bot` is the **remote companion channel** for the iCub robot. It runs as a YARP `RFModule` and exposes iCub as a Telegram chatbot, allowing registered users to chat with the robot from their phones at any time. It:

- Subscribes to iCub's **hunger state** broadcast (`/interactionManager/hunger:o`) and adapts its conversation personality accordingly
- Broadcasts **HS3 (starving) alerts** to all subscribers, begging them to come feed the robot in person
- Maintains **per-user long-term memory** (name, age, likes, dislikes, inside jokes, trust level, etc.), continuously updating it from natural language cues in messages  
- Mirrors that memory to `/memory/telegram_bot.db`, which `interactionManager` reads during SS3 in-person conversations for LLM personalization
- Maintains **per-user conversation memory** (rolling history + periodic LLM-generated summaries) so the chatbot "remembers" past exchanges
- Uses **Azure OpenAI** for all reply generation, with hunger-state-specific system prompt overlays
- All speech strings and prompt templates are externalized to `prompts.json` under the `"telegram_bot"` key

---

### 11.2 YARP Ports & RPC Interface

**YARP Ports:**

| Port | Type | Direction | Purpose |
|---|---|---|---|
| `/{module}/hunger:i` | BufferedPortBottle | IN | Receives hunger state from `/interactionManager/hunger:o` |
| `/{module}/rpc` | Port (RPC) | IN | Management commands (status, set_hs, reload_prompts) |

Default `module_name = "telegramBot"`, overridable via `--name` flag.

**RPC Commands:**

| Command | Arguments | Returns |
|---|---|---|
| `status` | — | JSON object: `effective_hs`, `raw_hs`, `hs_stale`, `subscribers`, `tg_offset`, `tg_thread_alive`, `queue_size` |
| `set_hs` | `HS1\|HS2\|HS3` | Set manual hunger override (bypasses staleness check) |
| `reload_prompts` | — | Hot-reload `prompts.json` without restarting |

---

### 11.3 Message Handling

All Telegram messages arrive via a **long-poll daemon thread** (`_tg_poll_loop`). Updates are queued and drained in `updateModule()` at up to **25 messages per cycle** (10 Hz).

**Supported Telegram commands:**

| Command | Behaviour |
|---|---|
| `/start` | Register subscriber, clear conversation history, send personalised greeting (uses stored name if known) |
| `/reset` | Clear conversation history only (user memory kept) |
| Any text | Full LLM-powered reply, with hunger overlay and user context |

**Text message pipeline (`_on_text`):**
```
1. Upsert subscriber record and update last_seen_at
2. Call _effective_hs() → choose hunger personality overlay
3. Load chat memory (summary + rolling history) from DB
4. Load user record and build a background-context system snippet
5. Inject time context: current day/time and gap since last message
6. Inject HS3 override system prompt if starving, else inject summary → history
7. HS2: inject forced hunger comment if overdue (every HS2_HUNGER_EVERY_N messages)
8. Call _llm_chat(messages) → send reply to Telegram
9. Append turn to history; if SUMMARY_EVERY_TURNS reached → re-summarize via LLM
10. Persist updated memory to DB
```

---

### 11.4 Hunger-Aware Personality

The bot's personality adapts to the robot's current hunger state:

| HS | Behaviour |
|---|---|
| `HS1` (≥ 60 % — satisfied) | Normal friendly chat using `base_system_prompt` |
| `HS2` (25–60 % — hungry) | Normal chat but slips casual hunger side-comments in every `HS2_HUNGER_EVERY_N` messages via `hs2_force_hunger_system` overlay |
| `HS3` (< 25 % — starving) | Full `hs3_override_system` override: every reply pivots back to begging the user to come feed the robot in person (panicked / emotional tone) |

**Staleness guard:** If the hunger port has not received an update for `HS_STALE_SEC = 60 s`, `_effective_hs()` silently falls back to `HS1` regardless of the last received state.

---

### 11.5 HS3 Broadcast System

When `_effective_hs()` **enters** HS3, all subscribers receive an LLM-generated broadcast message immediately. During sustained HS3, periodic re-broadcasts are sent using per-subscriber cooldowns.

```
_maybe_hs3_broadcast() — called every updateModule() cycle:
  ├─ effective_hs != HS3 → return
  ├─ entering HS3 (prev != HS3):
  │    → broadcast to ALL subscribers (no cooldown check)
  └─ sustained HS3:
       → broadcast only to subscribers whose last_broadcast_at
         was > HS3_BROADCAST_COOLDOWN_SEC (30 min) ago
         AND who have not chatted in the last HS3_SKIP_RECENT_SEC (10 min)

Broadcast message:
  → _llm_hs3_broadcast() using hs3_broadcast_system + hs3_broadcast_user prompts
  → Falls back to hs3_broadcast_fallback string if LLM fails
```

---

### 11.6 User Memory & Personalization

`user_memory` is a per-user dict (keyed by Telegram `chat_id`) stored in `memory/telegram_bot.db`. It is read by `interactionManager` during SS3 face-to-face conversations for LLM personalization.

**Fields extracted passively from message content and metadata:**

| Field | How extracted |
|---|---|
| `name` | Telegram `from.first_name` on first message; also regex patterns `"my name is X"`, `"call me X"`, etc. |
| `nickname` | Regex: `"you can call me X"`, `"my nickname is X"`, `"everyone calls me X"`, etc. |
| `age` | Regex: `"I'm 23"`, `"just turned 23"`, `"turning 30 soon"`, `"I'll be 25 next month"`, etc. |
| `likes` | Regex: `"I like/love/enjoy/adore X"`, `"I'm a fan of X"`, `"my favourite is X"`, etc. (max 3; FIFO on overflow) |
| `dislikes` | Regex: `"I hate X"`, `"can't stand X"`, `"not a fan of X"`, etc. (max 5) |
| `favorite_topics` | Regex: `"I'm into X"`, `"I love talking about X"`, `"I nerd out about X"`, etc. (max 5) |
| `last_personal_update` | Regex: life events (`I'm sick`, `I just got promoted`, `my exam is today`, etc.) — max 120 chars |
| `conversation_style` | Derived from message length (short/medium/long), presence of emoji, presence of playful slang |
| `relationship_style` | Set to `"protective"` on empathy signals (`"poor iCub"`, `"are you ok"`, etc.) |
| `inside_jokes` | References that appear **≥ 2 times** across separate messages (`"remember when we..."`, `"the X thing"`, etc.). Unconfirmed candidates expire after 30 days. Max 20 pending candidates, max 5 confirmed jokes. |
| `trust_level` | Escalates from `"friend"` → `"close_friend"` on explicit trust signals (`"I've never told anyone"`, `"you really get me"`, etc.) |

**Migration:** On startup, if `memory/user_memory.json` exists it is imported into the DB once and renamed to `user_memory.json.migrated`.

**Normalization for matching** (`_normalize_for_matching`):
- Smart/curly apostrophes → plain ASCII `'`
- 3+ consecutive identical chars collapsed (`loooove` → `love`)
- Multi-whitespace collapsed to single space

---

### 11.7 Conversation Memory

Per-user conversation state is stored in the `chat_memory` table:

| Field | Description |
|---|---|
| `summary` | LLM-generated summary of past conversation (max 400 chars); re-generated every `SUMMARY_EVERY_TURNS = 8` turns |
| `messages_json` | Rolling window of last `MAX_HISTORY_TURNS × 2 = 20` messages (user + assistant pairs), each with a Unix timestamp |
| `turn_count` | Total turns seen since last `/reset` |

Each user message in history is prefixed with a compact timestamp label (e.g. `[Mon 6 Mar 2026, 11:42 PM, CET]`) injected by `_format_history_label()` when building the LLM messages list. A time-gap note (`"Their previous message was 3 days ago"`) is also injected as a system message to give the LLM temporal awareness.

---

### 11.8 LLM Integration

**Backend:** Azure OpenAI via the `openai` Python SDK (`AzureOpenAI`)

**Env loading order:** `memory/llm.env` then `.env` (both `override=False`)

**Deployment selection (in priority order):**
1. `AZURE_DEPLOYMENT_GPT5_MINI`
2. `AZURE_OPENAI_DEPLOYMENT`
3. `AZURE_DEPLOYMENT`
4. Hard default: `"gpt5-mini"`

**Max tokens:** `TELEGRAM_LLM_MAX_TOKENS` env var (default `4000`). Automatically falls back from `max_completion_tokens` to `max_tokens` if the endpoint rejects the parameter.

**Retries:** 3 attempts with exponential back-off (`0.6 × 2ⁿ s`) on `APIConnectionError`, `APITimeoutError`, `RateLimitError`.

| LLM Function | Purpose |
|---|---|
| `_llm_chat(messages)` | General-purpose chat completion (all reply types) |
| `_llm_summarize(history, user_record)` | Generate compact conversation summary; anchors known facts (name, likes) so they are never dropped |
| `_llm_hs3_broadcast()` | Generate a one-off starving-alert broadcast message |

---

### 11.9 Database Schema

**File:** `memory/telegram_bot.db` (WAL mode, 5 s busy timeout)

| Table | Description |
|---|---|
| `meta` | Key-value store (currently used to persist `tg_offset` across restarts) |
| `subscribers` | One row per Telegram user who sent `/start`: `chat_id`, `started_at`, `last_seen_at`, `last_broadcast_at` |
| `chat_memory` | Per-user conversation memory: `summary`, `messages_json`, `turn_count`, `updated_at` |
| `user_memory` | Per-user long-term profile (JSON blob): name, age, likes, dislikes, inside jokes, trust level, etc. Read by `interactionManager` for SS3 personalization |

---

### 11.10 Threading Architecture

```
Main thread (updateModule @ 10Hz)
  ├─ _read_hunger()             → drains /hunger:i port (non-blocking)
  ├─ _process_tg_updates()      → drains tg_updates queue (max 25 per cycle),
  │                               calls _handle_update() synchronously
  └─ _maybe_hs3_broadcast()     → fires HS3 alert messages if needed

RPC handle thread               → respond() (YARP managed)

_tg_poll_loop (daemon thread)
  └─ Long-polls Telegram getUpdates every 20 s
     Pushes updates into tg_updates queue (drops oldest on overflow)
```

Note: LLM calls in `_on_text()` are **synchronous** — the main thread blocks on the Azure API response. In typical deployments this is acceptable because user messages arrive infrequently; for high-throughput scenarios a future refactor could off-load to a worker thread pool.

---

### 11.11 Prompts (`prompts.json`)

All strings are under the `"telegram_bot"` key in `prompts.json`. Loaded once at `configure()` time; hot-reloadable via `reload_prompts` RPC.

| Key | Used by |
|---|---|
| `base_system_prompt` | Base system personality for all LLM calls |
| `hs_overlays.HS1` / `HS2` / `HS3` | Hunger-state-specific system prompt appended to base |
| `hs3_override_system` | Full system override injected in HS3 per-message calls |
| `hs2_force_hunger_system` | Injected when HS2 hunger comment is overdue |
| `summary_injection` | Template wrapping the conversation summary (placeholder: `{summary}`) |
| `summarize_system` | System prompt for the summarization call |
| `hs3_broadcast_system` / `hs3_broadcast_user` | System + user prompts for HS3 broadcast generation |
| `hs3_broadcast_fallback` | Fallback broadcast text when LLM fails |
| `fallback_hs3` / `fallback_hs2` / `fallback_default` | Per-HS fallback reply when LLM fails |
| `start_greeting_with_name` | `/start` greeting when name is known (placeholder: `{name}`) |
| `start_greeting` | `/start` greeting when name is unknown |
| `reset_reply` | Reply sent after `/reset` |

---

### 11.12 Key Constants

| Constant | Value | Purpose |
|---|---|---|
| `MODULE_HZ` | 10.0 | Main loop rate |
| `HS_STALE_SEC` | 60.0 s | Declare hunger reading stale if no update within this window |
| `TG_POLL_TIMEOUT_SEC` | 20 s | Telegram long-poll timeout per request |
| `TG_HTTP_TIMEOUT_SEC` | 35 s | HTTP request timeout |
| `MAX_HISTORY_TURNS` | 10 | Rolling message window size (in turns; 20 messages) |
| `SUMMARY_EVERY_TURNS` | 8 | Regenerate conversation summary every N turns |
| `MAX_USER_CHARS` | 500 | Truncate incoming message at this length |
| `MAX_REPLY_CHARS` | 4096 | Telegram message limit; long replies are chunked |
| `HS3_BROADCAST_COOLDOWN_SEC` | 1800 s | Per-subscriber cooldown for sustained HS3 re-broadcasts |
| `HS3_SKIP_RECENT_SEC` | 600 s | Skip HS3 re-broadcast if subscriber chatted within this window |
| `JOKE_CANDIDATE_TTL_SEC` | 30 days | Expire unconfirmed inside-joke candidates after this period |
| `JOKE_CANDIDATE_MAX` | 20 | Max pending inside-joke candidates per user |
| `HS2_HUNGER_EVERY_N` | 3 | Force a hunger side-comment after N messages without one (HS2) |
