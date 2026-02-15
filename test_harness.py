#!/usr/bin/env python3
"""
TEST HARNESS - Comprehensive YARP Port Simulator

Simulates all YARP ports required by faceSelector.py and interactionManager.py modules:

PUBLISHERS (simulated sensor outputs):
    /alwayson/vision/landmarks:o    - Face detection landmarks
    /alwayson/vision/img:o          - Camera images (RGB)
    /alwayson/stm/context:o         - Short-term memory context
    /speech2text/text:o             - Speech-to-text output
    /acapelaSpeak/bookmark:o        - TTS bookmark events

RECEIVERS (simulated actuator inputs):
    /acapelaSpeak/speech:i          - TTS speech commands

RPC SERVERS (simulated services):
    /interactionInterface           - Behavior execution
    /objectRecognition/rpc          - Face registration
    /faceTracker/rpc                - Face tracking
    /speech2text/rpc                - STT configuration

SCENARIOS:
    1. SS1: Unknown person, not greeted (with/without response)
    2. SS2: Name acquisition (name provided/not provided)
    3. SS3: Known person greeting (with/without response)
    4. SS4: Multi-turn conversation
    5. Multiple faces with varying spatial states
    6. Learning state progression (LS1-LS4)

Usage:
    python test_harness.py [--scenario <name>] [--interactive]
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import yarp
except ImportError:
    print("ERROR: YARP Python bindings are required.")
    sys.exit(1)


# ==================== Enums and Constants ====================

class Zone(Enum):
    FAR_LEFT = "FAR_LEFT"
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"
    FAR_RIGHT = "FAR_RIGHT"
    UNKNOWN = "UNKNOWN"


class Distance(Enum):
    SO_CLOSE = "SO_CLOSE"
    CLOSE = "CLOSE"
    FAR = "FAR"
    VERY_FAR = "VERY_FAR"
    UNKNOWN = "UNKNOWN"


class Attention(Enum):
    MUTUAL_GAZE = "MUTUAL_GAZE"
    NEAR_GAZE = "NEAR_GAZE"
    AWAY = "AWAY"


class ContextLabel(Enum):
    UNCERTAIN = -1
    CALM = 0
    LIVELY = 1


class SocialState(Enum):
    SS1 = "ss1"  # Unknown, Not Greeted Today
    SS2 = "ss2"  # Unknown, Greeted Today
    SS3 = "ss3"  # Known, Not Greeted Today
    SS4 = "ss4"  # Known, Greeted Today, Not talked to
    SS5 = "ss5"  # Known, Greeted Today, Talked to


# ==================== Data Classes ====================

@dataclass
class SimulatedFace:
    """Represents a face in the simulated scene."""
    face_id: str = "unknown"
    track_id: int = 0
    bbox: Tuple[float, float, float, float] = (200.0, 150.0, 120.0, 140.0)  # x, y, w, h
    zone: Zone = Zone.CENTER
    distance: Distance = Distance.CLOSE
    attention: Attention = Attention.MUTUAL_GAZE
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    cos_angle: float = 0.95
    gaze_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    is_talking: int = 0
    time_in_view: float = 5.0

    def to_bottle(self, bottle: yarp.Bottle):
        """Serialize face data to YARP Bottle format."""
        bottle.addString("face_id")
        bottle.addString(self.face_id)
        bottle.addString("track_id")
        bottle.addInt32(self.track_id)
        bottle.addString("zone")
        bottle.addString(self.zone.value)
        bottle.addString("distance")
        bottle.addString(self.distance.value)
        bottle.addString("attention")
        bottle.addString(self.attention.value)
        bottle.addString("pitch")
        bottle.addFloat64(self.pitch)
        bottle.addString("yaw")
        bottle.addFloat64(self.yaw)
        bottle.addString("roll")
        bottle.addFloat64(self.roll)
        bottle.addString("cos_angle")
        bottle.addFloat64(self.cos_angle)
        bottle.addString("is_talking")
        bottle.addInt32(self.is_talking)
        bottle.addString("time_in_view")
        bottle.addFloat64(self.time_in_view)
        
        # Add bbox as nested list
        bbox_btl = bottle.addList()
        bbox_btl.addString("bbox")
        for v in self.bbox:
            bbox_btl.addFloat64(v)
        
        # Add gaze_direction as nested list
        gaze_btl = bottle.addList()
        gaze_btl.addString("gaze_direction")
        for v in self.gaze_direction:
            gaze_btl.addFloat64(v)


@dataclass
class ScenarioStep:
    """A single step in a test scenario."""
    description: str
    duration: float = 2.0  # seconds
    faces: List[SimulatedFace] = field(default_factory=list)
    context_label: ContextLabel = ContextLabel.CALM
    stt_responses: List[str] = field(default_factory=list)  # STT outputs to send
    expected_speech: Optional[str] = None  # Expected TTS from module


@dataclass
class Scenario:
    """Complete test scenario."""
    name: str
    description: str
    steps: List[ScenarioStep] = field(default_factory=list)


# ==================== Predefined Test Scenarios ====================

def create_scenarios() -> Dict[str, Scenario]:
    """Create all predefined test scenarios."""
    scenarios = {}

    # ---- Scenario 1: SS1 with successful greeting response ----
    scenarios["ss1_success"] = Scenario(
        name="ss1_success",
        description="Unknown person greets and responds - SS1 → SS2",
        steps=[
            ScenarioStep(
                description="Unknown person enters, centered, close, mutual gaze",
                duration=3.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=1,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=2.0
                )],
                context_label=ContextLabel.CALM,
            ),
            ScenarioStep(
                description="Person responds to greeting",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=1,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=5.0
                )],
                context_label=ContextLabel.CALM,
                stt_responses=["Hello!", "Hi there!"],
            ),
        ]
    )

    # ---- Scenario 2: SS1 with no response ----
    scenarios["ss1_no_response"] = Scenario(
        name="ss1_no_response",
        description="Unknown person does not respond to greeting - SS1 fails",
        steps=[
            ScenarioStep(
                description="Unknown person enters but doesn't respond",
                duration=3.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=2,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.AWAY,
                    time_in_view=3.0
                )],
                context_label=ContextLabel.UNCERTAIN,
            ),
            ScenarioStep(
                description="Robot waits for response (timeout)",
                duration=12.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=2,
                    zone=Zone.CENTER,
                    distance=Distance.FAR,
                    attention=Attention.AWAY,
                    time_in_view=15.0
                )],
                context_label=ContextLabel.UNCERTAIN,
                stt_responses=[],  # No response
            ),
        ]
    )

    # ---- Scenario 3: SS2 with name provided ----
    scenarios["ss2_name_provided"] = Scenario(
        name="ss2_name_provided",
        description="Person provides name when asked - SS2 → SS4",
        steps=[
            ScenarioStep(
                description="Robot asks for name",
                duration=3.0,
                faces=[SimulatedFace(
                    face_id="12345",  # 5-digit code = unknown
                    track_id=3,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=10.0
                )],
                context_label=ContextLabel.CALM,
            ),
            ScenarioStep(
                description="Person says their name",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="12345",
                    track_id=3,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=15.0
                )],
                context_label=ContextLabel.CALM,
                stt_responses=["My name is Marco", "I'm Marco"],
            ),
        ]
    )

    # ---- Scenario 4: SS2 without name (person refuses) ----
    scenarios["ss2_no_name"] = Scenario(
        name="ss2_no_name",
        description="Person does not provide name - SS2 fails",
        steps=[
            ScenarioStep(
                description="Robot asks for name",
                duration=3.0,
                faces=[SimulatedFace(
                    face_id="67890",
                    track_id=4,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.NEAR_GAZE,
                    time_in_view=10.0
                )],
                context_label=ContextLabel.LIVELY,
            ),
            ScenarioStep(
                description="Person gives evasive response",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="67890",
                    track_id=4,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.NEAR_GAZE,
                    time_in_view=15.0
                )],
                context_label=ContextLabel.LIVELY,
                stt_responses=["I don't want to say", "None of your business"],
            ),
        ]
    )

    # ---- Scenario 5: SS3 known person greeting ----
    scenarios["ss3_known_greeting"] = Scenario(
        name="ss3_known_greeting",
        description="Known person enters and is greeted by name - SS3 → SS4",
        steps=[
            ScenarioStep(
                description="Known person (Alice) enters",
                duration=3.0,
                faces=[SimulatedFace(
                    face_id="Alice",
                    track_id=5,
                    zone=Zone.LEFT,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=2.0
                )],
                context_label=ContextLabel.CALM,
            ),
            ScenarioStep(
                description="Alice responds to personal greeting",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="Alice",
                    track_id=5,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=7.0
                )],
                context_label=ContextLabel.CALM,
                stt_responses=["Hello iCub!", "Hi, nice to see you again!"],
            ),
        ]
    )

    # ---- Scenario 6: SS4 conversation ----
    scenarios["ss4_conversation"] = Scenario(
        name="ss4_conversation",
        description="Multi-turn conversation - SS4 → SS5",
        steps=[
            ScenarioStep(
                description="Robot starts conversation",
                duration=3.0,
                faces=[SimulatedFace(
                    face_id="Bob",
                    track_id=6,
                    zone=Zone.CENTER,
                    distance=Distance.SO_CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=20.0
                )],
                context_label=ContextLabel.LIVELY,
            ),
            ScenarioStep(
                description="Turn 1: User responds",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="Bob",
                    track_id=6,
                    zone=Zone.CENTER,
                    distance=Distance.SO_CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=25.0
                )],
                context_label=ContextLabel.LIVELY,
                stt_responses=["I'm doing great, thanks for asking!"],
            ),
            ScenarioStep(
                description="Turn 2: Continued conversation",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="Bob",
                    track_id=6,
                    zone=Zone.CENTER,
                    distance=Distance.SO_CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=30.0
                )],
                context_label=ContextLabel.LIVELY,
                stt_responses=["Yes, I went to the park today"],
            ),
            ScenarioStep(
                description="Turn 3: Final exchange",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="Bob",
                    track_id=6,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=35.0
                )],
                context_label=ContextLabel.LIVELY,
                stt_responses=["It was really nice, the weather was perfect"],
            ),
        ]
    )

    # ---- Scenario 7: Multiple faces ----
    scenarios["multi_face"] = Scenario(
        name="multi_face",
        description="Multiple faces with varying eligibility",
        steps=[
            ScenarioStep(
                description="Three faces: unknown (center), known SS5 (left), unknown (right-far)",
                duration=5.0,
                faces=[
                    SimulatedFace(
                        face_id="unknown",
                        track_id=10,
                        bbox=(300.0, 150.0, 100.0, 120.0),
                        zone=Zone.CENTER,
                        distance=Distance.CLOSE,
                        attention=Attention.MUTUAL_GAZE,
                        time_in_view=3.0
                    ),
                    SimulatedFace(
                        face_id="Charlie",  # Known, already talked (SS5)
                        track_id=11,
                        bbox=(50.0, 160.0, 90.0, 110.0),
                        zone=Zone.LEFT,
                        distance=Distance.CLOSE,
                        attention=Attention.MUTUAL_GAZE,
                        time_in_view=30.0
                    ),
                    SimulatedFace(
                        face_id="unknown",
                        track_id=12,
                        bbox=(500.0, 200.0, 80.0, 95.0),
                        zone=Zone.FAR_RIGHT,
                        distance=Distance.VERY_FAR,
                        attention=Attention.AWAY,
                        time_in_view=1.0
                    ),
                ],
                context_label=ContextLabel.CALM,
            ),
        ]
    )

    # ---- Scenario 8: Spatial state variations ----
    scenarios["spatial_states"] = Scenario(
        name="spatial_states",
        description="Face transitions through different spatial states",
        steps=[
            ScenarioStep(
                description="Person far away, not looking",
                duration=3.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=20,
                    zone=Zone.FAR_RIGHT,
                    distance=Distance.VERY_FAR,
                    attention=Attention.AWAY,
                    time_in_view=1.0
                )],
                context_label=ContextLabel.UNCERTAIN,
            ),
            ScenarioStep(
                description="Person approaches, near gaze",
                duration=3.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=20,
                    zone=Zone.RIGHT,
                    distance=Distance.FAR,
                    attention=Attention.NEAR_GAZE,
                    time_in_view=4.0
                )],
                context_label=ContextLabel.CALM,
            ),
            ScenarioStep(
                description="Person in center, close, mutual gaze",
                duration=3.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=20,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=7.0
                )],
                context_label=ContextLabel.CALM,
            ),
            ScenarioStep(
                description="Person very close",
                duration=3.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=20,
                    zone=Zone.CENTER,
                    distance=Distance.SO_CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=10.0
                )],
                context_label=ContextLabel.LIVELY,
            ),
        ]
    )

    # ---- Scenario 9: Context variations ----
    scenarios["context_variations"] = Scenario(
        name="context_variations",
        description="Different ambient context labels",
        steps=[
            ScenarioStep(
                description="Calm environment",
                duration=4.0,
                faces=[SimulatedFace(
                    face_id="Diana",
                    track_id=30,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=5.0
                )],
                context_label=ContextLabel.CALM,
            ),
            ScenarioStep(
                description="Lively environment",
                duration=4.0,
                faces=[SimulatedFace(
                    face_id="Diana",
                    track_id=30,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=9.0
                )],
                context_label=ContextLabel.LIVELY,
            ),
            ScenarioStep(
                description="Uncertain environment",
                duration=4.0,
                faces=[SimulatedFace(
                    face_id="Diana",
                    track_id=30,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.NEAR_GAZE,
                    time_in_view=13.0
                )],
                context_label=ContextLabel.UNCERTAIN,
            ),
        ]
    )

    # ---- Scenario 10: Person talking ----
    scenarios["talking_person"] = Scenario(
        name="talking_person",
        description="Person detected as talking",
        steps=[
            ScenarioStep(
                description="Person starts silent",
                duration=2.0,
                faces=[SimulatedFace(
                    face_id="Eve",
                    track_id=40,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    is_talking=0,
                    time_in_view=2.0
                )],
                context_label=ContextLabel.CALM,
            ),
            ScenarioStep(
                description="Person is talking",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="Eve",
                    track_id=40,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    is_talking=1,
                    time_in_view=7.0
                )],
                context_label=ContextLabel.CALM,
                stt_responses=["I have a question for you"],
            ),
        ]
    )

    # ---- Scenario 11: Full interaction flow (SS1 → SS2 → SS4 → SS5) ----
    scenarios["full_flow"] = Scenario(
        name="full_flow",
        description="Complete interaction: unknown → name → conversation",
        steps=[
            ScenarioStep(
                description="Unknown person approaches",
                duration=3.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=50,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=3.0
                )],
                context_label=ContextLabel.CALM,
            ),
            ScenarioStep(
                description="Person responds to greeting (SS1 → SS2)",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=50,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=8.0
                )],
                context_label=ContextLabel.CALM,
                stt_responses=["Hello!"],
            ),
            ScenarioStep(
                description="Person provides name (SS2 → SS4)",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=50,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=13.0
                )],
                context_label=ContextLabel.CALM,
                stt_responses=["My name is Francesco"],
            ),
            ScenarioStep(
                description="Conversation turn 1 (SS4)",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="Francesco",
                    track_id=50,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=18.0
                )],
                context_label=ContextLabel.CALM,
                stt_responses=["I'm doing well, thank you"],
            ),
            ScenarioStep(
                description="Conversation ends (SS4 → SS5)",
                duration=5.0,
                faces=[SimulatedFace(
                    face_id="Francesco",
                    track_id=50,
                    zone=Zone.CENTER,
                    distance=Distance.CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=23.0
                )],
                context_label=ContextLabel.CALM,
                stt_responses=["Goodbye!"],
            ),
        ]
    )

    # ---- Scenario 12: Learning states test ----
    scenarios["learning_states"] = Scenario(
        name="learning_states",
        description="Test faces at different learning state requirements",
        steps=[
            ScenarioStep(
                description="LS1 face: Any spatial state allowed",
                duration=4.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=60,
                    zone=Zone.FAR_LEFT,  # Would fail LS2+
                    distance=Distance.VERY_FAR,  # Would fail LS2+
                    attention=Attention.AWAY,  # Would fail LS3+
                    time_in_view=2.0
                )],
                context_label=ContextLabel.CALM,
            ),
            ScenarioStep(
                description="LS2 requirements: CENTER zone, FAR distance OK",
                duration=4.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=61,
                    zone=Zone.CENTER,
                    distance=Distance.FAR,
                    attention=Attention.AWAY,  # AWAY still OK for LS2
                    time_in_view=2.0
                )],
                context_label=ContextLabel.CALM,
            ),
            ScenarioStep(
                description="LS3 requirements: CLOSE, NEAR_GAZE minimum",
                duration=4.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=62,
                    zone=Zone.RIGHT,
                    distance=Distance.CLOSE,
                    attention=Attention.NEAR_GAZE,
                    time_in_view=2.0
                )],
                context_label=ContextLabel.CALM,
            ),
            ScenarioStep(
                description="LS4 requirements: SO_CLOSE, MUTUAL_GAZE only",
                duration=4.0,
                faces=[SimulatedFace(
                    face_id="unknown",
                    track_id=63,
                    zone=Zone.LEFT,
                    distance=Distance.SO_CLOSE,
                    attention=Attention.MUTUAL_GAZE,
                    time_in_view=2.0
                )],
                context_label=ContextLabel.CALM,
            ),
        ]
    )

    return scenarios


# ==================== Test Harness Class ====================

class TestHarness:
    """Main test harness managing all simulated YARP ports and scenarios."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._running = False
        self._lock = threading.Lock()

        # Episode and context tracking
        self.episode_id = 0
        self.chunk_id = -1
        self.current_context = ContextLabel.CALM

        # Current scene state
        self.current_faces: List[SimulatedFace] = []
        self.stt_queue: List[str] = []  # Queue of STT responses to send
        self.received_speech: List[str] = []  # Captured TTS commands
        self.rpc_logs: List[Dict] = []  # Captured RPC calls

        # YARP ports - Publishers
        self.landmarks_port: Optional[yarp.BufferedPortBottle] = None
        self.img_port: Optional[yarp.BufferedPortImageRgb] = None
        self.context_port: Optional[yarp.BufferedPortBottle] = None
        self.stt_port: Optional[yarp.BufferedPortBottle] = None
        self.bookmark_port: Optional[yarp.BufferedPortBottle] = None

        # YARP ports - Receivers
        self.speech_in_port: Optional[yarp.BufferedPortBottle] = None

        # YARP RPC servers
        self.interaction_interface_rpc: Optional[yarp.RpcServer] = None
        self.object_recognition_rpc: Optional[yarp.RpcServer] = None
        self.face_tracker_rpc: Optional[yarp.RpcServer] = None
        self.stt_rpc: Optional[yarp.RpcServer] = None

        # Background threads
        self._publish_thread: Optional[threading.Thread] = None
        self._rpc_threads: List[threading.Thread] = []

        # Timing
        self.publish_rate = 20.0  # Hz for landmarks/images
        self.context_rate = 0.2  # Hz for context

        # Image dimensions
        self.img_width = 640
        self.img_height = 480

        # Faces directory for mock registration
        self.faces_dir = "/tmp/test_harness_faces"
        os.makedirs(self.faces_dir, exist_ok=True)

    def log(self, level: str, message: str):
        """Print log message with timestamp."""
        if self.verbose or level in ("ERROR", "WARNING"):
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] [{level}] [TestHarness] {message}")

    def configure(self) -> bool:
        """Open all YARP ports."""
        try:
            # Publishers
            self.landmarks_port = yarp.BufferedPortBottle()
            if not self.landmarks_port.open("/alwayson/vision/landmarks:o"):
                self.log("ERROR", "Failed to open landmarks port")
                return False

            self.img_port = yarp.BufferedPortImageRgb()
            if not self.img_port.open("/alwayson/vision/img:o"):
                self.log("ERROR", "Failed to open image port")
                return False

            self.context_port = yarp.BufferedPortBottle()
            if not self.context_port.open("/alwayson/stm/context:o"):
                self.log("ERROR", "Failed to open context port")
                return False

            self.stt_port = yarp.BufferedPortBottle()
            if not self.stt_port.open("/speech2text/text:o"):
                self.log("ERROR", "Failed to open STT port")
                return False

            self.bookmark_port = yarp.BufferedPortBottle()
            if not self.bookmark_port.open("/acapelaSpeak/bookmark:o"):
                self.log("ERROR", "Failed to open bookmark port")
                return False

            # Receivers
            self.speech_in_port = yarp.BufferedPortBottle()
            if not self.speech_in_port.open("/acapelaSpeak/speech:i"):
                self.log("ERROR", "Failed to open speech input port")
                return False

            # RPC Servers
            self.interaction_interface_rpc = yarp.RpcServer()
            if not self.interaction_interface_rpc.open("/interactionInterface"):
                self.log("ERROR", "Failed to open interactionInterface RPC")
                return False

            self.object_recognition_rpc = yarp.RpcServer()
            if not self.object_recognition_rpc.open("/objectRecognition/rpc"):
                self.log("ERROR", "Failed to open objectRecognition RPC")
                return False

            self.face_tracker_rpc = yarp.RpcServer()
            if not self.face_tracker_rpc.open("/faceTracker/rpc"):
                self.log("ERROR", "Failed to open faceTracker RPC")
                return False

            self.stt_rpc = yarp.RpcServer()
            if not self.stt_rpc.open("/speech2text/rpc"):
                self.log("ERROR", "Failed to open speech2text RPC")
                return False

            self.log("INFO", "All ports opened successfully")
            return True

        except Exception as e:
            self.log("ERROR", f"Configuration failed: {e}")
            return False

    def start(self):
        """Start background publishing and RPC handling threads."""
        self._running = True

        # Start publish thread
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()

        # Start RPC handler threads
        rpc_handlers = [
            (self.interaction_interface_rpc, self._handle_interaction_interface_rpc),
            (self.object_recognition_rpc, self._handle_object_recognition_rpc),
            (self.face_tracker_rpc, self._handle_face_tracker_rpc),
            (self.stt_rpc, self._handle_stt_rpc),
        ]

        for rpc_port, handler in rpc_handlers:
            t = threading.Thread(target=self._rpc_loop, args=(rpc_port, handler), daemon=True)
            t.start()
            self._rpc_threads.append(t)

        # Start speech receiver thread
        speech_thread = threading.Thread(target=self._speech_receiver_loop, daemon=True)
        speech_thread.start()
        self._rpc_threads.append(speech_thread)

        self.log("INFO", "Test harness started")

    def stop(self):
        """Stop all threads and close ports."""
        self._running = False
        time.sleep(0.2)

        for port in [self.landmarks_port, self.img_port, self.context_port,
                     self.stt_port, self.bookmark_port, self.speech_in_port,
                     self.interaction_interface_rpc, self.object_recognition_rpc,
                     self.face_tracker_rpc, self.stt_rpc]:
            if port:
                try:
                    port.interrupt()
                    port.close()
                except Exception:
                    pass

        self.log("INFO", "Test harness stopped")

    # ==================== Publishing ====================

    def _publish_loop(self):
        """Main loop for publishing sensor data."""
        last_landmarks = 0.0
        last_context = 0.0
        landmarks_period = 1.0 / self.publish_rate
        context_period = 1.0 / self.context_rate

        while self._running:
            now = time.time()

            # Publish landmarks at high rate
            if now - last_landmarks >= landmarks_period:
                self._publish_landmarks()
                self._publish_image()
                last_landmarks = now

            # Publish context at lower rate
            if now - last_context >= context_period:
                self._publish_context()
                last_context = now

            time.sleep(0.01)

    def _publish_landmarks(self):
        """Publish current face landmarks."""
        with self._lock:
            faces = self.current_faces.copy()

        bottle = self.landmarks_port.prepare()
        bottle.clear()

        for face in faces:
            face_btl = bottle.addList()
            face.to_bottle(face_btl)

        self.landmarks_port.write()

    def _publish_image(self):
        """Publish a synthetic test image."""
        if self.img_port.getOutputCount() == 0:
            return

        img = self.img_port.prepare()
        img.resize(self.img_width, self.img_height)

        # Create synthetic RGB image with face boxes
        rgb_data = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        rgb_data[:, :, 0] = 100  # Gray-ish background
        rgb_data[:, :, 1] = 100
        rgb_data[:, :, 2] = 100

        # Draw face boxes
        with self._lock:
            for face in self.current_faces:
                x, y, w, h = face.bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                # Clip to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.img_width - 1, x2), min(self.img_height - 1, y2)

                # Draw rectangle (green for mutual gaze, yellow for near, white for away)
                if face.attention == Attention.MUTUAL_GAZE:
                    color = (0, 255, 0)
                elif face.attention == Attention.NEAR_GAZE:
                    color = (255, 255, 0)
                else:
                    color = (255, 255, 255)

                # Top and bottom edges
                rgb_data[y1:y1+2, x1:x2] = color
                rgb_data[y2-2:y2, x1:x2] = color
                # Left and right edges
                rgb_data[y1:y2, x1:x1+2] = color
                rgb_data[y1:y2, x2-2:x2] = color

        # Copy to YARP image
        row = img.getRowSize()
        buf = img.getRawImage()
        out_raw = np.frombuffer(buf, dtype=np.uint8, count=self.img_height * row)
        out_raw = out_raw.reshape((self.img_height, row))
        out_raw[:, :self.img_width * 3] = rgb_data.reshape((self.img_height, self.img_width * 3))

        self.img_port.write()

    def _publish_context(self):
        """Publish current context."""
        bottle = self.context_port.prepare()
        bottle.clear()

        bottle.addInt32(self.episode_id)
        bottle.addInt32(self.chunk_id)
        bottle.addInt32(self.current_context.value)

        self.context_port.write()

    def publish_stt(self, text: str):
        """Publish STT output."""
        bottle = self.stt_port.prepare()
        bottle.clear()

        # Format: [["text", "speaker"]]
        outer = bottle.addList()
        inner = outer.addList()
        inner.addString(text)
        inner.addString("user")

        self.stt_port.write()
        self.log("INFO", f"Published STT: '{text}'")

    def publish_tts_bookmark(self, value: int):
        """Publish TTS bookmark (0=start, 1=end)."""
        bottle = self.bookmark_port.prepare()
        bottle.clear()
        bottle.addInt32(value)
        self.bookmark_port.write()
        self.log("DEBUG", f"Published bookmark: {value}")

    # ==================== Speech Receiver ====================

    def _speech_receiver_loop(self):
        """Receive and process TTS commands."""
        while self._running:
            bottle = self.speech_in_port.read(False)
            if bottle and bottle.size() > 0:
                text = bottle.get(0).asString()
                self.log("INFO", f"Received TTS: '{text}'")
                
                with self._lock:
                    self.received_speech.append(text)

                # Simulate TTS: send start, wait, send end
                threading.Thread(
                    target=self._simulate_tts_playback,
                    args=(text,),
                    daemon=True
                ).start()

            time.sleep(0.05)

    def _simulate_tts_playback(self, text: str):
        """Simulate TTS playback with bookmarks."""
        # Estimate duration based on text length (rough: 10 chars/sec)
        duration = max(0.5, len(text) / 10.0)
        duration = min(duration, 3.0)  # Cap at 3 seconds

        self.publish_tts_bookmark(0)  # Start
        time.sleep(duration)
        self.publish_tts_bookmark(1)  # End

        # After TTS ends, check if we should send an STT response
        time.sleep(0.5)  # Brief pause before "user speaks"
        with self._lock:
            if self.stt_queue:
                response = self.stt_queue.pop(0)
                self.publish_stt(response)

    # ==================== RPC Handlers ====================

    def _rpc_loop(self, port: yarp.RpcServer, handler: Callable):
        """Generic RPC handling loop."""
        while self._running:
            cmd = yarp.Bottle()
            reply = yarp.Bottle()
            
            if port.read(cmd, False):
                handler(cmd, reply)
                port.reply(reply)
            
            time.sleep(0.01)

    def _handle_interaction_interface_rpc(self, cmd: yarp.Bottle, reply: yarp.Bottle):
        """Handle /interactionInterface RPC commands."""
        reply.clear()
        
        if cmd.size() < 1:
            reply.addString("error")
            reply.addString("Empty command")
            return

        command = cmd.get(0).asString()
        self.log("INFO", f"interactionInterface RPC: {command}")

        with self._lock:
            self.rpc_logs.append({
                "port": "interactionInterface",
                "command": command,
                "timestamp": datetime.now().isoformat()
            })

        # Handle common behaviour commands
        if command in ("ao_wave", "ao_hi", "ao_idle", "ao_look"):
            reply.addString("ok")
            self.log("INFO", f"Executed behaviour: {command}")
        elif command == "help":
            reply.addString("ok")
            reply.addString("Commands: ao_wave, ao_hi, ao_idle, ao_look")
        else:
            reply.addString("ok")  # Accept any command

    def _handle_object_recognition_rpc(self, cmd: yarp.Bottle, reply: yarp.Bottle):
        """Handle /objectRecognition/rpc commands."""
        reply.clear()

        if cmd.size() < 1:
            reply.addString("error")
            return

        command = cmd.get(0).asString()
        self.log("INFO", f"objectRecognition RPC: {cmd.toString()}")

        with self._lock:
            self.rpc_logs.append({
                "port": "objectRecognition",
                "command": cmd.toString(),
                "timestamp": datetime.now().isoformat()
            })

        if command == "register":
            # register <name> <track_id>
            if cmd.size() >= 3:
                name = cmd.get(1).asString()
                track_id = cmd.get(2).asInt32()
                
                # Create mock face file
                face_file = os.path.join(self.faces_dir, f"{name}.txt")
                with open(face_file, 'w') as f:
                    f.write(f"track_id={track_id}\nregistered={datetime.now().isoformat()}")
                
                self.log("INFO", f"Registered face: {name} (track_id={track_id})")
                reply.addString("ok")
            else:
                reply.addString("error")
                reply.addString("Usage: register <name> <track_id>")
        else:
            reply.addString("ok")

    def _handle_face_tracker_rpc(self, cmd: yarp.Bottle, reply: yarp.Bottle):
        """Handle /faceTracker/rpc commands."""
        reply.clear()

        if cmd.size() < 1:
            reply.addString("error")
            return

        command = cmd.get(0).asString()
        self.log("INFO", f"faceTracker RPC: {cmd.toString()}")

        with self._lock:
            self.rpc_logs.append({
                "port": "faceTracker",
                "command": cmd.toString(),
                "timestamp": datetime.now().isoformat()
            })

        if command == "track":
            # track <name>
            if cmd.size() >= 2:
                target = cmd.get(1).asString()
                self.log("INFO", f"Tracking face: {target}")
            reply.addString("ok")
        elif command == "stop":
            self.log("INFO", "Stopped face tracking")
            reply.addString("ok")
        else:
            reply.addString("ok")

    def _handle_stt_rpc(self, cmd: yarp.Bottle, reply: yarp.Bottle):
        """Handle /speech2text/rpc commands."""
        reply.clear()

        if cmd.size() < 1:
            reply.addString("error")
            return

        command = cmd.get(0).asString()
        self.log("INFO", f"speech2text RPC: {cmd.toString()}")

        with self._lock:
            self.rpc_logs.append({
                "port": "speech2text",
                "command": cmd.toString(),
                "timestamp": datetime.now().isoformat()
            })

        if command == "set":
            # set <language>
            if cmd.size() >= 2:
                lang = cmd.get(1).asString()
                self.log("INFO", f"STT language set to: {lang}")
            reply.addString("ok")
        elif command == "start":
            self.log("INFO", "STT started")
            reply.addString("ok")
        elif command == "stop":
            self.log("INFO", "STT stopped")
            reply.addString("ok")
        else:
            reply.addString("ok")

    # ==================== Scenario Execution ====================

    def set_faces(self, faces: List[SimulatedFace]):
        """Set current faces in the scene."""
        with self._lock:
            self.current_faces = faces

    def set_context(self, label: ContextLabel):
        """Set current context label."""
        self.current_context = label

    def queue_stt_response(self, text: str):
        """Queue an STT response to be sent after next TTS."""
        with self._lock:
            self.stt_queue.append(text)

    def clear_stt_queue(self):
        """Clear pending STT responses."""
        with self._lock:
            self.stt_queue.clear()

    def run_scenario(self, scenario: Scenario):
        """Execute a test scenario."""
        self.log("INFO", f"{'='*60}")
        self.log("INFO", f"Running scenario: {scenario.name}")
        self.log("INFO", f"Description: {scenario.description}")
        self.log("INFO", f"{'='*60}")

        for i, step in enumerate(scenario.steps):
            self.log("INFO", f"Step {i+1}/{len(scenario.steps)}: {step.description}")

            # Set faces
            self.set_faces(step.faces)

            # Set context
            self.set_context(step.context_label)

            # Queue STT responses
            for response in step.stt_responses:
                self.queue_stt_response(response)

            # Wait for step duration
            time.sleep(step.duration)

        self.log("INFO", f"Scenario '{scenario.name}' completed")
        self.log("INFO", f"{'='*60}")

    def run_all_scenarios(self, delay_between: float = 3.0):
        """Run all predefined scenarios in sequence."""
        scenarios = create_scenarios()

        for name, scenario in scenarios.items():
            self.run_scenario(scenario)
            time.sleep(delay_between)

    def run_interactive_mode(self):
        """Run in interactive mode allowing manual control."""
        self.log("INFO", "Interactive mode started")
        self.log("INFO", "Commands: faces, context, stt, scenario, list, status, quit")

        scenarios = create_scenarios()

        while self._running:
            try:
                cmd = input("\n[TestHarness]> ").strip().lower()

                if not cmd:
                    continue

                parts = cmd.split()
                command = parts[0]

                if command == "quit" or command == "q":
                    break

                elif command == "list":
                    print("\nAvailable scenarios:")
                    for name, scen in scenarios.items():
                        print(f"  {name}: {scen.description}")

                elif command == "scenario" or command == "s":
                    if len(parts) < 2:
                        print("Usage: scenario <name>")
                        continue
                    name = parts[1]
                    if name in scenarios:
                        self.run_scenario(scenarios[name])
                    else:
                        print(f"Unknown scenario: {name}")

                elif command == "faces":
                    if len(parts) < 2:
                        print("Usage: faces <count> OR faces clear")
                        continue
                    if parts[1] == "clear":
                        self.set_faces([])
                        print("Cleared all faces")
                    else:
                        count = int(parts[1])
                        faces = []
                        for i in range(count):
                            faces.append(SimulatedFace(
                                face_id=f"person_{i}" if random.random() > 0.5 else "unknown",
                                track_id=100 + i,
                                bbox=(100 + i * 150, 100, 100, 120),
                                zone=random.choice(list(Zone)),
                                distance=random.choice(list(Distance)),
                                attention=random.choice(list(Attention)),
                                time_in_view=random.uniform(1.0, 20.0)
                            ))
                        self.set_faces(faces)
                        print(f"Added {count} random faces")

                elif command == "context" or command == "c":
                    if len(parts) < 2:
                        print("Usage: context calm|lively|uncertain")
                        continue
                    label = parts[1]
                    if label == "calm":
                        self.set_context(ContextLabel.CALM)
                    elif label == "lively":
                        self.set_context(ContextLabel.LIVELY)
                    elif label == "uncertain":
                        self.set_context(ContextLabel.UNCERTAIN)
                    else:
                        print("Invalid context. Use: calm, lively, uncertain")
                        continue
                    print(f"Context set to: {label}")

                elif command == "stt":
                    if len(parts) < 2:
                        print("Usage: stt <text to send>")
                        continue
                    text = " ".join(parts[1:])
                    self.publish_stt(text)

                elif command == "status":
                    with self._lock:
                        print(f"\nCurrent state:")
                        print(f"  Faces: {len(self.current_faces)}")
                        for f in self.current_faces:
                            print(f"    - {f.face_id} (track={f.track_id}, zone={f.zone.value}, " +
                                  f"dist={f.distance.value}, attn={f.attention.value})")
                        print(f"  Context: {self.current_context.name}")
                        print(f"  STT queue: {len(self.stt_queue)} messages")
                        print(f"  Received TTS: {len(self.received_speech)} messages")
                        print(f"  RPC logs: {len(self.rpc_logs)} calls")

                elif command == "help" or command == "h":
                    print("\nCommands:")
                    print("  list              - List available scenarios")
                    print("  scenario <name>   - Run a specific scenario")
                    print("  faces <count>     - Add random faces")
                    print("  faces clear       - Remove all faces")
                    print("  context <label>   - Set context (calm/lively/uncertain)")
                    print("  stt <text>        - Send STT output")
                    print("  status            - Show current state")
                    print("  quit              - Exit")

                else:
                    print(f"Unknown command: {command}. Type 'help' for commands.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


# ==================== Continuous Publishing Mode ====================

def run_continuous_mode(harness: TestHarness, scenario_name: Optional[str] = None):
    """Run continuous publishing with cycling scenarios."""
    scenarios = create_scenarios()

    if scenario_name:
        if scenario_name not in scenarios:
            print(f"Unknown scenario: {scenario_name}")
            print(f"Available: {', '.join(scenarios.keys())}")
            return
        scenario_list = [scenarios[scenario_name]]
    else:
        scenario_list = list(scenarios.values())

    print(f"\nContinuous mode: cycling through {len(scenario_list)} scenario(s)")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            for scenario in scenario_list:
                harness.run_scenario(scenario)
                time.sleep(2.0)  # Pause between scenarios
    except KeyboardInterrupt:
        print("\nStopping continuous mode...")


# ==================== Main Entry Point ====================

def main():
    parser = argparse.ArgumentParser(
        description="Test Harness for faceSelector and interactionManager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_harness.py                      # Interactive mode
    python test_harness.py --continuous         # Cycle all scenarios continuously
    python test_harness.py --scenario ss1_success  # Run specific scenario continuously
    python test_harness.py --list               # List available scenarios
    python test_harness.py --quiet              # Less verbose output
        """
    )
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode (default)")
    parser.add_argument("--continuous", "-c", action="store_true",
                       help="Run scenarios continuously")
    parser.add_argument("--scenario", "-s", type=str, default=None,
                       help="Specific scenario to run (use with --continuous)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available scenarios and exit")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Reduce output verbosity")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Run all scenarios once and exit")

    args = parser.parse_args()

    # List scenarios
    if args.list:
        scenarios = create_scenarios()
        print("\nAvailable test scenarios:")
        print("=" * 60)
        for name, scenario in scenarios.items():
            print(f"\n  {name}")
            print(f"    Description: {scenario.description}")
            print(f"    Steps: {len(scenario.steps)}")
        print()
        return 0

    # Initialize YARP
    yarp.Network.init()
    if not yarp.Network.checkNetwork():
        print("ERROR: YARP network not available")
        print("Start yarpserver first: yarpserver --write")
        return 1

    # Create and configure harness
    harness = TestHarness(verbose=not args.quiet)

    if not harness.configure():
        print("ERROR: Failed to configure test harness")
        return 1

    harness.start()

    print("\n" + "=" * 60)
    print("TEST HARNESS - YARP Port Simulator")
    print("=" * 60)
    print("\nPublishing on ports:")
    print("  /alwayson/vision/landmarks:o")
    print("  /alwayson/vision/img:o")
    print("  /alwayson/stm/context:o")
    print("  /speech2text/text:o")
    print("  /acapelaSpeak/bookmark:o")
    print("\nReceiving on ports:")
    print("  /acapelaSpeak/speech:i")
    print("\nRPC servers:")
    print("  /interactionInterface")
    print("  /objectRecognition/rpc")
    print("  /faceTracker/rpc")
    print("  /speech2text/rpc")
    print("\n" + "=" * 60)

    # Print connection commands
    print("\nTo connect faceSelector:")
    print("  yarp connect /alwayson/vision/landmarks:o /faceSelector/landmarks:i")
    print("  yarp connect /alwayson/vision/img:o /faceSelector/img:i")
    print("\nTo connect interactionManager:")
    print("  yarp connect /alwayson/stm/context:o /interactionManager/context:i")
    print("  yarp connect /alwayson/vision/landmarks:o /interactionManager/landmarks:i")
    print("  yarp connect /speech2text/text:o /interactionManager/stt:i")
    print("  yarp connect /acapelaSpeak/bookmark:o /interactionManager/acapela_bookmark:i")
    print("  yarp connect /interactionManager/speech:o /acapelaSpeak/speech:i")
    print("\n" + "=" * 60 + "\n")

    try:
        if args.all:
            # Run all scenarios once
            harness.run_all_scenarios(delay_between=2.0)
        elif args.continuous:
            # Continuous mode
            run_continuous_mode(harness, args.scenario)
        else:
            # Interactive mode (default)
            harness.run_interactive_mode()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        harness.stop()
        yarp.Network.fini()
        print("Test harness shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())