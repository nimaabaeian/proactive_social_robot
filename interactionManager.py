"""
interactionManager.py - YARP RFModule for Social Interaction State Trees

Uses Ollama LLM (Phi-3 mini) for natural language understanding and generation.

YARP Connections (run after starting):
    yarp connect /alwayson/stm/context:o /interactionManager/context:i
    yarp connect /alwayson/vision/landmarks:o /interactionManager/landmarks:i
    yarp connect /speech2text/text:o /interactionManager/stt:i
    yarp connect /acapelaSpeak/bookmark:o /interactionManager/acapela_bookmark:i
    yarp connect /interactionManager/speech:o /acapelaSpeak/speech:i

RPC Usage:
    echo "run <track_id> <face_id> <ss1|ss2|ss3|ss4>" | yarp rpc /interactionManager
"""

import glob
import json
import os
import random
import sqlite3
import string
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

import yarp


class InteractionManagerModule(yarp.RFModule):

    # ==================== Constants ====================
    
    FACES_DIR = "/usr/local/src/robot/cognitiveInteraction/objectRecognition/modules/objectRecognition/faces"
    OLLAMA_URL = "http://localhost:11434"
    LLM_MODEL = "phi3:mini"
    DB_FILE = "interaction_data.db"
    
    # Timeouts (seconds)
    TTS_TIMEOUT = 30.0
    STT_TIMEOUT = 10.0
    FILE_VERIFY_TIMEOUT = 5.0
    LLM_TIMEOUT = 60.0
    
    # SS4 limits
    SS4_MAX_TURNS = 5
    SS4_MAX_TIME = 120.0
    
    # Context labels
    CONTEXT_LABELS = {-1: "uncertain", 0: "calm", 1: "lively"}
    
    # Valid social states
    VALID_STATES = {"ss1", "ss2", "ss3", "ss4"}

    # ==================== Lifecycle ====================

    def __init__(self):
        super().__init__()
        self.module_name = "interactionManager"
        self.period = 1.0
        self._running = True
        self.run_lock = threading.Lock()
        self.log_buffer: List[Dict] = []
        
        # YARP ports (initialized in configure)
        # Note: RPC is built into RFModule, no separate rpc_port needed
        self.context_port: Optional[yarp.BufferedPortBottle] = None
        self.landmarks_port: Optional[yarp.BufferedPortBottle] = None
        self.stt_port: Optional[yarp.BufferedPortBottle] = None
        self.bookmark_port: Optional[yarp.BufferedPortBottle] = None
        self.speech_port: Optional[yarp.Port] = None
        
        # Error recovery
        self.llm_retry_attempts = 3
        self.llm_retry_delay = 1.0
        self.ollama_last_check = 0.0
        self.ollama_check_interval = 60.0  # Check Ollama health every 60s

    def configure(self, rf: yarp.ResourceFinder) -> bool:
        """Configure module and open YARP ports."""
        try:
            if rf.check("name"):
                self.module_name = rf.find("name").asString()
            
            # Set module name - RFModule handles RPC automatically via respond()
            self.setName(self.module_name)
            
            # Create port objects as instance variables
            self.context_port = yarp.BufferedPortBottle()
            self.landmarks_port = yarp.BufferedPortBottle()
            self.stt_port = yarp.BufferedPortBottle()
            self.bookmark_port = yarp.BufferedPortBottle()
            self.speech_port = yarp.Port()
            
            # Open all ports
            ports = [
                (self.context_port, "context:i"),
                (self.landmarks_port, "landmarks:i"),
                (self.stt_port, "stt:i"),
                (self.bookmark_port, "acapela_bookmark:i"),
                (self.speech_port, "speech:o"),
            ]
            
            for port, suffix in ports:
                port_name = f"/{self.module_name}/{suffix}"
                if not port.open(port_name):
                    self._log("ERROR", f"Failed to open {port_name}")
                    return False
                self._log("INFO", f"Opened {port_name}")
            
            # Initialize tracking file
            self._ensure_json_file("last_greeted.json", [])
            
            # Initialize database
            self._init_db()
            
            self._log("INFO", "InteractionManagerModule configured successfully")
            return True
            
        except Exception as e:
            self._log("ERROR", f"Configuration failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def interruptModule(self) -> bool:
        """Interrupt all ports."""
        self._log("INFO", "Interrupting module...")
        for port in [self.context_port, self.landmarks_port, 
                     self.stt_port, self.bookmark_port, self.speech_port]:
            if port:
                port.interrupt()
        return True

    def close(self) -> bool:
        """Close all ports."""
        self._log("INFO", "Closing module...")
        for port in [self.context_port, self.landmarks_port,
                     self.stt_port, self.bookmark_port, self.speech_port]:
            if port:
                port.close()
        return True

    def getPeriod(self) -> float:
        return self.period

    def updateModule(self) -> bool:
        return self._running

    # ==================== RPC Handler ====================

    def respond(self, cmd: yarp.Bottle, reply: yarp.Bottle) -> bool:
        """Handle RPC commands: run, help, quit."""
        reply.clear()
        
        try:
            if cmd.size() < 1:
                return self._reply_error(reply, "Empty command")
            
            command = cmd.get(0).asString()
            self._log("DEBUG", f"RPC command received: {command}")
            
            # Help command
            if command == "help":
                return self._reply_ok(reply, {
                    "success": True,
                    "commands": ["run <track_id> <face_id> <ss1|ss2|ss3|ss4>", "help", "quit"]
                })
            
            # Quit command
            if command == "quit":
                self._running = False
                self.stopModule()
                return self._reply_ok(reply, {"success": True, "message": "Shutting down"})
            
            # Run command
            if command != "run":
                return self._reply_error(reply, f"Unknown command: {command}")
            
            if cmd.size() < 4:
                return self._reply_error(reply, "Usage: run <track_id> <face_id> <ss1|ss2|ss3|ss4>")
            
            track_id = cmd.get(1).asInt32()
            face_id = cmd.get(2).asString()
            social_state = cmd.get(3).asString().lower()
            
            if social_state not in self.VALID_STATES:
                return self._reply_error(reply, f"Invalid social state: {social_state}")
            
            # Acquire lock IMMEDIATELY before executing interaction
            if not self.run_lock.acquire(blocking=False):
                self._log("WARNING", "Another interaction already in progress")
                return self._reply_error(reply, "Another action is running")
            
            try:
                self.log_buffer = []
                self._log("INFO", "=== Starting new interaction ===")
                self._log("INFO", f"Parameters: track_id={track_id}, face_id={face_id}, state={social_state}")
                self.ensure_stt_ready("english")
                self._log("INFO", f"Starting {social_state} for track_id={track_id}, face_id={face_id}")
                
                result = self._execute_state_tree(track_id, face_id, social_state)
                self._save_to_db(track_id, face_id, social_state, result)
                
                ordered_trace = self._build_ordered_trace(result, track_id, face_id, social_state)
                reply.addString("ok")
                reply.addString(json.dumps(ordered_trace, ensure_ascii=False))
            finally:
                self.run_lock.release()
            
            return True
            
        except Exception as e:
            self._log("ERROR", f"Exception in respond: {e}")
            return self._reply_error(reply, str(e))

    def _reply_ok(self, reply: yarp.Bottle, data: Dict) -> bool:
        """Send success reply with simplified two-string format."""
        reply.addString("ok")
        reply.addString(json.dumps(data, ensure_ascii=False))
        return True

    def _reply_error(self, reply: yarp.Bottle, error: str) -> bool:
        """Send error reply with simplified two-string format."""
        reply.addString("ok")
        reply.addString(json.dumps({"success": False, "error": error, "logs": self.log_buffer}, ensure_ascii=False))
        return True

    # ==================== State Tree Execution ====================

    def _execute_state_tree(self, track_id: int, face_id: str, social_state: str) -> Dict[str, Any]:
        """Execute state tree with automatic transitions."""
        result = {
            "success": False,
            "initial_state": social_state,
            "final_state": social_state,
            "transitions": [],
            "logs": []
        }
        
        state_handlers = {
            "ss1": self.run_ss1,
            "ss2": self.run_ss2,
            "ss3": self.run_ss3,
            "ss4": self.run_ss4,
        }
        
        current_state = social_state
        current_face_id = face_id
        
        while current_state in state_handlers:
            self._log("INFO", f"Executing state: {current_state}")
            state_result = state_handlers[current_state](track_id, current_face_id)
            
            result[f"{current_state}_result"] = state_result
            result["final_state"] = current_state
            
            next_state = state_result.get("next_state")
            if next_state:
                result["transitions"].append(f"{current_state} -> {next_state}")
                new_name = state_result.get("extracted_name")
                if new_name:
                    current_face_id = new_name
                
                if next_state == "ss5":
                    result["final_state"] = "ss5"
                    result["success"] = True
                    break
                current_state = next_state
            else:
                result["success"] = state_result.get("success", False)
                if current_state == "ss4" and result["success"]:
                    result["final_state"] = "ss5"
                break
        
        result["logs"] = self.log_buffer.copy()
        return result

    # ==================== State SS1: Initial Greeting ====================

    def run_ss1(self, track_id: int, face_id: str) -> Dict[str, Any]:
        """SS1: Register face, greet, detect response → SS2 on success."""
        result = self._init_result(face_submission="failed", greet_attempt="failed", 
                                    response_detected=False, assigned_code=None)
        
        try:
            # Step 0: Register face with unique code
            self._log("INFO", "SS1 Step 1/4: Generating unique face code")
            code = self._generate_unique_code()
            result["assigned_code"] = code
            self._log("INFO", f"SS1: Generated code '{code}'")
            
            self._log("INFO", "SS1 Step 2/4: Registering face with objectRecognition")
            if not self._register_face(code, track_id) or not self._verify_face_registration(code):
                self._log("ERROR", "SS1: Face registration failed")
                return result
            
            result["face_submission"] = "successful"
            self._log("INFO", f"SS1: Face registered successfully as '{code}'")
            
            # Step 1: Read context
            self._log("INFO", "SS1: Reading environment context")
            context = self._read_and_store_context(result)
            self._log("INFO", f"SS1: Context = {context.get('label_str', 'unknown')}")
            
            # Step 2: Greet based on context
            self._log("INFO", "SS1 Step 3/4: Performing greeting gesture")
            behaviour = "ao_wave" if context.get("label_int") == 0 else "ao_hi"
            self._log("INFO", f"SS1: Executing behaviour '{behaviour}'")
            if self._execute_behaviour(behaviour):
                result["greet_attempt"] = "successful"
                self._log("INFO", "SS1: Waiting for TTS to complete")
                self.wait_tts_end(self.TTS_TIMEOUT)
            
            # Step 3: Log greeting
            self._log("INFO", "SS1: Recording greeting in memory")
            self._write_last_greeted(track_id, face_id, code)
            
            # Step 4: Detect response
            self._log("INFO", "SS1 Step 4/4: Waiting for user response")
            if self._detect_greeting_response(result):
                self._log("INFO", "SS1: Response detected! → Transitioning to SS2")
                result["success"] = True
                result["next_state"] = "ss2"
            else:
                self._log("WARNING", "SS1: No response detected, interaction ending")
            
            return result
            
        except Exception as e:
            return self._handle_error(result, "SS1", e)

    # ==================== State SS2: Name Acquisition ====================

    def run_ss2(self, track_id: int, face_id: str) -> Dict[str, Any]:
        """SS2: Ask name, extract via LLM → SS4 on success."""
        result = self._init_result(attempts=0, extracted_name=None, 
                                    name_confidence=0.0, rename_success=False)
        
        try:
            context = self._read_and_store_context(result)
            max_attempts = 2 if context.get("label_int") != 0 else 1
            
            for attempt in range(max_attempts):
                result["attempts"] = attempt + 1
                self._log("INFO", f"SS2: === Attempt {attempt + 1}/{max_attempts} ===")
                
                # Ask name
                self._log("INFO", f"SS2: Generating name question using LLM")
                ask_text = self._llm_generate_ask_name() or "What is your name?"
                self._log("INFO", f"SS2: Asking: '{ask_text}'")
                self._speak(ask_text)
                self.wait_tts_end(self.TTS_TIMEOUT)
                self._clear_stt_buffer()
                
                # Wait for response
                self._log("INFO", f"SS2: Listening for response (timeout={self.STT_TIMEOUT}s)")
                utterance = self.wait_user_utterance(self.STT_TIMEOUT)
                result["user_utterance"] = utterance
                
                if not utterance:
                    self._log("WARNING", f"SS2: No speech detected in attempt {attempt + 1}")
                    continue
                
                self._log("INFO", f"SS2: Heard: '{utterance}'")
                
                # Check for valid response
                self._log("INFO", "SS2: Checking if response is valid")
                if not self._llm_detect_any_response(utterance).get("responded"):
                    self._log("WARNING", "SS2: LLM classified as non-response")
                    continue
                
                # Extract name
                self._log("INFO", "SS2: Extracting name from utterance")
                extraction = self._llm_extract_name(utterance)
                result["name_extraction_result"] = extraction
                
                if extraction.get("answered") and extraction.get("name"):
                    name = extraction["name"]
                    result["extracted_name"] = name
                    result["name_confidence"] = extraction.get("confidence", 0.0)
                    self._log("INFO", f"SS2: Name extracted: '{name}' (confidence={result['name_confidence']:.2f})")
                    
                    # Update face file and tracker
                    self._log("INFO", "SS2: Updating face recognition system")
                    code = self._find_face_code_for_track(track_id) or face_id
                    result["rename_success"] = self._rename_face_file(code, name)
                    self._set_face_tracker(name)
                    self._update_last_greeted_name(track_id, name)
                    
                    self._log("INFO", f"SS2: Success! → Transitioning to SS4")
                    result["success"] = True
                    result["next_state"] = "ss4"
                    return result
                else:
                    self._log("WARNING", "SS2: Could not extract name from response")
            
            self._log("INFO", "SS2: Failed after all attempts")
            return result
            
        except Exception as e:
            return self._handle_error(result, "SS2", e)

    # ==================== State SS3: Known Person Greeting ====================

    def run_ss3(self, track_id: int, face_id: str) -> Dict[str, Any]:
        """SS3: Greet by name → SS4 on response."""
        result = self._init_result(attempts=0, response_detected=False)
        
        try:
            context = self._read_and_store_context(result)
            self._set_face_tracker(face_id)
            
            max_attempts = 2 if context.get("label_int") != 0 else 1
            
            for attempt in range(max_attempts):
                result["attempts"] = attempt + 1
                self._log("INFO", f"SS3: === Attempt {attempt + 1}/{max_attempts} ===")
                
                self._log("INFO", f"SS3: Greeting known person '{face_id}'")
                self._speak(f"Hello {face_id}")
                self.wait_tts_end(self.TTS_TIMEOUT)
                self._clear_stt_buffer()
                
                self._log("INFO", "SS3: Waiting for greeting response")
                if self._detect_greeting_response(result):
                    self._log("INFO", "SS3: Response detected! → Transitioning to SS4")
                    result["success"] = True
                    result["next_state"] = "ss4"
                    return result
                else:
                    self._log("WARNING", f"SS3: No response in attempt {attempt + 1}")
            
            self._log("INFO", "SS3: Failed after all attempts")
            return result
            
        except Exception as e:
            return self._handle_error(result, "SS3", e)

    # ==================== State SS4: Conversation ====================

    def run_ss4(self, track_id: int, face_id: str) -> Dict[str, Any]:
        """SS4: Conversation loop → SS5 if user responds at least once."""
        result = self._init_result(turns=0, user_responses=[], robot_utterances=[])
        
        try:
            start_time = time.time()
            self._read_and_store_context(result)
            self._set_face_tracker(face_id)
            
            # Conversation starter
            self._log("INFO", "SS4: Generating conversation starter using LLM")
            starter = self._llm_generate_convo_starter() or "How are you doing today?"
            result["robot_utterances"].append(starter)
            self._log("INFO", f"SS4: Starting conversation: '{starter}'")
            self._speak(starter)
            self.wait_tts_end(self.TTS_TIMEOUT)
            self._clear_stt_buffer()
            
            # Conversation loop
            self._log("INFO", f"SS4: Starting conversation loop (max {self.SS4_MAX_TURNS} turns, {self.SS4_MAX_TIME}s timeout)")
            user_responded = False
            
            while result["turns"] < self.SS4_MAX_TURNS:
                elapsed = time.time() - start_time
                if elapsed > self.SS4_MAX_TIME:
                    self._log("INFO", f"SS4: Time limit reached ({elapsed:.1f}s)")
                    break
                
                self._log("INFO", f"SS4: Listening for user response (turn {result['turns']+1}/{self.SS4_MAX_TURNS})")
                utterance = self.wait_user_utterance(self.STT_TIMEOUT)
                if not utterance:
                    self._log("INFO", "SS4: No response detected, ending conversation")
                    break
                
                result["turns"] += 1
                result["user_responses"].append(utterance)
                user_responded = True
                self._log("INFO", f"SS4 Turn {result['turns']}: User said: '{utterance}'")
                
                self._log("INFO", "SS4: Generating followup response using LLM")
                followup = self._llm_generate_followup(utterance, result["user_responses"]) or "I see."
                result["robot_utterances"].append(followup)
                self._log("INFO", f"SS4: Robot responds: '{followup}'")
                self._speak(followup)
                self.wait_tts_end(self.TTS_TIMEOUT)
                self._clear_stt_buffer()
            
            if user_responded:
                result["success"] = True
                result["next_state"] = "ss5"
                self._log("INFO", f"SS4: Conversation successful ({result['turns']} turns) → SS5")
            else:
                self._log("WARNING", "SS4: User never responded, interaction failed")
            
            return result
            
        except Exception as e:
            return self._handle_error(result, "SS4", e)

    # ==================== YARP Port Helpers ====================

    def read_context_latest(self) -> Dict[str, Any]:
        """Read latest context: episode_id, chunk_id, label."""
        result = {"episode_id": None, "chunk_id": None, "label_int": -1, "label_str": "uncertain"}
        
        try:
            bottle = self.context_port.read(False)
            if bottle and bottle.size() >= 3:
                result["episode_id"] = bottle.get(0).asInt32()
                result["chunk_id"] = bottle.get(1).asInt32()
                label = bottle.get(2).asInt32()
                result["label_int"] = label
                result["label_str"] = self.CONTEXT_LABELS.get(label, "uncertain")
        except Exception as e:
            self._log("WARNING", f"Context read failed: {e}")
        
        return result

    def parse_landmarks_latest(self) -> List[Dict]:
        """Parse face landmarks from vision port."""
        landmarks = []
        try:
            bottle = self.landmarks_port.read(False)
            if bottle:
                for i in range(bottle.size()):
                    face = bottle.get(i).asList()
                    if face:
                        data = self._parse_face_bottle(face)
                        if data:
                            landmarks.append(data)
        except Exception as e:
            self._log("WARNING", f"Landmarks parse failed: {e}")
        return landmarks

    def _parse_face_bottle(self, bottle: yarp.Bottle) -> Optional[Dict]:
        """Parse single face bottle to dict."""
        data = {}
        try:
            i = 0
            while i < bottle.size() - 1:
                key = bottle.get(i).asString()
                val = bottle.get(i + 1)
                
                if key in ["face_id", "zone", "distance", "attention"]:
                    data[key] = val.asString()
                elif key in ["track_id", "is_talking"]:
                    data[key] = val.asInt32()
                elif key in ["time_in_view", "pitch", "yaw", "roll", "cos_angle"]:
                    data[key] = val.asFloat64()
                elif key in ["bbox", "gaze_direction"] and val.isList():
                    lst = val.asList()
                    data[key] = [lst.get(j).asFloat64() for j in range(lst.size())]
                i += 2
            return data if data else None
        except Exception as e:
            self._log("WARNING", f"Face bottle parse failed: {e}")
            return None

    def get_target_landmarks(self, track_id: int, face_id: str) -> Optional[Dict]:
        """Get landmarks for specific person (prefer track_id)."""
        faces = self.parse_landmarks_latest()
        
        for face in faces:
            if face.get("track_id") == track_id:
                return face
        
        if face_id and face_id.lower() != "unknown":
            for face in faces:
                if face.get("face_id") == face_id:
                    return face
        
        return None

    def wait_tts_end(self, timeout: float) -> bool:
        """Wait for TTS end bookmark (0→1), with fallback."""
        start = time.time()
        got_start = False
        
        while time.time() - start < timeout:
            bottle = self.bookmark_port.read(False)
            if bottle and bottle.size() > 0:
                try:
                    v = bottle.get(0)
                    bookmark = v.asInt32() if v.isInt32() else int(v.asString())
                    if bookmark == 0:
                        got_start = True
                    elif bookmark == 1 and got_start:
                        return True
                except (ValueError, AttributeError):
                    pass
            time.sleep(0.02)
        
        # Fallback safety waits
        if not got_start:
            self._log("WARNING", "No TTS start bookmark; using fallback wait")
            time.sleep(1.4)
        elif got_start:
            self._log("WARNING", "TTS started but no end bookmark; using safety wait")
            time.sleep(1.0)
        
        return False

    def wait_user_utterance(self, timeout: float) -> Optional[str]:
        """Wait for STT output: [["text", "speaker"]]."""
        start = time.time()
        while time.time() - start < timeout:
            bottle = self.stt_port.read(False)
            if bottle and bottle.size() > 0:
                text = self._extract_stt_text(bottle)
                if text and text.strip():
                    return text.strip()
            time.sleep(0.1)
        return None

    def _extract_stt_text(self, bottle: yarp.Bottle) -> Optional[str]:
        """Extract text from STT bottle format (handles 1-level and 2-level nesting)."""
        try:
            if bottle.size() == 0:
                return None

            first = bottle.get(0)

            # Case A: first is a list
            if first.isList():
                inner = first.asList()
                if not inner or inner.size() == 0:
                    return None

                # If inner[0] is itself a list: [[["text","speaker"]]]
                inner0 = inner.get(0)
                if inner0.isList():
                    inner2 = inner0.asList()
                    if inner2 and inner2.size() > 0:
                        return inner2.get(0).asString()
                    return None

                # Normal: [["text","speaker"]]
                return inner.get(0).asString()

            # Case B: direct string
            if first.isString():
                return first.asString()

        except Exception as e:
            self._log("WARNING", f"STT parse failed: {e}")

        return None

    def _clear_stt_buffer(self):
        """Clear pending STT messages."""
        while self.stt_port.read(False):
            pass

    def _clear_bookmark_buffer(self):
        """Clear pending bookmark messages."""
        while self.bookmark_port.read(False):
            pass

    def ensure_stt_ready(self, language: str = "italian") -> bool:
        """Configure and start STT module."""
        try:
            self._yarp_rpc("/speech2text/rpc", f"set {language}")
            self._yarp_rpc("/speech2text/rpc", "start")
            return True
        except Exception as e:
            self._log("ERROR", f"STT setup failed: {e}")
            return False

    # ==================== Speech Output ====================

    def _speak(self, text: str) -> bool:
        """Speak text with bookmark markers (tries mkr, then mrk)."""
        self._clear_bookmark_buffer()
        return self._speak_with_tag(text, "mkr") or self._speak_with_tag(text, "mrk")

    def _speak_with_tag(self, text: str, tag: str) -> bool:
        """Speak with specified bookmark tag via YARP port (no shell quoting issues)."""
        try:
            if not self.speech_port:
                self._log("ERROR", "speech_port not initialized")
                return False
            
            msg = f"\\{tag}=0\\ {text} \\{tag}=1\\"
            b = yarp.Bottle()
            b.clear()
            b.addString(msg)
            self.speech_port.write(b)
            return True
        except Exception as e:
            self._log("ERROR", f"Speak failed ({tag}): {e}")
            return False

    # ==================== YARP RPC Commands ====================

    def _yarp_rpc(self, port: str, command: str, timeout: int = 10) -> str:
        """Execute YARP RPC command via subprocess (raises on failure)."""
        try:
            proc = subprocess.Popen(
                ["yarp", "rpc", port],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = proc.communicate(input=command + "\n", timeout=timeout)
            
            if proc.returncode != 0:
                raise RuntimeError(f"RPC failed: {port} :: {command} :: {stderr.strip()}")
            
            return stdout.strip()
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(f"RPC timeout: {port} :: {command}")
        except Exception as e:
            raise RuntimeError(f"RPC error: {port} :: {command} :: {str(e)}")

    def _register_face(self, name: str, track_id: int) -> bool:
        """Register face with objectRecognition."""
        try:
            self._yarp_rpc("/objectRecognition", f"name {name} id {track_id}")
            return True
        except Exception as e:
            self._log("ERROR", f"Face registration failed: {e}")
            return False

    def _execute_behaviour(self, behaviour: str) -> bool:
        """Execute behaviour via interactionInterface."""
        try:
            self._yarp_rpc("/interactionInterface", f"exe {behaviour}")
            return True
        except Exception as e:
            self._log("ERROR", f"Behaviour execution failed: {e}")
            return False

    def _set_face_tracker(self, name: str) -> bool:
        """Set face tracker target."""
        try:
            self._yarp_rpc("/faceTracker", f"set face {name}")
            return True
        except Exception as e:
            self._log("ERROR", f"Face tracker failed: {e}")
            return False

    # ==================== Face File Management ====================

    def _generate_unique_code(self) -> str:
        """Generate unique 5-digit code not in faces directory."""
        existing = set()
        try:
            if os.path.exists(self.FACES_DIR):
                for f in os.listdir(self.FACES_DIR):
                    name = os.path.splitext(f)[0]
                    if name.isdigit() and len(name) == 5:
                        existing.add(name)
        except Exception:
            pass
        
        for _ in range(1000):
            code = ''.join(random.choices(string.digits, k=5))
            if code not in existing:
                return code
        return str(int(time.time() * 1000) % 100000).zfill(5)

    def _verify_face_registration(self, code: str) -> bool:
        """Verify face file exists within timeout."""
        start = time.time()
        while time.time() - start < self.FILE_VERIFY_TIMEOUT:
            try:
                if os.path.exists(self.FACES_DIR):
                    if any(code in f for f in os.listdir(self.FACES_DIR)):
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def _find_face_code_for_track(self, track_id: int) -> Optional[str]:
        """Find face code for track_id from last_greeted.json."""
        try:
            entries = self._load_json("last_greeted.json", [])
            for entry in reversed(entries):
                if entry.get("track_id") == track_id:
                    return entry.get("assigned_code_or_name")
        except Exception:
            pass
        return None

    def _rename_face_file(self, old_name: str, new_name: str) -> bool:
        """Rename face file from code to actual name."""
        try:
            files = glob.glob(os.path.join(self.FACES_DIR, f"{old_name}*"))
            if not files:
                return False
            
            old_path = files[0]
            ext = os.path.splitext(old_path)[1]
            new_path = os.path.join(self.FACES_DIR, f"{new_name}{ext}")
            
            # Handle collision
            counter = 1
            while os.path.exists(new_path):
                new_path = os.path.join(self.FACES_DIR, f"{new_name}_{counter}{ext}")
                counter += 1
            
            os.rename(old_path, new_path)
            self._log("INFO", f"Renamed: {old_path} → {new_path}")
            return True
        except Exception as e:
            self._log("ERROR", f"Rename failed: {e}")
            return False

    # ==================== Database Persistence ====================

    def _init_db(self):
        """Initialize SQLite database for interaction logging."""
        try:
            conn = sqlite3.connect(self.DB_FILE)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    track_id INTEGER,
                    initial_face_id TEXT,
                    initial_state TEXT,
                    final_state TEXT,
                    success BOOLEAN,
                    extracted_name TEXT,
                    transitions TEXT,
                    full_result_json TEXT
                )
            ''')
            conn.commit()
            conn.close()
            self._log("INFO", f"Database initialized: {self.DB_FILE}")
        except Exception as e:
            self._log("ERROR", f"Database initialization failed: {e}")

    def _save_to_db(self, track_id: int, face_id: str, initial_state: str, result: dict):
        """Save interaction result to database with error handling."""
        try:
            conn = sqlite3.connect(self.DB_FILE, timeout=10.0)
            cursor = conn.cursor()
            
            # Extract high-level metrics
            success = result.get("success", False)
            final_state = result.get("final_state", "")
            extracted_name = None
            
            # Check for extracted name in SS2 result
            if "ss2_result" in result:
                extracted_name = result["ss2_result"].get("extracted_name")
            
            # Format transitions as comma-separated string
            transitions = ", ".join(result.get("transitions", []))
            
            # Serialize full result to JSON with error handling
            try:
                full_result_json = json.dumps(result, ensure_ascii=False)
            except (TypeError, ValueError) as json_error:
                self._log("WARNING", f"Could not serialize result to JSON: {json_error}")
                full_result_json = json.dumps({"error": "serialization_failed", "success": success})
            
            cursor.execute('''
                INSERT INTO interactions 
                (track_id, initial_face_id, initial_state, final_state, success, extracted_name, transitions, full_result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (track_id, face_id, initial_state, final_state, success, extracted_name, transitions, full_result_json))
            
            conn.commit()
            conn.close()
            self._log("INFO", f"Interaction saved to database: {initial_state} → {final_state}")
        except sqlite3.Error as db_error:
            self._log("ERROR", f"Database save failed (SQLite error): {db_error}")
            # Try to save to fallback JSON file
            self._save_to_fallback_log(track_id, face_id, initial_state, result)
        except Exception as e:
            self._log("ERROR", f"Database save failed: {e}")
            self._save_to_fallback_log(track_id, face_id, initial_state, result)
    
    def _save_to_fallback_log(self, track_id: int, face_id: str, initial_state: str, result: dict):
        """Fallback: save interaction to JSON file if database fails."""
        try:
            fallback_file = "interaction_fallback.json"
            entries = self._load_json(fallback_file, [])
            entries.append({
                "timestamp": datetime.now().isoformat(),
                "track_id": track_id,
                "face_id": face_id,
                "initial_state": initial_state,
                "result": result
            })
            self._save_json(fallback_file, entries)
            self._log("INFO", "Interaction saved to fallback log")
        except Exception as e:
            self._log("ERROR", f"Fallback save also failed: {e}")

    # ==================== JSON Persistence ====================

    def _ensure_json_file(self, filename: str, default: Any):
        """Create JSON file with default if missing."""
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                json.dump(default, f)

    def _load_json(self, filename: str, default: Any) -> Any:
        """Load JSON file or return default."""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, filename: str, data: Any):
        """Save data to JSON file."""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def _write_last_greeted(self, track_id: int, face_id: str, code: str):
        """Append entry to last_greeted.json."""
        try:
            entries = self._load_json("last_greeted.json", [])
            entries.append({
                "timestamp": datetime.now().isoformat(),
                "track_id": track_id,
                "face_id": face_id,
                "assigned_code_or_name": code
            })
            self._save_json("last_greeted.json", entries)
        except Exception as e:
            self._log("ERROR", f"Write last_greeted failed: {e}")

    def _update_last_greeted_name(self, track_id: int, new_name: str):
        """Update name in last_greeted.json."""
        try:
            entries = self._load_json("last_greeted.json", [])
            for entry in reversed(entries):
                if entry.get("track_id") == track_id:
                    entry["assigned_code_or_name"] = new_name
                    entry["name_updated"] = datetime.now().isoformat()
                    break
            self._save_json("last_greeted.json", entries)
        except Exception as e:
            self._log("ERROR", f"Update last_greeted failed: {e}")

    # ==================== LLM Integration ====================

    def _check_ollama_binary_exists(self) -> bool:
        """Check if Ollama server binary is installed."""
        paths = ["/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/ollama/bin/ollama"]
        for path in paths:
            if os.path.exists(path):
                self._log("INFO", f"Found Ollama binary at {path}")
                return True
        return False

    def _install_ollama_server(self) -> bool:
        """Install Ollama server if not present."""
        try:
            self._log("INFO", "Ollama server not found. Installing...")
            result = subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                self._log("INFO", "Ollama server installed successfully")
                return True
            else:
                self._log("ERROR", f"Ollama installation failed: {result.stderr}")
                return False
        except Exception as e:
            self._log("ERROR", f"Failed to install Ollama: {e}")
            return False

    def _start_ollama_server(self) -> bool:
        """Start Ollama server in background if not running."""
        try:
            # Check if already running
            try:
                req = urllib.request.Request(f"{self.OLLAMA_URL}/api/tags")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        self._log("INFO", "Ollama server already running")
                        return True
            except:
                pass  # Server not running, will start it
            
            self._log("INFO", "Starting Ollama server...")
            # Start in background, redirect output to log file
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=open("/tmp/ollama_server.log", "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
            
            # Wait for server to be ready (up to 30 seconds)
            for i in range(30):
                time.sleep(1)
                try:
                    req = urllib.request.Request(f"{self.OLLAMA_URL}/api/tags")
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        if resp.status == 200:
                            self._log("INFO", f"Ollama server started successfully (took {i+1}s)")
                            return True
                except:
                    continue
            
            self._log("ERROR", "Ollama server failed to start within 30 seconds")
            return False
        except Exception as e:
            self._log("ERROR", f"Failed to start Ollama server: {e}")
            return False

    def ensure_ollama_and_model(self) -> bool:
        """Ensure Ollama server is installed, running, and model is available."""
        try:
            # Step 1: Check if Ollama binary exists, install if not
            if not self._check_ollama_binary_exists():
                self._log("WARNING", "Ollama binary not found, attempting installation...")
                if not self._install_ollama_server():
                    self._log("ERROR", "Could not install Ollama server. Please install manually: curl -fsSL https://ollama.com/install.sh | sh")
                    return False
            
            # Step 2: Ensure Ollama server is running
            if not self._start_ollama_server():
                self._log("ERROR", "Could not start Ollama server")
                return False
            
            # Step 3: Check if model is available, pull if not
            req = urllib.request.Request(f"{self.OLLAMA_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                
                if not any(self.LLM_MODEL in m for m in models):
                    self._log("INFO", f"Model {self.LLM_MODEL} not found. Pulling... (this may take a few minutes)")
                    result = subprocess.run(
                        ["ollama", "pull", self.LLM_MODEL],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    if result.returncode == 0:
                        self._log("INFO", f"Model {self.LLM_MODEL} pulled successfully")
                    else:
                        self._log("ERROR", f"Failed to pull model: {result.stderr}")
                        return False
                else:
                    self._log("INFO", f"Model {self.LLM_MODEL} already available")
            
            self.ollama_last_check = time.time()
            self._log("INFO", "Ollama setup complete and ready")
            return True
        except Exception as e:
            self._log("ERROR", f"Ollama setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_ollama_health(self) -> bool:
        """Quick health check for Ollama service."""
        try:
            req = urllib.request.Request(f"{self.OLLAMA_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _llm_request(self, prompt: str, json_format: bool = False) -> str:
        """Send prompt to LLM with retry logic, return response text."""
        last_error = None
        
        for attempt in range(self.llm_retry_attempts):
            try:
                # Check Ollama health periodically
                current_time = time.time()
                if current_time - self.ollama_last_check > self.ollama_check_interval:
                    if not self._check_ollama_health():
                        self._log("WARNING", "Ollama health check failed, attempting anyway...")
                    self.ollama_last_check = current_time
                
                payload = {"model": self.LLM_MODEL, "prompt": prompt, "stream": False}
                if json_format:
                    payload["format"] = "json"
                
                data = json.dumps(payload).encode()
                req = urllib.request.Request(
                    f"{self.OLLAMA_URL}/api/generate",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                
                with urllib.request.urlopen(req, timeout=self.LLM_TIMEOUT) as resp:
                    result = json.loads(resp.read().decode()).get("response", "").strip()
                    if result:  # Success
                        if attempt > 0:
                            self._log("INFO", f"LLM request succeeded on attempt {attempt + 1}")
                        return result
                    else:
                        self._log("WARNING", f"LLM returned empty response (attempt {attempt + 1})")
                        last_error = "Empty response from LLM"
                        
            except Exception as e:
                last_error = str(e)
                self._log("WARNING", f"LLM request failed (attempt {attempt + 1}/{self.llm_retry_attempts}): {e}")
                if attempt < self.llm_retry_attempts - 1:
                    time.sleep(self.llm_retry_delay)
        
        self._log("ERROR", f"LLM request failed after {self.llm_retry_attempts} attempts: {last_error}")
        return ""

    def _llm_json(self, prompt: str) -> Dict:
        """Get JSON response from LLM with robust parsing."""
        text = self._llm_request(prompt, json_format=True)
        if not text:
            self._log("ERROR", "LLM returned empty response - returning safe default")
            return {}
        
        # Extract JSON object
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                result = json.loads(text[start:end+1])
                # Validate result is actually a dict
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as e:
                self._log("WARNING", f"JSON extraction failed: {e}")
        
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
            else:
                self._log("WARNING", f"LLM returned non-dict JSON: {type(result)}")
                return {}
        except json.JSONDecodeError:
            self._log("ERROR", f"LLM non-JSON response: {text[:100]}...")
            return {}

    def _llm_detect_greeting_response(self, utterance: str) -> Dict:
        """Detect if utterance is a greeting response."""
        prompt = f'''Does this text contain a greeting word like: hello, hi, hey, yes, yeah, sure, okay, good morning?
Text: "{utterance}"
Answer only: {{"responded": true, "confidence": 1.0}} or {{"responded": false, "confidence": 0.0}}'''
        
        result = self._llm_json(prompt)
        return {
            "responded": result.get("responded", False) is True,
            "confidence": float(result.get("confidence", 0) or 0)
        }

    def _llm_detect_any_response(self, utterance: str) -> Dict:
        """Detect if utterance contains meaningful response."""
        prompt = f'''Instruction: Analyze if this English text contains any meaningful response.
Text: "{utterance}"
Output ONLY a raw JSON object with this exact schema: {{"responded": true/false, "confidence": 0.0-1.0}}
JSON:'''
        
        result = self._llm_json(prompt)
        return {
            "responded": result.get("responded", False) is True,
            "confidence": float(result.get("confidence", 0) or 0)
        }

    def _llm_extract_name(self, utterance: str) -> Dict:
        """Extract name from utterance (won't invent names)."""
        prompt = f'''Instruction: Extract the person's name from this English text. ONLY extract if clearly stated. Do NOT invent names. If no name is found, use null.
Text: "{utterance}"
Output ONLY a raw JSON object with this exact schema: {{"answered": true/false, "name": "extracted_name" or null, "confidence": 0.0-1.0}}
JSON:'''
        
        result = self._llm_json(prompt)
        return {
            "answered": result.get("answered", False) is True,
            "name": result.get("name") or None,
            "confidence": float(result.get("confidence", 0) or 0)
        }

    def _llm_generate_ask_name(self) -> str:
        """Generate English question asking for name."""
        text = self._llm_request(
            'Generate a short English question to ask someone their name. '
            'Examples: "What is your name?", "May I have your name?" Output ONLY the question.'
        )
        return text.strip('"\'').strip() if text and len(text) < 100 else "What is your name?"

    def _llm_generate_convo_starter(self) -> str:
        """Generate English conversation starter."""
        text = self._llm_request(
            'Generate a short, friendly English conversation starter. '
            'Examples: "How are you doing today?", "How is your day going?" Output ONLY the sentence, under 15 words.'
        )
        return text.strip('"\'').strip() if text and len(text) < 150 else "How are you doing today?"

    def _llm_generate_followup(self, last_utterance: str, history: List[str]) -> str:
        """Generate English followup response."""
        history_text = "\n".join(f"- {u}" for u in history[-3:])
        text = self._llm_request(
            f'You are having a friendly English conversation.\n'
            f'User said: "{last_utterance}"\nRecent: {history_text}\n'
            f'Generate a short English response (1 sentence, under 20 words). Output ONLY the response.'
        )
        return text.strip('"\'').strip() if text and len(text) < 200 else "I see, interesting!"

    # ==================== Helpers ====================

    def _init_result(self, **kwargs) -> Dict[str, Any]:
        """Initialize result dict with common fields."""
        return {"success": False, "context": None, "context_label": None, 
                "next_state": None, **kwargs}

    def _read_and_store_context(self, result: Dict) -> Dict:
        """Read context and store in result."""
        context = self.read_context_latest()
        result["context"] = context
        result["context_label"] = context.get("label_str")
        return context

    def _detect_greeting_response(self, result: Dict) -> bool:
        """Detect greeting response and update result."""
        self._clear_stt_buffer()
        utterance = self.wait_user_utterance(self.STT_TIMEOUT)
        result["user_utterance"] = utterance
        
        if utterance:
            detection = self._llm_detect_greeting_response(utterance)
            result["response_detected"] = detection.get("responded", False)
            result["response_confidence"] = detection.get("confidence", 0.0)
            return result["response_detected"]
        return False

    def _handle_error(self, result: Dict, state: str, error: Exception) -> Dict:
        """Handle exception in state handler."""
        self._log("ERROR", f"{state} failed: {error}")
        result["error"] = str(error)
        return result

    def _build_ordered_trace(self, raw_result: Dict, track_id: int, initial_face_id: str, initial_state: str) -> Dict:
        """Transform raw execution result into structured JSON trace."""
        trace = {
            "success": raw_result.get("success", False),
            "track_id": track_id,
            "initial_face_id": initial_face_id,
            "initial_state": initial_state,
            "final_state": raw_result.get("final_state", initial_state),
            "context_label": raw_result.get("context_label", "unknown"),
            "steps": []
        }
        
        # Process SS1 if present
        if "ss1_result" in raw_result:
            ss1 = raw_result["ss1_result"]
            details = {
                "face_registered_as": ss1.get("assigned_code"),
                "greet_attempt": ss1.get("greet_attempt"),
                "response_detected": ss1.get("response_detected", False)
            }
            if "error" in ss1:
                details["error"] = ss1["error"]
            
            trace["steps"].append({
                "step": "ss1",
                "status": "success" if ss1.get("success") else "failed",
                "action": "greeting_new_person",
                "details": details
            })
        
        # Process SS2 if present
        if "ss2_result" in raw_result:
            ss2 = raw_result["ss2_result"]
            details = {
                "attempts_made": ss2.get("attempts", 0),
                "user_said": ss2.get("user_utterance"),
                "extracted_name": ss2.get("extracted_name"),
                "confidence": ss2.get("name_confidence", 0.0)
            }
            if "error" in ss2:
                details["error"] = ss2["error"]
            
            trace["steps"].append({
                "step": "ss2",
                "status": "success" if ss2.get("success") else "failed",
                "action": "asking_name",
                "details": details
            })
        
        # Process SS3 if present
        if "ss3_result" in raw_result:
            ss3 = raw_result["ss3_result"]
            details = {
                "user_said": ss3.get("user_utterance"),
                "response_detected": ss3.get("response_detected", False)
            }
            if "error" in ss3:
                details["error"] = ss3["error"]
            
            trace["steps"].append({
                "step": "ss3",
                "status": "success" if ss3.get("success") else "failed",
                "action": "greeting_known_person",
                "details": details
            })
        
        # Process SS4 if present
        if "ss4_result" in raw_result:
            ss4 = raw_result["ss4_result"]
            
            # Build dialogue transcript
            transcript = []
            robot_utterances = ss4.get("robot_utterances", [])
            user_responses = ss4.get("user_responses", [])
            
            for i in range(max(len(robot_utterances), len(user_responses))):
                if i < len(robot_utterances):
                    transcript.append(f"Robot: {robot_utterances[i]}")
                if i < len(user_responses):
                    transcript.append(f"User: {user_responses[i]}")
            
            details = {
                "turns_count": ss4.get("turns", 0),
                "dialogue_transcript": transcript
            }
            if "error" in ss4:
                details["error"] = ss4["error"]
            
            trace["steps"].append({
                "step": "ss4",
                "status": "finished" if ss4.get("success") else "incomplete",
                "action": "conversation",
                "details": details
            })
        
        return trace

    def _log(self, level: str, message: str):
        """Log with timestamp."""
        ts = datetime.now().isoformat()
        print(f"[{ts}] [{level}] {message}")
        self.log_buffer.append({"timestamp": ts, "level": level, "message": message})

    def _debug_peek_ports_once(self):
        """Debug helper: peek at raw STT and bookmark port data."""
        b = self.stt_port.read(False)
        if b:
            self._log("DEBUG", f"STT raw: {b.toString()}")
        
        m = self.bookmark_port.read(False)
        if m:
            self._log("DEBUG", f"BM raw: {m.toString()}")


# ==================== Main ====================

if __name__ == "__main__":
    import sys
    
    yarp.Network.init()
    
    if not yarp.Network.checkNetwork():
        print("ERROR: YARP network not available")
        sys.exit(1)
    
    module = InteractionManagerModule()
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.configure(sys.argv)
    
    print("Starting InteractionManagerModule...")
    print("Connect ports:")
    print("  yarp connect /alwayson/stm/context:o /interactionManager/context:i")
    print("  yarp connect /alwayson/vision/landmarks:o /interactionManager/landmarks:i")
    print("  yarp connect /speech2text/text:o /interactionManager/stt:i")
    print("  yarp connect /acapelaSpeak/bookmark:o /interactionManager/acapela_bookmark:i")
    print("  yarp connect /interactionManager/speech:o /acapelaSpeak/speech:i")
    print("Usage: echo \"run <track_id> <face_id> <ss1|ss2|ss3|ss4>\" | yarp rpc /interactionManager")
    
    try:
        if module.configure(rf):
            module.ensure_ollama_and_model()
            while module.updateModule():
                time.sleep(module.getPeriod())
        else:
            print("ERROR: Configuration failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        module.interruptModule()
        module.close()
        yarp.Network.fini()
