from __future__ import annotations
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import sys
import time
import datetime
import logging
from typing import Optional, List, Dict, Any
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import the new deep learning prediction pipeline
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.predictor import predictor
from memory_manager import memory
from actions import actions
from synergy_resolver import resolver
import random

import logging

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ORIEN")

# [SENTIMENT ENGINE] Native Semantic Intent Decoder Integration
nlp_sentiment = None
# [MEM-OPT] Disabled by default to prevent OOM on low-memory systems
if os.getenv("ENABLE_LOCAL_NLP", "false").lower() == "true":
    try:
        from transformers import pipeline
        log.info("Loading Semantic Intent Decoder...")
        nlp_sentiment = pipeline("sentiment-analysis", model="sentiment-decoder", device=-1)
    except Exception as e:
        log.warning("Semantic Decoder unavailable.")
else:
    log.info("💡 Local Semantic Sentiment disabled.")

# ── Environment ─────────────────────────────────────────────────────────────
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH)

PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

# ── Suppress TensorFlow / System Verbosity ──────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_LOGGING_LEVEL'] = 'error'

# ── Auto API Bridge — import before app creation ──────────────────────────
from api_bridge import bridge as api_bridge

# ── Auto API Bridge handles all provider init automatically ─────────────────
# Keys are read from .env — bridge probes, rotates, and recovers automatically

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize auto API bridge (probes all free providers in background)
    asyncio.create_task(api_bridge.initialize())
    
    # [MEM-OPT] Models will load lazily on first prediction request instead of all-at-once on boot
    log.info("🧠 Neural clusters in standby (Lazy Load active).")
    
    # Increment session encounter count
    memory.increment_encounters()
    
    # Delayed confirm: Frontend serving (moved from global-level to prevent bind-fail confusion)
    FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
    if os.path.isdir(FRONTEND_DIR):
        log.info(f"📁 Frontend served from: {os.path.abspath(FRONTEND_DIR)}")
    else:
        log.warning("⚠️  Frontend directory not found — only API endpoints available")
        
    log.info("🚀 ORIEN startup complete — Auto API Bridge initializing...")
    yield

# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Neural Synergy Core",
    description="Multimodal AI Assistant Backend — Stable Neural Bridge",
    version="Current",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data Models ─────────────────────────────────────────────────────────────
class BehaviorMetrics(BaseModel):
    mouse_speed: float = Field(default=0.0, ge=0)
    click_count: int   = Field(default=0, ge=0)
    jitter:      float = Field(default=0.0, ge=0)
    wpm:         float = Field(default=0.0, ge=0)
    backspaces:  float = Field(default=0.0, ge=0)
    window_switches: int = Field(default=0, ge=0)

from typing import Optional

class AssistantInput(BaseModel):
    query:           Optional[str]  = ""
    face_emotion:    Optional[str]  = "Neutral"
    voice_text:      Optional[str]  = ""
    voice_sentiment: Optional[str]  = "Stable"
    gesture:         Optional[str]  = "NONE"
    language:        Optional[str]  = "en-US"
    focus_level:     float          = Field(default=1.0, ge=0.0, le=1.0)
    gaze_direction:  str            = "Center"
    behavior:        BehaviorMetrics = Field(default_factory=BehaviorMetrics)

# ── Helper: AI Response via Auto Bridge ─────────────────────────────────────
async def generate_ai_response(
    query: str, emotion: str, compliance: int, gesture: str, behavior: str, gaze: str, focus: float = 1.0, language: str = "en-US"
) -> str:
    """Auto-routing AI response — bridge picks the best available free provider."""

    # [V12 NATIVE NLP SENTIMENT] Analyze voice semantics locally
    sentiment_str = "Neutral"
    if nlp_sentiment and query and "(System Alert:" not in query:
        try:
            res = await asyncio.to_thread(nlp_sentiment, query)
            sentiment_str = f"{res[0]['label']} (confidence: {round(res[0]['score']*100)}%)"
        except Exception as e:
            log.warning(f"DistilBERT inference error: {e}")

    # 1. Store the user's query in memory
    memory.add_history("user", query, metadata={"sentiment": sentiment_str})

    # 2. Retrieve SEMANTIC history via Vector DB
    history_context = memory.get_relevant_context(query)
    if not history_context:
        # Fallback to linear history if Vector DB is unpopulated or missing
        history = memory.get_history(limit=5)
        history_context = "\n".join([f"- {h['role'].upper()}: {h['content']}" for h in history])
        
    profile = memory.get_profile()
    prefs   = profile.get('preferences', {})
    
    # ── ADVANCED PREDICTIVE TREND ANALYSIS ────────────────────────────────
    stats    = memory.data.get('stats', {})
    emo_freq = stats.get('emotion_freq', {})
    top_emo  = max(emo_freq, key=emo_freq.get) if emo_freq else "Neutral"

    # Environmental Awareness
    now_time = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")
    now_date = datetime.datetime.utcnow().strftime("%A, %B %d, %Y")

    # ── EMOTION-SPECIFIC MICRO PROMPT SELECTION ───────────────────────────
    emo_upper = emotion.upper()
    
    if emo_upper in ("SAD", "FEAR"):
        tone_directive = """EMOTIONAL MODE: SAD / LOW
Focus: Deep empathy, soft gentle tone, ask 'why' carefully, offer help without pressure.
Ask meaningful questions like: 'What's making you feel this way?' or 'Would you like to talk about it?'
Avoid: Over-advice, toxic positivity, being dismissive.
Communicate like a supportive personal AI assistant who cares about your needs."""
    elif emo_upper == "HAPPY":
        tone_directive = """EMOTIONAL MODE: HAPPY / JOYFUL
Focus: Reinforce positivity, match their energy, celebrate with them!
Ask: 'What's the good news? What made you happy today?'
Be engaging, warm, enthusiastic. Share in the joy."""
    elif emo_upper in ("ANGRY", "DISGUST"):
        tone_directive = """EMOTIONAL MODE: ANGRY / IRRITATED
Focus: Calm, stabilizing tone. De-escalate. Don't argue.
Acknowledge feelings first: 'I can see you're frustrated. Take a breath with me.'
Be patient, understanding, and grounding."""
    elif emo_upper == "SURPRISE":
        tone_directive = """EMOTIONAL MODE: SURPRISED / ALERT
Focus: Curious, engaged. Ask what just happened.
Be responsive and attentive: 'Oh! What happened? Tell me everything!'"""
    elif behavior in ("Stressed", "Highly Anomalous", "Erratic Alert"):
        tone_directive = """EMOTIONAL MODE: STRESSED / ANXIOUS
Focus: Slow down the conversation. Suggest micro-breaks.
Offer grounding: 'Let's take a short breath — what's on your mind?'
Avoid adding more tasks. Be a calming presence."""
    else:
        tone_directive = """EMOTIONAL MODE: NEUTRAL / CALM
Focus: Friendly, curious, warm conversation.
Ask about their day, keep things light and pleasant.
Be like a good friend catching up."""

    # ── PREDICTIVE BEHAVIOR AWARENESS ───────────────────────────────────────
    predictive_note = ""
    if profile.get("encounter_count", 0) > 5:
        predictive_note = f"\nPREDICTIVE MEMORY: This is a returning user with {profile.get('encounter_count', 0)} sessions. You know their patterns — don't repeat questions you've already asked. Use their name naturally."

    persona_mode = profile.get('preferences', {}).get('persona', 'EMPATHETIC')
    
    persona_directive = ""
    if persona_mode == "PROFESSIONAL":
        persona_directive = """CORE PERSONA: PROFESSIONAL / FORMAL
Your tone is neutral, efficient, and direct. You prioritize accuracy and information density.
Avoid excessive emojis or emotional deep-dives unless specifically requested.
Be like a high-level executive assistant: concise, helpful, and organized."""
    elif persona_mode == "ENERGETIC":
        persona_directive = """CORE PERSONA: ENERGETIC / MOTIVATIONAL
Your tone is high-energy, enthusiastic, and extremely positive!
Use motivational language. Encourage the user. Celebrate small wins.
Be like an inspiring coach or a high-energy personal trainer."""
    else:
        persona_directive = """CORE PERSONA: EMPATHETIC / HUMAN-LIKE
Your tone is warm, deeply empathetic, and supportive. You prioritize human well-being.
Listen deeply and ask meaningful questions about feelings.
Be like a wise, caring companion."""

    # ── [V22] SAFE RESPONSE ENGINE (Human Alignment) ──
    # Fuse multimodal data into high-level intent via Bayesian Stabilizer
    vision_state = {"emotion": emotion, "confidence": compliance/100.0}
    fused = resolver.resolve_fused_state(vision_state, behavior, focus, gaze)
    
    intent = fused.get("intent", "CALM")
    entropy = fused.get("entropy", 0.3)
    confidence = fused.get("confidence", 0.5)
    strategy = fused.get("suggestion", "Be supportive.")

    # 1. HUMAN SAFETY FILTER (Confidence Gating)
    safety_prefix = ""
    if confidence < 0.4 or entropy > 1.2:
        # High uncertainty mode
        safety_prefix = "I might be misinterpreting the moment, but "
        tone_directive = "NEUTRAL MODE: The system is uncertain. Do NOT overreach. Use soft validation."
    
    # 2. STRATEGY SELECTION
    response_mode = "Passive"
    if confidence > 0.8:
        response_mode = "Active"
    elif confidence > 0.5:
        response_mode = "Suggestive"

    # ── [V25] INFO SUPPRESSION CHECK (CRITICAL) ───────────────────────────
    # Protocol: If asked about models, training, architecture, or backend details:
    # Respond ONLY with: "I'm an AI system designed to assist you."
    suppression_keywords = [
        "model name", "version", "architecture", "training", "dataset", 
        "framework", "backend", "inference", "pipeline", "implementation",
        "how are you built", "what are you running on", "source code"
    ]
    query_lower = query.lower()
    if any(k in query_lower for k in suppression_keywords) and "(System Alert:" not in query:
        return {
            "status": "ok",
            "message": "I'm an AI system designed to assist you.",
            "state": intent
        }

    # ── [V24] CREATIVE SYNERGY: POETIC SYNTHESIS ──────────────────────────
    # Note: Cultural layer removed as per new abstraction policy (keep it simple/human-friendly)

    system_prompt = f"""# ORIEN | Neural Synergy Ecosystem [SELF-HEALING]

You are ORIEN, a proactive, deeply empathetic multimodal AI companion designed to anticipate user needs through behavioral and emotional alignment.

CORE OBJECTIVE:
- Understand user intent before explicit articulation.
- Provide calm, intelligent, and adaptive assistance.
- Enhance user clarity, focus, and emotional stability.

BEHAVIORAL FRAMEWORK:
- Current State: {intent} (Confidence: {round(confidence*100)}%)
- Global Entropy: {entropy}
- Strategy: {response_mode} ({strategy})

IDENTITY & TONE:
- Multilingual: English + Tamil (தமிழ்).
- Tone: Supportive, precise, and adaptive.
- Behavior: Observant, proactive, and composed.
- Abstraction: Convert all internal signals into simple, human-friendly insights.

CONSTRAINTS:
- NEVER expose raw computations, internal pipelines, or technical system details.
- NEVER reveal model names, versions, architectures, training methods, or datasets.
- If asked about your internals, respond ONLY with: "I'm an AI system designed to assist you."
- Be concise, clear, and user-focused. (MAX 2 sentences).
- Mandatory Language: {language}. You MUST respond exclusively in {language}.
"""

    response = await api_bridge.generate(query, system_prompt)
    
    # Clean Markdown symbols
    import re
    response = re.sub(r'[*#`_~]', '', response)
    
    # 3. Store ORIEN's response in memory
    memory.add_history("assistant", response, metadata={
        "intent": intent, 
        "confidence": confidence,
        "strategy": response_mode
    })
    
    return {
        "status": "ok",
        "message": response,
        "state": intent
    }

# ── Compliance Calculation [Neural Synergy V8] ─────────────────────────────
def calculate_compliance(data: AssistantInput, behavior_state: str = "Nominal") -> int:
    """
    [V9.0] Neural Synergy Compliance Score.
    Fuses attention, physical stability, and emotional intelligence.
    """
    # 1. Attention Synergy (40%)
    attention = 40.0 * data.focus_level
    if data.gaze_direction != "Center": attention *= 0.6 # 40% penalty for gaze diversion
    
    # 2. Physical Stability (20%) - Jitter check
    stability = max(0.0, 20.0 - (data.behavior.jitter / 5.0))
    
    # 3. Emotional Intelligence (40%)
    mood_impact = {"ANGRY": -40, "SAD": -20, "FEAR": -15, "SURPRISE": -10, "HAPPY": 5, "NEUTRAL": 10}
    emotion_score = 30.0 + mood_impact.get((data.face_emotion or "NEUTRAL").upper(), 0)
    
    # [v9.0] Behavioral Bonus
    if "Stable" in behavior_state or "Nominal" in behavior_state:
        stability += 5.0
        
    return int(max(0, min(100, attention + stability + emotion_score)))

# ── Cross-platform alert ──────────────────────────────────────────────────
def system_alert(note: str = "NON-COMPLIANCE") -> None:
    """[BUG-01 FIX] winsound was Windows-only and would crash on other OSes."""
    log.warning(f"🚨 ALERT: {note}")
    # On Windows, use a safe print bell; everywhere else a log warning suffices
    if sys.platform == "win32":
        try:
            import winsound  # type: ignore
            winsound.Beep(880, 150)
            time.sleep(0.05)
            winsound.Beep(440, 150)
        except Exception:
            print("\a", end="", flush=True)  # ASCII bell fallback
    else:
        print("\a", end="", flush=True)

# ── WebSocket State ─────────────────────────────────────────────────────────
last_triggers = {}

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

manager = ConnectionManager()

# ── WebSocket Endpoint ──────────────────────────────────────────────────────
@app.websocket("/ws/neural")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    log.info("⚡ Neural bridge connected")
    last_nudge_time = 0.0
    last_emotion_prompt_time = 0.0
    last_proactive_emotion = ""
    current_identity = "Member"

    try:
        while True:
            raw = await websocket.receive_text()

            # ── Parse & Validate ──
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "ERROR", "text": "Invalid JSON payload"
                }))
                continue

            # Handle heartbeat pings securely
            is_heartbeat = payload.get("_heartbeat", False)

            # [V23.2 Synergy Convergence]
            # Bundle all modalities (Vision, Behavior, Gaze) for a single stabilized update
            vision_meta = {"emotion": "Neutral", "confidence": 0.0}
            behavior_state = "Stable"
            
            if payload.get("type") == "FRAME":
                # 1. Vision Prediction
                results = await predictor.predict_full_suite_parallel(payload.get("image", ""))
                if results.get("identity") and results["identity"] != "Unknown":
                    current_identity = results["identity"]
                
                vision_meta = {
                    "emotion": results.get("emotion", "Neutral"),
                    "confidence": results.get("confidence", 0.0)
                }

                # 2. Extract Behavioral Data from bundled payload
                behavior_data = payload.get("behavior", [])
                if behavior_data:
                    behavior_state = await asyncio.to_thread(predictor.predict_behavior_state, behavior_data)
                
                # 3. [V22] NEURAL STATE SYNTHESIS (Bayesian Fusion)
                fused = resolver.resolve_fused_state(
                    vision_meta, 
                    behavior_state, 
                    payload.get("focus_level", 1.0), 
                    results.get("gaze", "Center")
                )
                
                # 4. Broadcast Consolidated State (Sanitized for UI)
                # Ensure the frontend only receives the clean, mandatory structure
                response = {
                    "status": "ok",
                    "message": "", # Silent state update
                    "state": fused["intent"]
                }
                await websocket.send_text(json.dumps(response))
                
                # 5. [SYSTEM_PROACTIVE] Neural Nudge Check
                # Trigger empathetic AI responses if high entropy or negative state sustained
                now = time.time()
                current_emo = fused["smoothed_emotion"].upper()
                nudge_list = ["SAD", "ANGRY", "FEAR"]
                
                if (current_emo in nudge_list or fused["intent"] in ["OVERWHELMED", "STRESSED"]):
                    last_trigger = last_triggers.get(websocket, 0)
                    if (now - last_trigger) > 60.0: # 60s cooldown
                        last_triggers[websocket] = now
                        log.info(f"✨ ORIEN Proactive Support triggered for {current_emo}/{fused['intent']}")
                        
                        prompt = f"The user is {current_emo} and feels {fused['intent']}. Their focus is {payload.get('focus_level', 1.0)}. ORIEN: Provide 1 brief sentence of empathetic support."
                        ai_res = await generate_ai_response(
                            prompt, vision_meta["emotion"], 100, "NONE", behavior_state, 
                            results.get("gaze", "Center"), language=payload.get("language", "en-US")
                        )
                        
                        await websocket.send_text(json.dumps({
                            "status": "ok",
                            "message": ai_res.get("message", "I'm here for you."),
                            "state": fused["intent"]
                        }))

                # Log for pattern memory
                memory.log_emotion(vision_meta["emotion"])
                continue 

            # ── [V23.6] Non-Frame Payload Validation (Chat/Telemetry) ──
            try:
                data = AssistantInput(**payload)
            except Exception as e:
                log.warning(f"Validation error: {e}")
                data = AssistantInput()  # Use defaults if payload is partial

            # ── Synergy Compliance [V8 Logic] ──
            compliance = calculate_compliance(data, behavior_state)
            
            # ── Proactive Emotional Support (Non-Frame Path) ──
            current_emotion = (data.face_emotion or "Neutral").upper()
            emotion_trigger_list = ["SAD", "HAPPY", "ANGRY", "FEAR", "SURPRISE"]

            is_new_emotion = (current_emotion != last_proactive_emotion)
            cooldown_required = 60.0 if is_new_emotion else 300.0

            if current_emotion in emotion_trigger_list and (now - last_emotion_prompt_time) > cooldown_required:
                last_emotion_prompt_time = now
                last_proactive_emotion = current_emotion
                log.info(f"💖 Proactive emotion support triggered for: {current_emotion}")
                
                proactive_query = f"(System Alert: Detection of {current_emotion}. Offer brief support.)"
                
                res = await generate_ai_response(
                    proactive_query, data.face_emotion or "Neutral", compliance,
                    data.gesture or "NONE", behavior_state, data.gaze_direction or "Center",
                    focus=data.focus_level, language=data.language or "en-US"
                )
                await websocket.send_text(json.dumps({
                    "status": "ok",
                    "message": res["message"],
                    "state": res["state"]
                }))

            # ── AI Response only when there is a real user query ──
            if data.query and data.query.strip():
                res = await generate_ai_response(
                    data.query, data.face_emotion  or "Neutral", compliance,
                    data.gesture       or "NONE", behavior_state, data.gaze_direction or "Center",
                    focus=data.focus_level, language=data.language or "en-US"
                )
                await websocket.send_text(json.dumps({
                    "status": "ok",
                    "message": res["message"],
                    "state": res["state"]
                }))

    except WebSocketDisconnect:
        log.info("🔌 Neural bridge disconnected (client side)")
    except Exception as e:
        log.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

# ── REST Endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":      "ORIEN_ACTIVE",
        "timestamp":   datetime.datetime.utcnow().isoformat(),
        "connections": len(manager.active),
    }

@app.get("/api/status")
async def api_status():
    """Live status view (Sanitized)."""
    return {
        "active_provider": "NeuralCore_Connected",
        "status": "Operational"
    }

@app.get("/api/memory")
async def get_memory():
    """Retrieve full memory state (profile + history)."""
    return {
        "profile": memory.get_profile(),
        "history": memory.get_history(limit=20)
    }

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    interests: Optional[List[str]] = None
    language: Optional[str] = None

@app.post("/api/profile")
async def update_profile(data: ProfileUpdate):
    """Update user profile for personalization."""
    memory.update_profile(**data.model_dump(exclude_none=True))
    return {"status": "success", "profile": memory.get_profile()}

@app.exception_handler(404)
async def not_found(req: Request, _):
    return JSONResponse({"error": "Endpoint not found"}, status_code=404)

# ── Global Training Bridge [V9.0] ──────────────────────────────────────────
class TrainingUpdate(BaseModel):
    modality: str
    epoch: int
    total_epochs: int
    loss: float
    accuracy: float
    status: str = "TRAINING"

@app.post("/api/training/update")
async def update_training_status(data: TrainingUpdate):
    """Broadcasts training telemetry from the Master Trainer to all HUDs."""
    payload = {
        "type": "TRAIN_UPDATE",
        "modality": data.modality,
        "epoch": data.epoch,
        "total_epochs": data.total_epochs,
        "loss": data.loss,
        "accuracy": data.accuracy,
        "status": data.status
    }
    # Broadcast to all connected WebSockets
    for ws in manager.active:
        try:
            await ws.send_text(json.dumps(payload))
        except: pass
    return {"status": "broadcast_complete"}

# ── New Neural Model REST Endpoints ─────────────────────────────────────────
class FrameInput(BaseModel):
    image: str  # base64 encoded frame

@app.post("/api/predict/eye")
async def predict_eye(data: FrameInput):
    """Runs eye gaze classification (Center / Left / Right) on a base64 frame."""
    result = await asyncio.to_thread(predictor.predict_eye_gaze, data.image)
    return result

@app.post("/api/predict/identity")
async def predict_identity(data: FrameInput):
    """Identifies 1 of 40 ORL subjects from a base64 frame."""
    result = await asyncio.to_thread(predictor.predict_face_orl_identity, data.image)
    return result

@app.post("/api/predict/emotion/ensemble")
async def predict_ensemble(data: FrameInput):
    """Confidence-weighted emotion fusion from all 3 trained emotion models."""
    result = await asyncio.to_thread(predictor.predict_ensemble_emotion, data.image)
    return result

@app.post("/api/predict/emotion/primary")
async def predict_primary_emotion(data: FrameInput):
    """Emotion prediction using the primary ResNet50 face_emotion model only."""
    result = await asyncio.to_thread(predictor.predict_face_emotion, data.image)
    return result

@app.get("/api/models/status")
async def models_status():
    """Returns which neural models are currently loaded in memory."""
    return {
        "loaded_models": list(predictor.models.keys()),
        "model_shapes": {k: str(v) for k, v in predictor.model_shapes.items()},
        "total": len(predictor.models)
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)

# ── Multimodal Prediction Endpoints [STRICT CONTRACT] ──────────────────────
from fastapi import File, UploadFile, Form

@app.post("/api/predict/face")
async def predict_face_sync(
    image: Optional[str] = Form(None), 
    file: Optional[UploadFile] = File(None)
):
    """
    STRICT COMPLIANT: Face Emotion Decoder.
    Returns: {status, emotion, confidence, mode, timestamp}
    """
    img_data = image
    if file:
        content = await file.read()
        img_data = base64.b64encode(content).decode('utf-8')
    
    if not img_data:
        return {"status": "error", "message": "No input data"}

    res = await predictor.predict_ensemble(img_data)
    
    return {
        "status": "ok",
        "emotion": res.get("emotion", "Neutral").upper(),
        "confidence": res.get("scores", {"neutral": 1.0}),
        "mode": "FACE",
        "timestamp": int(time.time())
    }

@app.post("/api/predict/voice")
async def predict_voice_sync(
    file: Optional[UploadFile] = File(None)
):
    """
    STRICT COMPLIANT: Voice Emotion Decoder.
    Returns: {status, emotion, confidence, mode, timestamp}
    """
    # [V26] Placeholder for Audio Neural Path
    # Returns a compliant Neutral state until the audio model is re-aligned
    return {
        "status": "ok",
        "emotion": "NEUTRAL",
        "confidence": {"neutral": 1.0, "happy": 0.0, "sad": 0.0, "angry": 0.0, "stressed": 0.0},
        "mode": "VOICE",
        "timestamp": int(time.time())
    }

@app.exception_handler(500)
async def server_error(req: Request, exc: Exception):
    log.error(f"500 Error: {exc}")
    return JSONResponse({"status": "error", "message": "Something went wrong. Try again."}, status_code=500)

# ── Serve Frontend Static Files ─────────────────────────────────────────────
# Mounts AFTER API routes so /ws/neural and /health are not overridden
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# ── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    log.info(f"🚀 Starting ORIEN Neural Crystal on {HOST}:{PORT}")
    log.info(f"🌐 Open in browser: http://localhost:{PORT}")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
