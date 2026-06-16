"""
FastAPI Backend for Multi-Modal Field Service Assistant.

Exposes:
- /analyze-image (vision only)
- /voice-query (speech + agent)
- /get-guidance (full multimodal agent turn)
- /session/{id} (state)
- /upload/document (live PDF/MD/TXT ingestion to RAG - robust, no llama_index.readers)
- WebSocket for real-time streaming (future enhancement)
"""

import os
import uuid
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import structlog
from pathlib import Path as _Path
import tempfile
import shutil

from ai_core.models.schemas import (
    AnalyzeImageRequest, AnalyzeImageResponse,
    GuidanceRequest, GuidanceResponse,
    SessionStateResponse,
    QueryIntent
)
from ai_core.agents.field_agent import run_agent_turn, get_field_agent
from ai_core.speech.stt import transcribe_audio
from ai_core.speech.tts import synthesize_speech

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Multi-Modal Field Service Assistant API",
    version="1.0.0",
    description="Agentic, vision + voice + RAG powered assistant for field technicians"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (replace with Redis + DB in prod)
sessions: dict[str, dict] = {}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time(), "mock_mode": os.getenv("MOCK_MODE", "false")}

@app.post("/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image_endpoint(req: AnalyzeImageRequest):
    """Standalone vision analysis (useful for quick checks or pre-filtering)."""
    from ai_core.agents.tools.vision_tool import analyze_image as vision_tool
    start = time.time()
    
    result = vision_tool.invoke({
        "image_b64": req.image_b64,
        "focus": req.focus,
        "equipment_hint": req.equipment_hint
    })
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    latency = int((time.time() - start) * 1000)
    return AnalyzeImageResponse(
        vision_result=result["data"],
        processing_time_ms=latency
    )

@app.post("/get-guidance", response_model=GuidanceResponse)
async def get_guidance(req: GuidanceRequest):
    """
    Main entry point for full agentic reasoning.
    Accepts text + optional image. Runs the complete LangGraph workflow
    (including early Intent Classification + branched planner for troubleshooting vs general).
    """
    session_id = req.session_id or str(uuid.uuid4())
    
    result = await run_agent_turn(
        session_id=session_id,
        user_input=req.transcript,
        image_b64=req.image_b64,
        technician_id="tech-demo"
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    resp = result["response"] or {}
    
    # Update lightweight session
    sessions[session_id] = {
        "last_updated": time.time(),
        "equipment": (resp.get("vision") or {}).get("equipment_model") if resp else None,
        "last_confidence": resp.get("confidence"),
        "last_plan_summary": resp.get("plan", {}).get("diagnosis") if resp.get("plan") else None
    }
    
    # Determine if this was a troubleshooting path (for frontend rendering)
    intent_value = resp.get("query_intent")
    is_troubleshooting = (intent_value == QueryIntent.TROUBLESHOOTING.value) if intent_value else True
    
    return GuidanceResponse(
        session_id=session_id,
        plan=resp.get("plan"),
        immediate_response=resp.get("immediate", ""),
        voice_text=resp.get("voice", ""),
        citations=resp.get("plan", {}).get("citations", []) if resp.get("plan") else [],
        confidence=resp.get("confidence", 0.5),
        needs_clarification=resp.get("needs_more_info", False),
        safety_warnings=resp.get("plan", {}).get("warnings", []) if resp.get("plan") else [],
        trace_id=resp.get("trace_id", ""),
        next_actions=["Say 'next step'", "Upload new photo", "Ask for clarification"],
        # NEW: expose intent so frontend can render explanation vs repair-plan differently
        query_intent=intent_value,
        is_troubleshooting=is_troubleshooting
    )

@app.post("/voice-query")
async def voice_query(req: dict):
    """
    Accepts raw audio (base64) or pre-transcribed text.
    Returns both text response and synthesized audio (base64 WAV).
    """
    session_id = req.get("session_id") or str(uuid.uuid4())
    transcript = req.get("transcript")
    audio_b64 = req.get("audio_b64")
    image_b64 = req.get("image_b64")
    
    if not transcript and audio_b64:
        transcript = transcribe_audio(audio_b64)
        if not transcript:
            raise HTTPException(400, "Failed to transcribe audio")
    
    if not transcript:
        raise HTTPException(400, "No transcript or audio provided")
    
    # Run full agent
    agent_result = await run_agent_turn(session_id, transcript, image_b64)
    
    if not agent_result["success"]:
        raise HTTPException(500, agent_result.get("error"))
    
    voice_text = agent_result["response"].get("voice", "I have processed your request.")
    
    # Generate TTS
    audio_out_b64 = synthesize_speech(voice_text)
    
    return {
        "session_id": session_id,
        "transcript": transcript,
        "response_text": agent_result["response"].get("immediate"),
        "voice_text": voice_text,
        "audio_b64": audio_out_b64,
        "plan": agent_result["response"].get("plan"),
        "confidence": agent_result["response"].get("confidence"),
        "query_intent": agent_result["response"].get("query_intent")
    }

@app.get("/session/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    s = sessions[session_id]
    return SessionStateResponse(
        session_id=session_id,
        current_equipment={"model": s.get("equipment")} if s.get("equipment") else None,
        conversation_length=1,  # simplified
        last_plan_summary=s.get("last_plan_summary"),
        confidence=s.get("last_confidence", 0.0)
    )

# =============================================================================
# LIVE DOCUMENT UPLOAD (robust PDF/MD/TXT - fixes "No module named 'llama_index.readers'")
# =============================================================================
@app.post("/upload/document")
async def upload_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """
    Upload a new manual / SOP / procedure (PDF, .md, .txt).
    Immediately added to the live ChromaDB index using robust pypdf-first extraction.
    No full re-ingest required. No dependency on broken llama_index.readers.
    """
    allowed = {".pdf", ".md", ".txt", ".markdown"}
    suffix = _Path(file.filename).suffix.lower()
    
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {allowed}")
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        from ai_core.rag.index_manager import ingest_single_document
        success = ingest_single_document(
            file_path=tmp_path,
            original_filename=file.filename
        )
        
        if success:
            logger.info("✅ Live document upload succeeded", filename=file.filename)
            return {
                "success": True,
                "filename": file.filename,
                "message": f"Successfully added {file.filename} to knowledge base. Future queries will use it via RAG.",
                "description": description or ""
            }
        else:
            raise HTTPException(500, "Ingestion returned False (check server logs)")
            
    except Exception as e:
        logger.error("Upload/document failed", filename=file.filename, error=str(e))
        raise HTTPException(500, f"Upload failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

# WebSocket for future real-time streaming (placeholder)
active_connections = []

@app.websocket("/ws/guidance")
async def websocket_guidance(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        active_connections.remove(websocket)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Multi-Modal Field Service Assistant backend")
    # Warm up the agent graph (triggers intent classifier + full graph build)
    get_field_agent()
    logger.info("Agent graph warmed up - early Intent Classification + branched planner active")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)