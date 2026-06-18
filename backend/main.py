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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
import structlog
from pathlib import Path as _Path
import tempfile
import shutil

from ai_core.models.schemas import (
    AnalyzeImageRequest, AnalyzeImageResponse,
    GuidanceRequest, GuidanceResponse,
    SessionStateResponse,
    QueryIntent,
    TenantContext
)
from ai_core.agents.field_agent import run_agent_turn, get_field_agent
from ai_core.speech.stt import transcribe_audio
from ai_core.speech.tts import synthesize_speech

# === The production tenant dependency (header-first, JWT-ready) ===
from backend.dependencies.tenant import get_current_tenant

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="Multi-Modal Field Service Assistant API",
    version="1.0.0",
    description="Production multi-tenant SaaS with Director-based industry routing"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory session store (in prod → Redis/DB with tenant isolation)
sessions: dict[str, dict] = {}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "mock_mode": os.getenv("MOCK_MODE", "false"),
        "tenant_system": "production (Depends(get_current_tenant) + X-Tenant-ID header only)"
    }


@app.post("/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image_endpoint(req: AnalyzeImageRequest):
    from ai_core.agents.tools.vision_tool import analyze_image as vision_tool
    from ai_core.agents.tools.image_renderer_tool import annotate_image
    start = time.time()
    result = vision_tool.invoke({
        "image_b64": req.image_b64,
        "focus": req.focus,
        "equipment_hint": req.equipment_hint
    })
    if not result.get("success"):
        raise HTTPException(500, result.get("error"))
    annotated = None
    boxes = (result["data"] or {}).get("bounding_boxes") or []
    if boxes:
        ann = annotate_image.invoke({
            "image_b64": req.image_b64,
            "bounding_boxes": boxes,
            "detected_faults": (result["data"] or {}).get("detected_faults", []),
        })
        if ann.get("success"):
            annotated = ann["data"]
    return AnalyzeImageResponse(
        vision_result=result["data"],
        annotated_image=annotated,
        processing_time_ms=int((time.time() - start) * 1000)
    )


@app.post("/get-guidance", response_model=GuidanceResponse)
async def get_guidance(
    req: GuidanceRequest,
    tenant: TenantContext = Depends(get_current_tenant)   # ← PRODUCTION
):
    """
    Full agent turn.
    The Director (industry router) runs first inside the graph and receives the tenant.
    """
    session_id = req.session_id or str(uuid.uuid4())

    logger.info("get_guidance", tenant=tenant.tenant_id, industry=tenant.industry.value)

    result = await run_agent_turn(
        session_id=session_id,
        user_input=req.transcript,
        image_b64=req.image_b64,
        technician_id="tech-demo",
        tenant_context=tenant
    )

    if not result["success"]:
        raise HTTPException(500, result.get("error"))

    resp = result["response"] or {}

    sessions[session_id] = {
        "last_updated": time.time(),
        "tenant_id": tenant.tenant_id,
        "industry": tenant.industry.value,
        "equipment": (resp.get("vision") or {}).get("equipment_model"),
        "last_confidence": resp.get("confidence"),
        "last_plan_summary": resp.get("plan", {}).get("diagnosis") if resp.get("plan") else None,
    }

    intent_value = resp.get("query_intent")
    is_troubleshooting = bool(intent_value == QueryIntent.TROUBLESHOOTING.value) if intent_value else True

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
        query_intent=intent_value,
        is_troubleshooting=is_troubleshooting,
        query_industry=resp.get("query_industry"),
        was_cross_industry=resp.get("was_cross_industry", False),
        disclaimer=resp.get("disclaimer"),
        tenant_id=tenant.tenant_id,
        director_reasoning=resp.get("director_reasoning"),
        annotated_image=resp.get("annotated_image"),
    )


@app.post("/voice-query")
async def voice_query(req: dict, tenant: TenantContext = Depends(get_current_tenant)):
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

    agent_result = await run_agent_turn(session_id, transcript, image_b64, tenant_context=tenant)
    if not agent_result["success"]:
        raise HTTPException(500, agent_result.get("error"))

    voice_text = agent_result["response"].get("voice", "Processed.")
    audio_out_b64 = synthesize_speech(voice_text)

    return {
        "session_id": session_id,
        "transcript": transcript,
        "response_text": agent_result["response"].get("immediate"),
        "voice_text": voice_text,
        "audio_b64": audio_out_b64,
        "plan": agent_result["response"].get("plan"),
        "confidence": agent_result["response"].get("confidence"),
        "tenant_id": tenant.tenant_id,
        "industry": tenant.industry.value,
        "annotated_image": agent_result["response"].get("annotated_image"),
    }


@app.get("/session/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    s = sessions[session_id]
    return SessionStateResponse(
        session_id=session_id,
        current_equipment={"model": s.get("equipment")} if s.get("equipment") else None,
        conversation_length=1,
        last_plan_summary=s.get("last_plan_summary"),
        confidence=s.get("last_confidence", 0.0),
    )


# Tenant-scoped upload (production)
@app.post("/upload/document")
async def upload_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tenant: TenantContext = Depends(get_current_tenant)
):
    allowed = {".pdf", ".md", ".txt", ".markdown"}
    suffix = _Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        from ai_core.rag.index_manager import ingest_single_document
        success = ingest_single_document(file_path=tmp_path, original_filename=file.filename)

        if success:
            logger.info("document_uploaded", tenant=tenant.tenant_id, industry=tenant.industry.value, filename=file.filename)
            return {
                "success": True,
                "filename": file.filename,
                "tenant_id": tenant.tenant_id,
                "industry": tenant.industry.value,
                "message": "Document added to this tenant's knowledge."
            }
        raise HTTPException(500, "Ingestion failed")
    except Exception as e:
        logger.error("upload_failed", tenant=tenant.tenant_id, error=str(e))
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except: pass


# WebSocket (future streaming)
active_connections = []
@app.websocket("/ws/guidance")
async def websocket_guidance(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.send_text(f"Echo: {await websocket.receive_text()}")
    except WebSocketDisconnect:
        active_connections.remove(websocket)


@app.on_event("startup")
async def startup_event():
    logger.info("backend.start", mode="production-tenant", tenant_system="header-only + Depends(get_current_tenant)")
    get_field_agent()
    logger.info("graph.warmed_up — Director (industry router) is active")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)