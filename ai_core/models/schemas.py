"""
Pydantic models and TypedDicts for the Multi-Modal Field Service Assistant.
Strong typing throughout the agentic system.
"""
from __future__ import annotations
from typing import TypedDict, List, Optional, Literal, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

# =============================================================================
# Enums
# =============================================================================
class Modality(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    MULTIMODAL = "multimodal"

class AgentNode(str, Enum):
    INPUT_FUSION = "input_fusion"
    CONTEXT_LOADER = "context_loader"
    VISION_ANALYSIS = "vision_analysis"
    RETRIEVAL = "retrieval"
    PLANNER = "planner"
    TOOL_EXECUTOR = "tool_executor"
    REFLECTION = "reflection"
    SAFETY_GATE = "safety_gate"
    SYNTHESIZER = "synthesizer"
    MEMORY_UPDATER = "memory_updater"

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"

class SafetyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# =============================================================================
# Core Data Models
# =============================================================================
class Message(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    modality: Optional[Modality] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VisionResult(BaseModel):
    equipment_id: Optional[str] = None
    equipment_model: Optional[str] = None
    equipment_type: Optional[str] = None
    detected_faults: List[str] = Field(default_factory=list)
    visual_description: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    bounding_boxes: List[Dict[str, Any]] = Field(default_factory=list)  # for future AR
    ocr_text: List[str] = Field(default_factory=list)
    raw_vlm_output: Optional[str] = None

class RetrievedDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str  # "manual", "sop", "case", "diagram"
    page_or_section: Optional[str] = None

class RepairStep(BaseModel):
    step_number: int
    description: str
    estimated_time_min: int
    required_tools: List[str] = Field(default_factory=list)
    safety_notes: List[str] = Field(default_factory=list)
    verification_criteria: str = ""
    status: StepStatus = StepStatus.PENDING
    image_reference: Optional[str] = None  # diagram or photo id

class GuidancePlan(BaseModel):
    diagnosis: str
    confidence: float
    steps: List[RepairStep]
    required_parts: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    estimated_total_time_min: int
    escalation_recommended: bool = False
    citations: List[str] = Field(default_factory=list)

class SafetyAssessment(BaseModel):
    overall_risk: SafetyLevel
    flags: List[str]
    required_ppe: List[str]
    lockout_required: bool = False
    permit_required: bool = False
    mitigation_steps: List[str] = Field(default_factory=list)
    confidence: float

class AgentState(TypedDict):
    """The central state object passed through the LangGraph."""
    session_id: str
    technician_id: str
    timestamp: str
    current_modality: Modality
    latest_user_input: str
    current_image_b64: Optional[str]  # base64 encoded
    equipment_context: Dict[str, Any]
    conversation_history: List[Message]
    vision_result: Optional[VisionResult]
    retrieved_docs: List[RetrievedDocument]
    plan: Optional[GuidancePlan]
    safety_assessment: Optional[SafetyAssessment]
    current_step: int
    tools_used: List[str]
    confidence_score: float
    needs_more_info: bool
    needs_escalation: bool
    final_response: Optional[str]
    voice_response: Optional[str]  # TTS-ready text
    trace_id: str

# =============================================================================
# API Request / Response Models
# =============================================================================
class AnalyzeImageRequest(BaseModel):
    image_b64: str
    equipment_hint: Optional[str] = None
    focus: str = "general"

class AnalyzeImageResponse(BaseModel):
    vision_result: VisionResult
    processing_time_ms: int

class VoiceQueryRequest(BaseModel):
    audio_b64: Optional[str] = None  # for server-side STT
    transcript: Optional[str] = None  # if already transcribed client-side
    session_id: str
    image_b64: Optional[str] = None

class GuidanceRequest(BaseModel):
    session_id: str
    transcript: str
    image_b64: Optional[str] = None
    equipment_id: Optional[str] = None
    force_full_agent: bool = False

class GuidanceResponse(BaseModel):
    session_id: str
    plan: Optional[GuidancePlan] = None
    immediate_response: str
    voice_text: str
    citations: List[str]
    confidence: float
    needs_clarification: bool
    safety_warnings: List[str]
    trace_id: str
    next_actions: List[str] = Field(default_factory=list)

class SessionStateResponse(BaseModel):
    session_id: str
    current_equipment: Optional[Dict[str, Any]]
    conversation_length: int
    last_plan_summary: Optional[str]
    confidence: float

# =============================================================================
# Tool Output Schemas (for LangChain tools)
# =============================================================================
class ToolResult(BaseModel):
    success: bool
    data: Any
    error: Optional[str] = None
    tool_name: str
    latency_ms: int

class CaseSummary(BaseModel):
    case_id: str
    date: str
    equipment_model: str
    problem: str
    root_cause: str
    resolution: str
    outcome: str
    technician_notes: Optional[str] = None
