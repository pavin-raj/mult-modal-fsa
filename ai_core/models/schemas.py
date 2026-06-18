"""
Pydantic models and TypedDicts for the Multi-Modal Field Service Assistant.
Strong typing + Intent Classification for production-grade routing.
SaaS multi-tenant extensions for industry-specific deployments.
"""
from __future__ import annotations
from typing import TypedDict, List, Optional, Literal, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

# =============================================================================
# Enums
# =============================================================================
class QueryIntent(str, Enum):
    """Production-grade intent classification."""
    TROUBLESHOOTING = "troubleshooting"           # Equipment issue, fault diagnosis, repair needed
    GENERAL_EXPLANATION = "general_explanation"   # What is X? Explain HAZOP, definition, concept
    PROCEDURE_LOOKUP = "procedure_lookup"         # How to do LOTO, step-by-step procedure
    SAFETY_QUESTION = "safety_question"           # Is it safe to...? Risk assessment
    PARTS_LOOKUP = "parts_lookup"                 # What is the part number for...?
    UNKNOWN = "unknown"                           # Ambiguous or out of scope

class Industry(str, Enum):
    """Generic industry taxonomy for SaaS vertical rollouts. Easy to extend."""
    CONSTRUCTION = "construction"
    WATER_TREATMENT = "water_treatment"
    OIL_GAS = "oil_gas"
    MANUFACTURING = "manufacturing"
    ENERGY = "energy"
    GENERAL_INDUSTRIAL = "general_industrial"
    UNKNOWN = "unknown"

# =============================================================================
# SaaS / Multi-Tenant Models (Director + Tenant Context)
# =============================================================================
class TenantContext(BaseModel):
    """Lightweight tenant context (prototype fake auth → real JWT claims later)."""
    tenant_id: str = Field(..., description="Unique customer/organization identifier")
    industry: Industry = Field(..., description="The primary industry this tenant is licensed for")
    licensed_industries: List[Industry] = Field(default_factory=list, description="All industries this tenant may access")
    company_name: Optional[str] = None
    features: List[str] = Field(default_factory=list, description="Enabled features for this tenant")
    is_active: bool = True

    @property
    def primary_industry(self) -> Industry:
        return self.industry

class DirectorClassification(BaseModel):
    """Structured output from the multi-agent Director (industry router)."""
    primary_industry: Industry = Field(..., description="Best matching industry for this query")
    query_intent: QueryIntent = Field(..., description="The query intent")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Why this industry + intent was chosen")
    is_cross_industry: bool = Field(default=False, description="True if the query is outside the tenant's licensed industry")
    suggested_tools: List[str] = Field(default_factory=list)

class Modality(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    MULTIMODAL = "multimodal"

class AgentNode(str, Enum):
    INPUT_FUSION = "input_fusion"
    INTENT_CLASSIFIER = "intent_classifier"
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
    bounding_boxes: List[Dict[str, Any]] = Field(default_factory=list)
    ocr_text: List[str] = Field(default_factory=list)
    raw_vlm_output: Optional[str] = None

# ---- Add near the other core models (after VisionResult) ----
class AnnotatedImage(BaseModel):
    """The user's input image with detected faults drawn on it.
    `image_b64` is a data URL (data:image/png;base64,...) so the frontend
    can drop it straight into an <img src>."""
    image_b64: str
    format: str = "png"
    width: int = 0
    height: int = 0
    boxes_drawn: int = 0
    labels: List[str] = Field(default_factory=list)
    caption: Optional[str] = None

class RetrievedDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str
    page_or_section: Optional[str] = None

class RepairStep(BaseModel):
    step_number: int
    description: str
    estimated_time_min: int
    required_tools: List[str] = Field(default_factory=list)
    safety_notes: List[str] = Field(default_factory=list)
    verification_criteria: str = ""
    status: StepStatus = StepStatus.PENDING
    image_reference: Optional[str] = None

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

class IntentClassification(BaseModel):
    intent: QueryIntent = Field(..., description="The primary intent of the user's query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="How confident the classifier is")
    reasoning: str = Field(..., description="Brief explanation of why this intent was chosen")
    requires_image: bool = Field(default=False, description="Whether the query would benefit from an image")
    suggested_tools: List[str] = Field(default_factory=list, description="Which tools are likely needed")

class AgentState(TypedDict):
    """Central state object passed through the LangGraph."""
    session_id: str
    technician_id: str
    timestamp: str
    current_modality: Modality
    latest_user_input: str
    current_image_b64: Optional[str]
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
    voice_response: Optional[str]
    trace_id: str
    annotated_image: Optional[AnnotatedImage]
    
    # Intent Classification (Production)
    query_intent: Optional[QueryIntent]
    intent_confidence: Optional[float]
    intent_reasoning: Optional[str]

    # SaaS / Multi-Tenant Fields (Director + Tenant Context)
    tenant_context: Optional[TenantContext]
    query_industry: Optional[Industry]
    was_cross_industry: Optional[bool]
    director_reasoning: Optional[str]
    disclaimer: Optional[str]

# =============================================================================
# API Request / Response Models
# =============================================================================
class AnalyzeImageRequest(BaseModel):
    image_b64: str
    equipment_hint: Optional[str] = None
    focus: str = "general"

class AnalyzeImageResponse(BaseModel):
    vision_result: VisionResult
    annotated_image: Optional[AnnotatedImage] = None
    processing_time_ms: int

class GuidanceRequest(BaseModel):
    session_id: str
    transcript: str
    image_b64: Optional[str] = None
    equipment_id: Optional[str] = None
    force_full_agent: bool = False

    # SaaS Prototype Fields (fake tenant context - will become JWT claims)
    tenant_id: Optional[str] = Field(default="demo-tenant-001", description="Customer/organization ID")
    industry: Optional[Industry] = Field(default=Industry.CONSTRUCTION, description="Tenant's primary licensed industry")

class GuidanceResponse(BaseModel):
    session_id: str
    plan: Optional[GuidancePlan] = None
    annotated_image: Optional[AnnotatedImage] = None
    immediate_response: str
    voice_text: str
    citations: List[str]
    confidence: float
    needs_clarification: bool
    safety_warnings: List[str]
    trace_id: str
    next_actions: List[str] = Field(default_factory=list)
    
    # Intent + SaaS routing info
    query_intent: Optional[QueryIntent] = None
    is_troubleshooting: bool = True
    
    # SaaS fields
    query_industry: Optional[Industry] = None
    was_cross_industry: bool = False
    disclaimer: Optional[str] = None
    tenant_id: Optional[str] = None
    director_reasoning: Optional[str] = None

# =============================================================================
# Tool Output Schemas
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

class SessionStateResponse(BaseModel):
    session_id: str
    current_equipment: Optional[Dict[str, Any]] = None
    conversation_length: int = 0
    last_plan_summary: Optional[str] = None
    confidence: float = 0.0
