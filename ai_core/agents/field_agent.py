"""
Core Agentic Reasoning Engine for Multi-Modal Field Service Assistant.

Built with LangGraph for structured, reliable, step-by-step decision support.
This is the "brain" that orchestrates vision, RAG, safety, and planning.
"""
import os
import uuid
import time
from typing import Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool

from ai_core.models.schemas import (
    AgentState, 
    GuidancePlan, 
    RepairStep, 
    SafetyAssessment,
    QueryIntent,
    IntentClassification,
    Industry,
    TenantContext,
    DirectorClassification
)
from ai_core.agents.tools.vision_tool import analyze_image
from ai_core.agents.tools.rag_tool import retrieve_knowledge
from ai_core.agents.tools.safety_tool import assess_safety
from ai_core.agents.intent_classifier import classify_intent
from ai_core.agents.director import classify_with_director, build_disclaimer

import structlog

logger = structlog.get_logger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
MAX_STEPS = int(os.getenv("MAX_AGENT_STEPS", "12"))


# =============================================================================
# LLM Setup
# =============================================================================
def get_llm(temperature: float = 0.1):
    if MOCK_MODE:
        class MockLLM:
            def invoke(self, messages):
                class Resp:
                    content = '{"diagnosis": "Mock diagnosis", "confidence": 0.75, "steps": [{"step_number":1,"description":"Mock step - isolate equipment","estimated_time_min":5,"required_tools":["multimeter"],"safety_notes":["Wear PPE"],"verification_criteria":"Equipment is safe","status":"pending"}],"required_parts":["SEAL-X450-22"],"warnings":["Follow LOTO"],"estimated_total_time_min":45,"escalation_recommended":false,"citations":["SOP-EL-003"]}'
                return Resp()
        return MockLLM()
    return ChatOllama(model=LLM_MODEL, temperature=temperature, num_predict=1400)


# =============================================================================
# Tool Setup
# =============================================================================
tools = [analyze_image, retrieve_knowledge, assess_safety]
tool_node = ToolNode(tools)


# =============================================================================
# Graph Nodes
# =============================================================================

def input_fusion(state: AgentState) -> AgentState:
    logger.info("Node: input_fusion", session=state["session_id"])
    if not state.get("trace_id"):
        state["trace_id"] = str(uuid.uuid4())
    state["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    return state


def director_node(state: AgentState) -> AgentState:
    """
    Multi-agent Director (the router for industry-specific SaaS).
    
    Runs very early (right after input_fusion).
    Classifies primary industry + intent, detects cross-industry queries,
    and injects tenant context + disclaimer into the state for all downstream nodes.
    
    This is the heart of the "Director + industry-aware sub-agents" pattern.
    """
    logger.info("Node: director (multi-agent router)")
    
    user_input = state.get("latest_user_input", "")
    has_image = bool(state.get("current_image_b64"))
    
    # Extract tenant context (prototype uses what's passed in; later from JWT)
    tenant_ctx = state.get("tenant_context")
    if tenant_ctx is None:
        # Prototype fallback - treat as construction demo tenant
        tenant_ctx = TenantContext(
            tenant_id="demo-tenant-001",
            industry=Industry.CONSTRUCTION,
            licensed_industries=[Industry.CONSTRUCTION],
            company_name="Demo Construction Co"
        )
        state["tenant_context"] = tenant_ctx
    
    # Run the Director classification
    classification: DirectorClassification = classify_with_director(
        user_input=user_input,
        tenant_context=tenant_ctx,
        has_image=has_image
    )
    
    # Inject into state (used by planner, RAG, synthesizer, logs, frontend)
    state["query_industry"] = classification.primary_industry
    state["was_cross_industry"] = classification.is_cross_industry
    state["director_reasoning"] = classification.reasoning
    
    # Generate liability disclaimer when cross-industry (or when using only general knowledge)
    disclaimer = build_disclaimer(classification, tenant_ctx)
    state["disclaimer"] = disclaimer
    
    # Also keep the original query_intent for backward compatibility with existing branches
    # (the old intent_classifier_node can stay or be merged later)
    state["query_intent"] = classification.query_intent
    state["intent_confidence"] = classification.confidence
    state["intent_reasoning"] = classification.reasoning
    
    logger.info(
        "Director routing complete",
        tenant=tenant_ctx.tenant_id,
        licensed=tenant_ctx.industry.value,
        classified=classification.primary_industry.value,
        intent=classification.query_intent.value,
        cross_industry=classification.is_cross_industry,
        has_disclaimer=bool(disclaimer)
    )
    
    return state


def intent_classifier_node(state: AgentState) -> AgentState:
    """
    Production Intent Classification node.
    Runs early so we can route differently for troubleshooting vs knowledge queries.
    """
    logger.info("Node: intent_classifier")
    
    classification: IntentClassification = classify_intent(
        user_input=state.get("latest_user_input", ""),
        has_image=bool(state.get("current_image_b64"))
    )
    
    state["query_intent"] = classification.intent
    state["intent_confidence"] = classification.confidence
    state["intent_reasoning"] = classification.reasoning
    
    logger.info(
        "Intent classified",
        intent=classification.intent.value,
        confidence=classification.confidence,
        reasoning=classification.reasoning[:100]
    )
    return state


def context_loader(state: AgentState) -> AgentState:
    logger.info("Node: context_loader")
    if not state.get("equipment_context"):
        state["equipment_context"] = {
            "id": "UNKNOWN",
            "model": "Unknown Equipment",
            "location": "Unknown",
            "last_service": "N/A",
            "known_issues": []
        }
    return state


def vision_node(state: AgentState) -> AgentState:
    if not state.get("current_image_b64"):
        return state
    
    logger.info("Node: vision_analysis")
    result = analyze_image.invoke({
        "image_b64": state["current_image_b64"],
        "focus": "fault detection and equipment identification",
        "equipment_hint": state.get("equipment_context", {}).get("model")
    })
    
    if result.get("success"):
        state["vision_result"] = result["data"]
        state["tools_used"].append("analyze_image")
    return state


def retrieval_node(state: AgentState) -> AgentState:
    logger.info("Node: retrieval")
    
    query_parts = [state.get("latest_user_input", "")]
    if state.get("vision_result"):
        query_parts.append(state["vision_result"].get("visual_description", ""))
        query_parts.append(" ".join(state["vision_result"].get("detected_faults", [])))
    
    query = " ".join(query_parts).strip() or "troubleshooting procedure"
    model = state.get("equipment_context", {}).get("model")
    
    result = retrieve_knowledge.invoke({
        "query": query,
        "equipment_model": model,
        "top_k": 6
    })
    
    if result.get("success"):
        state["retrieved_docs"] = result["data"]
        state["tools_used"].append("retrieve_knowledge")
    return state


def planner_node(state: AgentState) -> AgentState:
    """
    Intelligent planner that branches based on QueryIntent.
    This is the core of the "proper production way".
    """
    logger.info("Node: planner", intent=state.get("query_intent"))
    
    intent = state.get("query_intent", QueryIntent.UNKNOWN)
    
    # Base context
    context = f"""User Query: {state.get('latest_user_input')}
Equipment Context: {state.get('equipment_context')}
Vision Analysis: {state.get('vision_result')}
Retrieved Knowledge (top excerpts):
{chr(10).join(f"- {d.get('content','')[:280]}..." for d in state.get('retrieved_docs', [])[:3])}
"""

    # === MOCK SHORT-CIRCUIT (guarantees branching for demo / when no Ollama) ===
    if MOCK_MODE:
        if intent == QueryIntent.TROUBLESHOOTING:
            state["plan"] = {
                "diagnosis": "Worn mechanical seal and/or impeller misalignment (based on symptoms + past cases).",
                "confidence": 0.78,
                "steps": [
                    {"step_number": 1, "description": "Isolate power and verify zero energy state (LOTO)", "estimated_time_min": 5, "required_tools": ["lockout kit"], "safety_notes": ["Follow SOP-EL-003", "Wear PPE"], "verification_criteria": "No voltage present", "status": "pending"},
                    {"step_number": 2, "description": "Drain system and remove coupling guard", "estimated_time_min": 8, "required_tools": ["wrench set"], "safety_notes": ["Confirm zero pressure"], "verification_criteria": "System drained", "status": "pending"},
                    {"step_number": 3, "description": "Replace mechanical seal (SEAL-X450-22) and check shaft runout", "estimated_time_min": 25, "required_tools": ["seal puller", "dial indicator"], "safety_notes": ["Support shaft during removal"], "verification_criteria": "Runout < 0.002 inch", "status": "pending"}
                ],
                "required_parts": ["SEAL-X450-22", "SHAFT-X450"],
                "warnings": ["Always perform full alignment after seal work"],
                "estimated_total_time_min": 45,
                "escalation_recommended": False,
                "citations": ["X-450-Centrifugal-Pump-Manual.md", "past_cases.json"]
            }
            state["confidence_score"] = 0.78
        else:
            # GENERAL_EXPLANATION / PROCEDURE_LOOKUP / etc. → empty steps, clear explanation
            if "hazop" in (state.get("latest_user_input", "").lower()):
                diagnosis = "HAZOP (Hazard and Operability Study) is a structured technique to identify potential hazards and operability problems in process systems before they occur."
            else:
                diagnosis = "This is general industrial knowledge / procedure information (not an active fault report)."
            state["plan"] = {
                "diagnosis": diagnosis,
                "confidence": 0.85,
                "steps": [],   # intentionally empty for non-troubleshooting
                "required_parts": [],
                "warnings": [],
                "estimated_total_time_min": 0,
                "escalation_recommended": False,
                "citations": ["X-450-Centrifugal-Pump-Manual.md", "SOP-EL-003-Lockout-Tagout.md"]
            }
            state["confidence_score"] = 0.85
        
        state["tools_used"].append("planner")
        return state

    # === REAL LLM PATH ===
    llm = get_llm(temperature=0.15)
    
    if intent == QueryIntent.TROUBLESHOOTING:
        system = SystemMessage(content="""You are an expert senior field service technician and diagnostician.

Your job is to produce a clear, safe, actionable repair plan.

Output ONLY valid JSON:
{
  "diagnosis": "concise root cause hypothesis",
  "confidence": 0.0-1.0,
  "steps": [ { "step_number": 1, "description": "...", "estimated_time_min": 5, "required_tools": [...], "safety_notes": [...], "verification_criteria": "...", "status": "pending" } ],
  "required_parts": [...],
  "warnings": [...],
  "estimated_total_time_min": 30,
  "escalation_recommended": false,
  "citations": ["document references"]
}

Rules:
- Every step must include safety_notes.
- Base recommendations on retrieved knowledge when possible.
- If confidence < 0.6, set escalation_recommended=true.
""")
    
    else:
        # General explanation / procedure / safety / unknown
        system = SystemMessage(content="""You are a helpful industrial knowledge assistant.

The user is asking a general question (not necessarily reporting a fault).

Output ONLY valid JSON in this format:
{
  "diagnosis": "Clear, direct answer to the user's question (1-2 sentences)",
  "confidence": 0.0-1.0,
  "steps": [],                    // Leave empty for non-troubleshooting queries
  "required_parts": [],
  "warnings": [],
  "estimated_total_time_min": 0,
  "escalation_recommended": false,
  "citations": ["relevant document references if any"]
}

Be accurate and helpful. If the question is about a standard (HAZOP, LOTO, etc.), give a clear explanation + key points.
""")

    human = HumanMessage(content=f"Current situation:\n{context}\n\nProduce the JSON response now.")

    try:
        response = llm.invoke([system, human])
        import json, re
        content = response.content.strip()
        json_str = re.search(r'\{.*\}', content, re.DOTALL)
        plan_dict = json.loads(json_str.group(0)) if json_str else json.loads(content)
        
        plan = GuidancePlan(**plan_dict)
        state["plan"] = plan.model_dump()
        state["confidence_score"] = plan.confidence
        state["tools_used"].append("planner")
        
    except Exception as e:
        logger.error("Planner failed", error=str(e), intent=intent)
        # Safe fallback
        state["plan"] = {
            "diagnosis": "Unable to generate a detailed response. Please provide more details or try again.",
            "confidence": 0.3,
            "steps": [],
            "required_parts": [],
            "warnings": ["System could not generate a reliable answer"],
            "estimated_total_time_min": 0,
            "escalation_recommended": False,
            "citations": []
        }
        state["confidence_score"] = 0.3
    
    return state


def safety_gate_node(state: AgentState) -> AgentState:
    """Only run safety assessment for troubleshooting queries."""
    intent = state.get("query_intent", QueryIntent.UNKNOWN)
    
    if intent != QueryIntent.TROUBLESHOOTING:
        logger.info("Skipping safety gate for non-troubleshooting intent", intent=intent)
        return state
    
    logger.info("Node: safety_gate")
    
    plan = state.get("plan", {})
    steps = [s.get("description", "") for s in plan.get("steps", [])]
    equip_type = state.get("equipment_context", {}).get("model", "general")
    
    result = assess_safety.invoke({"plan_steps": steps, "equipment_type": equip_type})
    
    if result.get("success"):
        state["safety_assessment"] = result["data"]
        state["tools_used"].append("assess_safety")
        
        if state.get("plan"):
            state["plan"]["safety_assessment"] = result["data"]
    
    return state


def reflection_node(state: AgentState) -> AgentState:
    logger.info("Node: reflection")
    
    confidence = state.get("confidence_score", 0.5)
    # CRITICAL FIX: initial_state sets safety_assessment=None explicitly.
    # state.get(key, default) returns the stored None (key present), not the default.
    # Must use "or {}" or explicit None check.
    safety = state.get("safety_assessment") or {}
    intent = state.get("query_intent", QueryIntent.UNKNOWN)
    
    state["needs_more_info"] = confidence < 0.55
    state["needs_escalation"] = (
        confidence < 0.45 or 
        (isinstance(safety, dict) and safety.get("overall_risk") in ["high", "critical"]) or
        (state.get("plan") or {}).get("escalation_recommended", False)
    )
    
    return state


def synthesizer_node(state: AgentState) -> AgentState:
    """
    Final response synthesizer.
    Creates different output styles depending on the intent.
    """
    logger.info("Node: synthesizer")
    
    plan = state.get("plan", {})
    vision = state.get("vision_result")
    safety = state.get("safety_assessment", {})
    intent = state.get("query_intent", QueryIntent.UNKNOWN)
    
    immediate = []
    
    if vision:
        immediate.append(f"Equipment identified: {vision.get('equipment_model', 'Unknown')}")
        if vision.get("detected_faults"):
            immediate.append(f"Visible issues: {', '.join(vision['detected_faults'])}")
    
    if plan.get("diagnosis"):
        immediate.append(plan["diagnosis"])
    
    immediate_text = ". ".join(immediate) if immediate else "I have processed your query."
    
    # Voice-friendly version
    if intent == QueryIntent.TROUBLESHOOTING:
        voice = f"{immediate_text}. "
        if plan.get("steps"):
            first_step = plan["steps"][0]["description"]
            voice += f"First recommended step: {first_step}. "
        if safety.get("lockout_required"):
            voice += "Remember: Lockout Tagout is required. "
        voice += "Do you want to proceed with the next step?"
    else:
        voice = immediate_text
        if plan.get("citations"):
            voice += " Sources: " + ", ".join(plan["citations"][:2]) + "."
    
    state["final_response"] = immediate_text
    state["voice_response"] = voice
    
    return state


def memory_updater_node(state: AgentState) -> AgentState:
    logger.info("Node: memory_updater", session=state["session_id"])
    # In production: persist structured observations, update equipment profiles, etc.
    return state


# =============================================================================
# Conditional Edges
# =============================================================================
def should_run_vision(state: AgentState) -> Literal["vision_analysis", "retrieval"]:
    if state.get("current_image_b64") and not state.get("vision_result"):
        return "vision_analysis"
    return "retrieval"


# =============================================================================
# Graph Construction
# =============================================================================
def build_field_service_agent():
    """Production LangGraph with Intent Classification."""
    workflow = StateGraph(AgentState)
    
    # Nodes
    workflow.add_node("input_fusion", input_fusion)
    workflow.add_node("director", director_node)                 # NEW: SaaS multi-agent Director (industry router)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("context_loader", context_loader)
    workflow.add_node("vision_analysis", vision_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("safety_gate", safety_gate_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("memory_updater", memory_updater_node)
    
    # Flow - SaaS Multi-tenant with Director (industry router) first
    workflow.set_entry_point("input_fusion")
    workflow.add_edge("input_fusion", "director")          # NEW: Multi-agent Director (industry + intent router)
    workflow.add_edge("director", "intent_classifier")     # Keep old intent classifier for now (can be merged later)
    workflow.add_edge("intent_classifier", "context_loader")
    
    workflow.add_conditional_edges(
        "context_loader",
        should_run_vision,
        {"vision_analysis": "vision_analysis", "retrieval": "retrieval"}
    )
    workflow.add_edge("vision_analysis", "retrieval")
    workflow.add_edge("retrieval", "planner")
    workflow.add_edge("planner", "safety_gate")
    workflow.add_edge("safety_gate", "reflection")
    workflow.add_edge("reflection", "synthesizer")
    workflow.add_edge("synthesizer", "memory_updater")
    workflow.add_edge("memory_updater", END)
    
    return workflow.compile()


# Singleton
_field_agent = None

def get_field_agent():
    global _field_agent
    if _field_agent is None:
        _field_agent = build_field_service_agent()
        logger.info("Production Field Service Agent graph compiled with Intent Classification")
    return _field_agent


# Convenience async runner (kept for backward compatibility)
async def run_agent_turn(
    session_id: str,
    user_input: str,
    image_b64: str | None = None,
    technician_id: str = "tech-001",
    # NEW: SaaS tenant context (fake for prototype, JWT claims later)
    tenant_context: Optional[TenantContext] = None,
) -> dict:
    agent = get_field_agent()
    
    # Build tenant context (prototype default)
    if tenant_context is None:
        tenant_context = TenantContext(
            tenant_id="demo-tenant-001",
            industry=Industry.CONSTRUCTION,
            licensed_industries=[Industry.CONSTRUCTION],
            company_name="Demo Tenant"
        )
    
    initial_state: AgentState = {
        "session_id": session_id,
        "technician_id": technician_id,
        "timestamp": "",
        "current_modality": "multimodal" if image_b64 else "voice",
        "latest_user_input": user_input,
        "current_image_b64": image_b64,
        "equipment_context": {},
        "conversation_history": [],
        "vision_result": None,
        "retrieved_docs": [],
        "plan": None,
        "safety_assessment": None,
        "current_step": 0,
        "tools_used": [],
        "confidence_score": 0.0,
        "needs_more_info": False,
        "needs_escalation": False,
        "final_response": None,
        "voice_response": None,
        "trace_id": "",
        "query_intent": None,
        "intent_confidence": None,
        "intent_reasoning": None,
        # NEW SaaS fields
        "tenant_context": tenant_context,
        "query_industry": None,
        "was_cross_industry": False,
        "director_reasoning": None,
        "disclaimer": None,
    }
    
    try:
        final_state = await agent.ainvoke(initial_state)
        
        # Extract SaaS fields for the response
        response_payload = {
            "immediate": final_state.get("final_response"),
            "voice": final_state.get("voice_response"),
            "plan": final_state.get("plan"),
            "vision": final_state.get("vision_result"),
            "confidence": final_state.get("confidence_score"),
            "needs_escalation": final_state.get("needs_escalation"),
            "trace_id": final_state.get("trace_id"),
            "tools_used": final_state.get("tools_used", []),
            "query_intent": final_state.get("query_intent"),
            "intent_confidence": final_state.get("intent_confidence"),
            # SaaS multi-tenant fields
            "query_industry": final_state.get("query_industry"),
            "was_cross_industry": final_state.get("was_cross_industry", False),
            "disclaimer": final_state.get("disclaimer"),
            "director_reasoning": final_state.get("director_reasoning"),
            "tenant_id": final_state.get("tenant_context", {}).tenant_id if final_state.get("tenant_context") else None,
        }
        
        return {
            "success": True,
            "state": final_state,
            "response": response_payload
        }
    except Exception as e:
        logger.error("Agent run failed", error=str(e), exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "response": {
                "immediate": "I encountered an error while processing your query. Please try again.",
                "voice": "Sorry, there was a problem processing your request.",
                "plan": None,
                "confidence": 0.0,
                "needs_escalation": True,
                "query_intent": QueryIntent.UNKNOWN,
                "query_industry": Industry.UNKNOWN,
                "was_cross_industry": False,
                "disclaimer": "An error occurred. Please try again.",
            }
        }
