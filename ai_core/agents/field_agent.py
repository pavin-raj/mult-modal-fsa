"""
Core Agentic Reasoning Engine for Multi-Modal Field Service Assistant.

Built with LangGraph for structured, reliable, step-by-step decision support.
This is the "brain" that orchestrates vision, RAG, safety, and planning.
"""
import os
import uuid
import time
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool

from ai_core.models.schemas import AgentState, GuidancePlan, RepairStep, SafetyAssessment
from ai_core.agents.tools.vision_tool import analyze_image
from ai_core.agents.tools.rag_tool import retrieve_knowledge
from ai_core.agents.tools.safety_tool import assess_safety
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
                    content = '{"diagnosis": "Mock diagnosis from agent", "steps": [{"step_number":1,"description":"Mock step - isolate equipment","estimated_time_min":5,"required_tools":["multimeter"],"safety_notes":["Wear PPE"],"verification_criteria":"Equipment is safe to work on","status":"pending"}],"required_parts":["SEAL-X450-22"],"warnings":["Follow LOTO"],"estimated_total_time_min":45,"escalation_recommended":false,"citations":["SOP-EL-003","MAN-X450-Rev4"]}'
                return Resp()
        return MockLLM()
    return ChatOllama(model=LLM_MODEL, temperature=temperature, num_predict=1200)

# =============================================================================
# Tool Setup
# =============================================================================
tools = [analyze_image, retrieve_knowledge, assess_safety]
tool_node = ToolNode(tools)

# =============================================================================
# Graph Nodes
# =============================================================================
def input_fusion(state: AgentState) -> AgentState:
    """Combine latest inputs and set initial context."""
    logger.info("Node: input_fusion", session=state["session_id"])
    if not state.get("trace_id"):
        state["trace_id"] = str(uuid.uuid4())
    state["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # If image present but no vision yet, mark for analysis
    if state.get("current_image_b64") and not state.get("vision_result"):
        state["current_modality"] = "multimodal"
    
    return state

def context_loader(state: AgentState) -> AgentState:
    """Load persistent context (simplified for now - would hit DB in prod)."""
    logger.info("Node: context_loader")
    if not state.get("equipment_context"):
        state["equipment_context"] = {
            "id": "PUMP-X450-001",
            "model": "X-450 Centrifugal Pump",
            "location": "Building A - Cooling Tower",
            "last_service": "2026-01-15",
            "known_issues": ["previous seal replacement 8 months ago"]
        }
    return state

def vision_node(state: AgentState) -> AgentState:
    """Call vision tool if image is available."""
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
    """Retrieve relevant knowledge."""
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
    """Generate structured diagnosis and step-by-step plan."""
    logger.info("Node: planner")
    llm = get_llm(temperature=0.2)
    
    context = f"""Equipment Context: {state.get('equipment_context')}
Vision Analysis: {state.get('vision_result')}
Retrieved Knowledge (top excerpts):
{chr(10).join(f"- {d.get('content','')[:300]}..." for d in state.get('retrieved_docs', [])[:3])}
User Query: {state.get('latest_user_input')}
"""
    
    system = SystemMessage(content="""You are an expert senior field service technician and diagnostician with 20+ years experience.

Your job is to produce a clear, safe, actionable repair plan.

Always output valid JSON matching this exact schema (no extra text):
{
  "diagnosis": "concise root cause hypothesis",
  "confidence": 0.0-1.0,
  "steps": [
    {
      "step_number": 1,
      "description": "Clear actionable instruction",
      "estimated_time_min": 5,
      "required_tools": ["list"],
      "safety_notes": ["list of warnings"],
      "verification_criteria": "How to know this step succeeded",
      "status": "pending"
    }
  ],
  "required_parts": ["part numbers or descriptions"],
  "warnings": ["critical safety or process warnings"],
  "estimated_total_time_min": 30,
  "escalation_recommended": false,
  "citations": ["document references"]
}

Rules:
- Every step must include safety_notes.
- If confidence < 0.65, set escalation_recommended=true and add warning.
- Prioritize LOTO and PPE in early steps when relevant.
- Base every recommendation on the retrieved knowledge when possible.
""")

    human = HumanMessage(content=f"Current situation:\n{context}\n\nProduce the JSON plan now.")

    try:
        response = llm.invoke([system, human])
        import json, re
        content = response.content.strip()
        json_str = re.search(r'\{.*\}', content, re.DOTALL)
        if json_str:
            plan_dict = json.loads(json_str.group(0))
        else:
            plan_dict = json.loads(content)
        
        plan = GuidancePlan(**plan_dict)
        state["plan"] = plan.model_dump()
        state["confidence_score"] = plan.confidence
        state["tools_used"].append("planner")
    except Exception as e:
        logger.error("Planner failed", error=str(e))
        # Fallback minimal plan
        state["plan"] = {
            "diagnosis": "Unable to generate full plan. Defaulting to safe isolation.",
            "confidence": 0.4,
            "steps": [{"step_number": 1, "description": "Stop work and isolate equipment using LOTO. Contact supervisor.", "estimated_time_min": 5, "required_tools": [], "safety_notes": ["Do not proceed without confirmation"], "verification_criteria": "Equipment isolated", "status": "pending"}],
            "required_parts": [],
            "warnings": ["System could not generate reliable plan"],
            "estimated_total_time_min": 5,
            "escalation_recommended": True,
            "citations": []
        }
        state["confidence_score"] = 0.4
    
    return state

def safety_gate_node(state: AgentState) -> AgentState:
    """Run mandatory safety assessment."""
    logger.info("Node: safety_gate")
    
    plan = state.get("plan", {})
    steps = [s.get("description", "") for s in plan.get("steps", [])]
    equip_type = state.get("equipment_context", {}).get("model", "general")
    
    result = assess_safety.invoke({"plan_steps": steps, "equipment_type": equip_type})
    
    if result.get("success"):
        state["safety_assessment"] = result["data"]
        state["tools_used"].append("assess_safety")
        
        # Inject safety into plan
        if state.get("plan"):
            state["plan"]["safety_assessment"] = result["data"]
            if result["data"].get("lockout_required"):
                state["plan"]["warnings"] = state["plan"].get("warnings", []) + ["LOCKOUT/TAGOUT REQUIRED - Do not proceed without verification"]
    
    return state

def reflection_node(state: AgentState) -> AgentState:
    """Self-critique and decide next action."""
    logger.info("Node: reflection")
    
    confidence = state.get("confidence_score", 0.5)
    safety = state.get("safety_assessment", {})
    
    state["needs_more_info"] = confidence < 0.6
    state["needs_escalation"] = (
        confidence < 0.5 or 
        safety.get("overall_risk") in ["high", "critical"] or
        state.get("plan", {}).get("escalation_recommended", False)
    )
    
    return state

def synthesizer_node(state: AgentState) -> AgentState:
    """Create final technician-friendly response (text + voice)."""
    logger.info("Node: synthesizer")
    
    plan = state.get("plan", {})
    vision = state.get("vision_result")
    safety = state.get("safety_assessment", {})
    
    # Build immediate response
    immediate = []
    if vision:
        immediate.append(f"Equipment identified: {vision.get('equipment_model', 'Unknown')}")
        if vision.get("detected_faults"):
            immediate.append(f"Visible issues: {', '.join(vision['detected_faults'])}")
    
    if plan.get("diagnosis"):
        immediate.append(f"Diagnosis: {plan['diagnosis']}")
    
    immediate_text = ". ".join(immediate) if immediate else "Analyzing situation..."
    
    # Voice-friendly summary
    voice = f"{immediate_text}. "
    if plan.get("steps"):
        first_step = plan["steps"][0]["description"]
        voice += f"First step: {first_step}. "
    if safety.get("lockout_required"):
        voice += "Remember: Lockout Tagout is required. "
    voice += "Do you want to proceed with the next step, or provide more information?"
    
    state["final_response"] = immediate_text
    state["voice_response"] = voice
    
    return state

def memory_updater_node(state: AgentState) -> AgentState:
    """Log the interaction for future retrieval (stub)."""
    logger.info("Node: memory_updater", session=state["session_id"])
    # In production: persist to Postgres + update vector store with new observation
    # For now we just log
    return state

# =============================================================================
# Conditional Edges
# =============================================================================
def should_run_vision(state: AgentState) -> Literal["vision_analysis", "retrieval"]:
    if state.get("current_image_b64") and not state.get("vision_result"):
        return "vision_analysis"
    return "retrieval"

def should_continue(state: AgentState) -> Literal["tool_executor", "synthesizer"]:
    # For simplicity in this version, we run tools via explicit nodes
    # In more advanced versions you would use ReAct style here
    return "synthesizer"

# =============================================================================
# Graph Construction
# =============================================================================
def build_field_service_agent():
    """Construct and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("input_fusion", input_fusion)
    workflow.add_node("context_loader", context_loader)
    workflow.add_node("vision_analysis", vision_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("safety_gate", safety_gate_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("memory_updater", memory_updater_node)
    
    # Define flow
    workflow.set_entry_point("input_fusion")
    workflow.add_edge("input_fusion", "context_loader")
    workflow.add_conditional_edges(
        "context_loader",
        should_run_vision,
        {
            "vision_analysis": "vision_analysis",
            "retrieval": "retrieval"
        }
    )
    workflow.add_edge("vision_analysis", "retrieval")
    workflow.add_edge("retrieval", "planner")
    workflow.add_edge("planner", "safety_gate")
    workflow.add_edge("safety_gate", "reflection")
    workflow.add_edge("reflection", "synthesizer")
    workflow.add_edge("synthesizer", "memory_updater")
    workflow.add_edge("memory_updater", END)
    
    return workflow.compile()

# Singleton compiled graph
_field_agent = None

def get_field_agent():
    global _field_agent
    if _field_agent is None:
        _field_agent = build_field_service_agent()
        logger.info("Field Service Agent graph compiled successfully")
    return _field_agent

# Convenience function for direct invocation
async def run_agent_turn(
    session_id: str,
    user_input: str,
    image_b64: str | None = None,
    technician_id: str = "tech-001"
) -> dict:
    """Run one full turn of the agent and return final state + response."""
    agent = get_field_agent()
    
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
        "trace_id": ""
    }
    
    try:
        final_state = await agent.ainvoke(initial_state)
        return {
            "success": True,
            "state": final_state,
            "response": {
                "immediate": final_state.get("final_response"),
                "voice": final_state.get("voice_response"),
                "plan": final_state.get("plan"),
                "vision": final_state.get("vision_result"),
                "confidence": final_state.get("confidence_score"),
                "needs_escalation": final_state.get("needs_escalation"),
                "trace_id": final_state.get("trace_id"),
                "tools_used": final_state.get("tools_used", [])
            }
        }
    except Exception as e:
        logger.error("Agent run failed", error=str(e), exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "response": {
                "immediate": "I encountered an error while reasoning. Please try again or contact support.",
                "voice": "Sorry, there was a problem processing your request.",
                "plan": None,
                "confidence": 0.0,
                "needs_escalation": True
            }
        }
