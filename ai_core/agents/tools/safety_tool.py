"""
Safety Assessment Tool.
Provides structured risk evaluation for proposed repair plans.
Includes both rule-based hard checks and LLM-based assessment.
"""
import os
import time
import json
from typing import List
from langchain_core.tools import tool
from ai_core.models.schemas import SafetyAssessment, SafetyLevel
import structlog

logger = structlog.get_logger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Hard-coded critical rules (these should never be bypassed)
CRITICAL_KEYWORDS = [
    "high voltage", "live electrical", "confined space", "pressurized system",
    "toxic chemical", "rotating equipment without guard", "loto", "lockout"
]

SAFETY_RULES = {
    "pump": {
        "default_ppe": ["safety glasses", "gloves", "steel-toe boots"],
        "high_risk_actions": ["remove coupling", "work on pressurized line"]
    }
}

@tool
def assess_safety(plan_steps: List[str], equipment_type: str = "general") -> dict:
    """
    Evaluate the safety implications of a proposed repair plan.
    Returns risk level, required PPE, mandatory procedures, and mitigation steps.
    
    This tool MUST be called before presenting any actionable guidance to the technician.
    """
    start = time.time()

    if MOCK_MODE:
        logger.info("Using MOCK safety assessment")
        assessment = {
            "overall_risk": "medium",
            "flags": ["Mechanical seal work requires LOTO", "Fluid may be hot"],
            "required_ppe": ["safety glasses", "chemical resistant gloves", "face shield if draining hot fluid"],
            "lockout_required": True,
            "permit_required": False,
            "mitigation_steps": [
                "Verify zero energy state with calibrated meter",
                "Have spill kit ready",
                "Confirm fluid temperature < 60C before draining"
            ],
            "confidence": 0.91
        }
        latency = int((time.time() - start) * 1000)
        return {
            "success": True,
            "data": assessment,
            "error": None,
            "tool_name": "assess_safety",
            "latency_ms": latency
        }

    # Real assessment logic
    flags = []
    risk_level = SafetyLevel.LOW
    required_ppe = ["safety glasses", "gloves", "steel-toe boots"]
    lockout_required = False
    mitigation = []

    plan_text = " ".join(plan_steps).lower()

    # Hard rules
    if any(kw in plan_text for kw in CRITICAL_KEYWORDS):
        risk_level = SafetyLevel.HIGH
        lockout_required = True
        flags.append("Critical safety procedure required (LOTO / isolation)")

    if "drain" in plan_text or "fluid" in plan_text:
        flags.append("Potential for fluid release - prepare containment")
        mitigation.append("Confirm fluid type and temperature before opening system")

    if "electrical" in plan_text or "motor" in plan_text:
        lockout_required = True
        required_ppe.append("electrical safety gloves")
        flags.append("Electrical isolation required")

    # Simple LLM-based nuance (if not mock)
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=os.getenv("LLM_MODEL", "llama3.2:3b"), temperature=0.0)

        prompt = f"""You are a certified industrial safety officer.

Equipment: {equipment_type}
Proposed plan steps:
{chr(10).join(f"- {s}" for s in plan_steps)}

Assess overall risk level (low/medium/high/critical).
List specific safety flags.
Recommend additional PPE beyond basic hard hat/glasses/gloves.
State whether formal LOTO or permit-to-work is mandatory.
Suggest 2-4 concrete mitigation actions.

Respond ONLY with valid JSON:
{{
  "overall_risk": "low|medium|high|critical",
  "flags": ["..."],
  "required_ppe": ["..."],
  "lockout_required": true/false,
  "permit_required": true/false,
  "mitigation_steps": ["..."],
  "confidence": 0.0-1.0
}}"""

        response = llm.invoke(prompt)
        content = response.content.strip()
        import re
        json_str = re.search(r'\{.*\}', content, re.DOTALL)
        if json_str:
            parsed = json.loads(json_str.group(0))
            # Merge with hard rules
            if parsed.get("overall_risk") in [e.value for e in SafetyLevel]:
                risk_level = SafetyLevel(parsed["overall_risk"])
            flags = list(set(flags + parsed.get("flags", [])))
            required_ppe = list(set(required_ppe + parsed.get("required_ppe", [])))
            lockout_required = lockout_required or parsed.get("lockout_required", False)
            mitigation = list(set(mitigation + parsed.get("mitigation_steps", [])))

    except Exception as e:
        logger.warning("LLM safety nuance failed, using rule-based only", error=str(e))

    assessment_dict = {
        "overall_risk": risk_level.value,
        "flags": flags,
        "required_ppe": required_ppe,
        "lockout_required": lockout_required,
        "permit_required": False,
        "mitigation_steps": mitigation,
        "confidence": 0.85 if not MOCK_MODE else 0.91
    }

    latency = int((time.time() - start) * 1000)
    return {
        "success": True,
        "data": assessment_dict,
        "error": None,
        "tool_name": "assess_safety",
        "latency_ms": latency
    }
