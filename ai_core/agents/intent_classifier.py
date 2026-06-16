"""
Production-grade Intent Classifier for Multi-Modal Field Service Assistant.

This is a dedicated, early node in the agent graph.
It uses structured output (Pydantic) for reliability — a best practice in production agentic systems.

Why a separate classifier?
- Allows different reasoning paths (troubleshooting vs knowledge vs safety).
- Improves latency (we can skip heavy nodes for simple queries).
- Better observability and routing.
- Prevents the "everything is a repair plan" problem we had before.
"""

import os
from typing import Optional
import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from ai_core.models.schemas import QueryIntent, IntentClassification

logger = structlog.get_logger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")


class _IntentOutput(BaseModel):
    """Internal structured output for the LLM."""
    intent: QueryIntent = Field(..., description="The primary intent of the user's query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier confidence")
    reasoning: str = Field(..., description="Short explanation for the chosen intent")
    requires_image: bool = Field(default=False)
    suggested_tools: list[str] = Field(default_factory=list)


INTENT_SYSTEM_PROMPT = """You are an expert intent classifier for an industrial field service AI assistant.

Classify the user's query into exactly one of these categories:

- **troubleshooting**: User is reporting a problem with equipment (vibration, leak, noise, fault, not working, etc.). They want diagnosis and repair guidance.
- **general_explanation**: User wants to understand a concept, definition, or standard (e.g. "What is HAZOP?", "Explain LOTO", "What is a mechanical seal?").
- **procedure_lookup**: User wants the steps for a known procedure (e.g. "Walk me through the lockout procedure", "How do I replace the impeller?").
- **safety_question**: User is asking about risks or whether something is safe (e.g. "Is it safe to work on a pressurized line?").
- **parts_lookup**: User is looking for part numbers or specifications.
- **unknown**: Query is ambiguous, off-topic, or unclear.

Be conservative with confidence. If unsure, use "unknown" or "general_explanation".

Return ONLY valid JSON matching the schema. Do not add extra text.
"""


def get_intent_classifier_llm():
    if MOCK_MODE:
        class MockClassifier:
            def invoke(self, messages):
                class Resp:
                    content = '{"intent": "general_explanation", "confidence": 0.85, "reasoning": "User is asking for a definition/explanation (HAZOP).", "requires_image": false, "suggested_tools": ["retrieve_knowledge"]}'
                return Resp()
        return MockClassifier()
    
    llm = ChatOllama(model=LLM_MODEL, temperature=0.0, num_predict=300)
    return llm.with_structured_output(_IntentOutput)


def classify_intent(user_input: str, has_image: bool = False) -> IntentClassification:
    """
    Production intent classifier.
    Runs early in the graph.
    """
    if not user_input.strip():
        return IntentClassification(
            intent=QueryIntent.UNKNOWN,
            confidence=0.9,
            reasoning="Empty query",
            requires_image=False,
            suggested_tools=[]
        )

    llm = get_intent_classifier_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_SYSTEM_PROMPT),
        ("human", "User query: {query}\nHas image attached: {has_image}")
    ])

    try:
        chain = prompt | llm
        result: _IntentOutput = chain.invoke({
            "query": user_input,
            "has_image": has_image
        })

        # Map to our public schema
        classification = IntentClassification(
            intent=result.intent,
            confidence=result.confidence,
            reasoning=result.reasoning,
            requires_image=result.requires_image or has_image,
            suggested_tools=result.suggested_tools
        )

        logger.info("Intent classified", 
                    intent=classification.intent, 
                    confidence=classification.confidence,
                    query_preview=user_input[:80])

        return classification

    except Exception as e:
        logger.error("Intent classification failed", error=str(e))
        # === ROBUST FALLBACK (works even without Ollama / in MOCK) ===
        # Simple keyword rules so that "HAZOP", "LOTO", "what is", "explain" 
        # correctly go to GENERAL_EXPLANATION instead of forcing troubleshooting.
        q = user_input.lower()
        if any(k in q for k in ["hazop", "what is", "explain", "definition", "loto", "lockout", "sop", "procedure for", "how to do"]):
            return IntentClassification(
                intent=QueryIntent.GENERAL_EXPLANATION,
                confidence=0.75,
                reasoning="Keyword rule: general knowledge / definition query (fallback, no LLM)",
                requires_image=has_image,
                suggested_tools=["retrieve_knowledge"]
            )
        if any(k in q for k in ["vibrat", "leak", "noise", "not working", "fault", "broken", "overheat", "pump", "seal"]):
            return IntentClassification(
                intent=QueryIntent.TROUBLESHOOTING,
                confidence=0.7,
                reasoning="Keyword rule: equipment symptom reported (fallback, no LLM)",
                requires_image=has_image,
                suggested_tools=["retrieve_knowledge", "analyze_image"]
            )
        return IntentClassification(
            intent=QueryIntent.UNKNOWN,
            confidence=0.5,
            reasoning=f"LLM unavailable, no strong keyword match: {str(e)[:60]}",
            requires_image=has_image,
            suggested_tools=["retrieve_knowledge"]
        )