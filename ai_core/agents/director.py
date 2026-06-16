"""
Multi-Agent Director (Router) for Industry-Specific SaaS.
Dedicated early node that routes queries by industry + intent.
"""

import os
from typing import Optional
import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from ai_core.models.schemas import Industry, QueryIntent, DirectorClassification, TenantContext

logger = structlog.get_logger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")


class _DirectorOutput(BaseModel):
    primary_industry: Industry = Field(...)
    query_intent: QueryIntent = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(...)
    is_cross_industry: bool = Field(default=False)
    suggested_tools: list[str] = Field(default_factory=list)


def get_director_llm():
    if MOCK_MODE:
        class MockDirector:
            def invoke(self, messages):
                class Resp:
                    content = '{"primary_industry": "construction", "query_intent": "general_explanation", "confidence": 0.82, "reasoning": "HAZOP definition query", "is_cross_industry": false, "suggested_tools": ["retrieve_knowledge"]}'
                return Resp()
        return MockDirector()
    llm = ChatOllama(model=LLM_MODEL, temperature=0.0, num_predict=400)
    return llm.with_structured_output(_DirectorOutput)


DIRECTOR_SYSTEM_PROMPT = "You are the Director for a multi-industry field service AI. Classify the query into primary_industry and query_intent. Detect if it is cross-industry for the tenant's licensed industry. Return only JSON."


def classify_with_director(user_input: str, tenant_context: Optional[TenantContext] = None, has_image: bool = False) -> DirectorClassification:
    tenant_id = tenant_context.tenant_id if tenant_context else "unknown"
    licensed = tenant_context.industry if tenant_context else Industry.UNKNOWN

    if not user_input.strip():
        return DirectorClassification(primary_industry=Industry.UNKNOWN, query_intent=QueryIntent.UNKNOWN, confidence=0.9, reasoning="Empty", is_cross_industry=False)

    llm = get_director_llm()
    prompt = ChatPromptTemplate.from_messages([("system", DIRECTOR_SYSTEM_PROMPT), ("human", "Query: {query}\nTenant licensed: {licensed}")])

    try:
        chain = prompt | llm
        result = chain.invoke({"query": user_input, "licensed": licensed.value})
        is_cross = result.is_cross_industry
        if tenant_context and result.primary_industry != licensed and result.primary_industry not in (Industry.GENERAL_INDUSTRIAL, Industry.UNKNOWN):
            is_cross = True
        return DirectorClassification(
            primary_industry=result.primary_industry,
            query_intent=result.query_intent,
            confidence=result.confidence,
            reasoning=result.reasoning,
            is_cross_industry=is_cross,
            suggested_tools=result.suggested_tools
        )
    except Exception as e:
        logger.error("Director failed", error=str(e))
        q = user_input.lower()
        ind = Industry.GENERAL_INDUSTRIAL
        if any(k in q for k in ["excavat", "crane", "concrete", "scaff"]): ind = Industry.CONSTRUCTION
        if any(k in q for k in ["clarifier", "sludge", "wastewater", "chlorin"]): ind = Industry.WATER_TREATMENT
        intent = QueryIntent.GENERAL_EXPLANATION if any(k in q for k in ["what is", "explain", "hazop", "loto"]) else QueryIntent.TROUBLESHOOTING if any(k in q for k in ["vibrat", "leak", "fault"]) else QueryIntent.UNKNOWN
        is_cross = tenant_context and ind != licensed and ind not in (licensed, Industry.GENERAL_INDUSTRIAL)
        return DirectorClassification(primary_industry=ind, query_intent=intent, confidence=0.6, reasoning="Keyword fallback", is_cross_industry=is_cross)


def build_disclaimer(classification: DirectorClassification, tenant_context: Optional[TenantContext]) -> Optional[str]:
    if not classification.is_cross_industry:
        return None
    tenant_ind = tenant_context.industry.value if tenant_context else "your industry"
    return f"This response uses only general industrial knowledge. Your account is licensed for {tenant_ind}. The topic appears to relate to {classification.primary_industry.value}. For industry-specific procedures, ensure your subscription covers it. Always follow company SOPs and local regulations."