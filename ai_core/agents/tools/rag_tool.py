"""
RAG Tool for the Field Service Agent.
Performs hybrid retrieval over technical manuals, SOPs, and past cases.
Uses LlamaIndex + ChromaDB.

IMPORTANT: Uses the centralized local Ollama embedding configuration
to avoid any accidental OpenAI embedding fallback.
"""
import os
import time
from typing import List, Optional
from langchain_core.tools import tool
from ai_core.models.schemas import RetrievedDocument
import structlog

from ai_core.rag.embeddings import ensure_local_embeddings

logger = structlog.get_logger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

# Lazy loaded globals
_vector_store = None
_retriever = None

def get_retriever():
    """Lazy initialization of LlamaIndex retriever (with local embeddings enforced)."""
    global _vector_store, _retriever
    if _retriever is not None:
        return _retriever

    if MOCK_MODE:
        _retriever = "mock"
        return _retriever

    try:
        from llama_index.core import VectorStoreIndex, StorageContext, Settings
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb

        # === CRITICAL: Ensure we are using local Ollama embeddings, not OpenAI ===
        embed_model = ensure_local_embeddings()

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        vector_store = ChromaVectorStore(chroma_collection=client.get_or_create_collection("fsa_knowledge"))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Explicitly pass the embed model when reconstructing the index
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        _retriever = index.as_retriever(similarity_top_k=8)
        logger.info("RAG retriever initialized from Chroma (local embeddings)")
        return _retriever
    except Exception as e:
        logger.error("Failed to initialize real RAG retriever", error=str(e))
        return "mock"


@tool
def retrieve_knowledge(query: str, equipment_model: Optional[str] = None, top_k: int = 6) -> dict:
    """
    Retrieve relevant technical information from manuals, SOPs, diagrams, and past repair cases.
    
    Use this for any question about procedures, specifications, troubleshooting steps, or similar past issues.
    Always call this tool before giving detailed guidance.
    """
    start = time.time()
    retriever = get_retriever()

    if retriever == "mock" or MOCK_MODE:
        logger.info("Using MOCK RAG retrieval", query=query)
        mock_docs = [
            RetrievedDocument(
                content="Step 1: Isolate power using LOTO procedure per SOP-EL-003. Verify zero energy state with multimeter before any work on rotating equipment.",
                metadata={"source": "SOP-EL-003", "section": "Lockout/Tagout", "model": "X-450"},
                score=0.94,
                source="sop",
                page_or_section="3.1"
            ),
            RetrievedDocument(
                content="For mechanical seal replacement on X-450 pump: Drain fluid, remove coupling guard, loosen set screws on seal collar. Use puller tool P-17. New seal part number SEAL-X450-22. Torque to 18 Nm.",
                metadata={"source": "MAN-X450-Rev4.pdf", "section": "5.2 Seal Replacement", "model": "X-450"},
                score=0.91,
                source="manual",
                page_or_section="42"
            ),
            RetrievedDocument(
                content="Case #2025-0842: Similar vibration + seal leak on Pump X-450 at Site B. Root cause was worn impeller shaft. Replaced shaft and seal. Resolved after 4.5 hours. Technician noted: 'Always check runout before reassembly.'",
                metadata={"source": "past_cases", "case_id": "2025-0842", "model": "X-450"},
                score=0.88,
                source="case",
                page_or_section=None
            )
        ]
        latency = int((time.time() - start) * 1000)
        return {
            "success": True,
            "data": [d.model_dump() for d in mock_docs[:top_k]],
            "error": None,
            "tool_name": "retrieve_knowledge",
            "latency_ms": latency
        }

    try:
        # Real LlamaIndex retrieval
        nodes = retriever.retrieve(query)
        
        docs = []
        for node in nodes[:top_k]:
            meta = node.metadata or {}
            docs.append(RetrievedDocument(
                content=node.get_content(),
                metadata=meta,
                score=getattr(node, 'score', 0.8),
                source=meta.get("source_type", "manual"),
                page_or_section=meta.get("page") or meta.get("section")
            ).model_dump())

        latency = int((time.time() - start) * 1000)
        return {
            "success": True,
            "data": docs,
            "error": None,
            "tool_name": "retrieve_knowledge",
            "latency_ms": latency
        }

    except Exception as e:
        logger.error("RAG retrieval failed", error=str(e))
        latency = int((time.time() - start) * 1000)
        return {
            "success": False,
            "data": [],
            "error": str(e),
            "tool_name": "retrieve_knowledge",
            "latency_ms": latency
        }
