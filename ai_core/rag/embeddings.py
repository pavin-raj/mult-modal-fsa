"""
Centralized embedding configuration for the Multi-Modal FSA.

This module ensures we ALWAYS use local Ollama embeddings and never fall back
to OpenAI.
"""
import os
import structlog
from llama_index.core import Settings

logger = structlog.get_logger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def get_ollama_embedding_model():
    """
    Returns a properly configured OllamaEmbedding instance and sets it globally
    in LlamaIndex Settings so that both ingestion and retrieval use local embeddings.
    """
    try:
        from llama_index.embeddings.ollama import OllamaEmbedding

        embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
            # You can add more kwargs if needed, e.g. request_timeout=120
        )

        # Set globally so LlamaIndex uses it everywhere (ingestion + retrieval)
        Settings.embed_model = embed_model

        logger.info(f"Local Ollama embeddings configured: model={EMBED_MODEL}, base_url={OLLAMA_BASE_URL}")
        return embed_model

    except Exception as e:
        logger.error("Failed to initialize OllamaEmbedding", error=str(e))
        raise RuntimeError(
            f"Could not load local embedding model '{EMBED_MODEL}' from Ollama at {OLLAMA_BASE_URL}.\n"
            "Please ensure:\n"
            "  1. Ollama is running (`ollama serve`)\n"
            "  2. The embedding model is pulled: `ollama pull nomic-embed-text`\n"
            "  3. You are not accidentally using an OpenAI embedding config.\n"
            "Alternatively, set MOCK_MODE=true in your .env for quick demos."
        ) from e


def ensure_local_embeddings():
    """Idempotent helper — call this early in any script that uses LlamaIndex RAG."""
    if getattr(Settings, "embed_model", None) is None or "ollama" not in str(type(Settings.embed_model)).lower():
        get_ollama_embedding_model()
    return Settings.embed_model
