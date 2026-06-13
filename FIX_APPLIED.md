# Fix Applied: OpenAI Embedding Error During Ingestion

**Date**: 2026-06-13  
**Problem**: `ingest_knowledge_base()` was failing with  
`ValueError: Could not load OpenAI embedding model. ... No API key found for OpenAI.`

**Root Cause**: LlamaIndex's default `Settings.embed_model` resolves to OpenAI when no explicit local embedder is configured. The code was not forcing a local Ollama embedding model.

## Changes Made

1. **requirements.txt** — Added:
   ```toml
   llama-index-embeddings-ollama>=0.3.0
   ```

2. **New file**: `ai_core/rag/embeddings.py`
   - Centralized, robust function `get_ollama_embedding_model()` + `ensure_local_embeddings()`
   - Explicitly creates `OllamaEmbedding` and sets it in `llama_index.core.Settings`
   - Clear error message telling the user exactly what to do (`ollama pull nomic-embed-text`)

3. **ai_core/rag/index_manager.py** (ingestion)
   - Now calls `ensure_local_embeddings()` before creating `VectorStoreIndex`
   - Removed duplicate direct OllamaEmbedding code (now uses the shared helper)

4. **ai_core/agents/tools/rag_tool.py** (runtime retrieval)
   - Now also calls `ensure_local_embeddings()` before loading the retriever
   - Explicitly passes the embed model when reconstructing the index from Chroma

5. **scripts/ingest_data.py**
   - Imports and calls `ensure_local_embeddings()` early (when not in MOCK_MODE)
   - Better user messaging

6. **.env.example** — Clarified the `EMBED_MODEL` line.

## How to Apply the Fix

```bash
# 1. Install the new dependency
cd multi-modal-fsa
pip install -r requirements.txt

# 2. Make sure Ollama is running and has the embedding model
ollama serve &          # if not already running
ollama pull nomic-embed-text

# 3. (Recommended) Set MOCK_MODE=false in your environment or .env
#    (or keep it true for instant demo)

# 4. Re-run ingestion
python scripts/ingest_data.py
```

You should now see:
```
Using local Ollama embeddings: nomic-embed-text @ http://localhost:11434
Built vector index with X documents
```

## Verification

After successful ingestion, you can test retrieval in the agent by running a query in the frontend or directly:

```python
from ai_core.agents.tools.rag_tool import retrieve_knowledge
result = retrieve_knowledge.invoke({"query": "mechanical seal replacement on X-450"})
print(result)
```

## Notes

- The entire RAG layer is now **guaranteed** to use local embeddings.
- The same protection was added to both ingestion and runtime retrieval paths.
- If you ever want to switch embedding providers, the single source of truth is now `ai_core/rag/embeddings.py`.
- `MOCK_MODE=true` still bypasses everything for fast demos.

The system is now robust against the "accidental OpenAI default" problem that LlamaIndex has by default.
