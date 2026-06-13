# Multi-Modal Field Service Assistant — Detailed Architecture

**Version**: 1.0  
**Date**: 2026-06-13  
**Status**: Core Design

## 1. System Context (C4 Level 1)

### Users
- **Primary**: Field Service Technicians (hands-busy, rugged environments, variable connectivity).
- **Secondary**: Supervisors, Knowledge Engineers, Safety Officers.

### External Systems
- CMMS / ERP (work orders, parts inventory, asset history) — future integration.
- IoT / SCADA sensors (real-time telemetry).
- Corporate Document Management / SharePoint / Confluence.
- Mobile device hardware (camera, mic, speakers, GPS).

### Goals
- Reduce mean-time-to-repair (MTTR) by 30-50%.
- Improve first-time-fix rate.
- Preserve tribal knowledge.
- Enable less-experienced technicians to perform complex tasks safely.
- Operate reliably in low-bandwidth / offline conditions.

## 2. Container View (C4 Level 2)

### Major Containers

1. **Mobile Client** (PWA / React Native)
   - Responsibilities: Capture multimodal input (photo + voice + text), render guidance, manage local state, offline caching.
   - Tech: HTML5 Camera API + Web Speech API (prototype), Capacitor or React Native (prod).

2. **API Gateway / Orchestrator** (FastAPI)
   - Responsibilities: Authentication, session management, routing to modality-specific services, streaming, rate limiting.
   - Tech: FastAPI, Redis (sessions), PostgreSQL (audit, users, structured data).

3. **Multimodal Agent Runtime** (LangGraph + Python)
   - Responsibilities: State machine orchestration, tool calling, reasoning, planning, memory.
   - Core: `FieldServiceAgent` graph with nodes for Planning, Tool Execution, Reflection, Safety Check.

4. **Vision Service**
   - Responsibilities: Image preprocessing, VLM inference (equipment ID, anomaly detection, OCR on labels), visual grounding.
   - Tech: Ollama (Llama-3.2-Vision or Qwen2.5-VL), OpenCV, Pillow. Optional dedicated CV model (YOLO for specific equipment families).

5. **Speech Service**
   - Responsibilities: Streaming STT (low-latency), Voice Activity Detection (VAD), TTS synthesis (natural, interruptible).
   - Tech: faster-whisper (server), browser Web Speech (fallback), Piper / Coqui TTS.

6. **Knowledge & Retrieval Service (Multimodal RAG)**
   - Responsibilities: Ingestion pipeline, hybrid search (vector + keyword + metadata), multimodal retrieval (text chunks + image embeddings/descriptions), reranking.
   - Tech: LlamaIndex, ChromaDB (or PGVector), embedding model (nomic-embed-text or multimodal embedder).

7. **Model Serving Layer**
   - Local-first: Ollama.
   - Production: vLLM / TGI / Ollama cluster, or managed (Bedrock, Vertex, etc.).
   - Caching: Redis for prompts/responses, model quantization (4-bit/8-bit).

8. **Data & Persistence Layer**
   - Vector: ChromaDB (knowledge + cases).
   - Relational: PostgreSQL (sessions, audit logs, equipment master, user profiles).
   - Object Storage: MinIO / S3 for raw images, PDFs, audio recordings (with retention policy).
   - Time-series (future): For sensor data.

## 3. Component View — Core Agentic Loop (C4 Level 3)

### The Central State Machine (LangGraph)

The heart of the system is a **LangGraph StateGraph** called `FieldServiceWorkflow`.

**State Schema** (Pydantic):
```python
class FSAState(TypedDict):
    session_id: str
    technician_id: str
    equipment_context: dict          # {id, model, location, history}
    current_image: bytes | None
    current_transcript: str
    conversation_history: list[Message]
    retrieved_knowledge: list[Document]
    vision_analysis: VisionResult | None
    plan: list[Step] | None
    current_step_index: int
    safety_flags: list[str]
    confidence: float
    tools_used: list[str]
    final_guidance: str | None
    needs_escalation: bool
```

**Graph Nodes** (in execution order, with conditional edges):

1. **Input Fusion Node**
   - Merges latest camera frame + voice transcript + text.
   - Runs lightweight VLM if image present (even before full agent).

2. **Context Loader**
   - Loads persistent equipment history, technician profile, previous session summaries.

3. **Vision Analysis Tool** (parallel)
   - If image: Equipment identification, fault/anomaly detection, OCR on nameplates, visual grounding of parts.

4. **Retrieval Tool** (hybrid)
   - Semantic search + keyword + metadata filters (equipment model, fault type).
   - Multimodal: retrieves both text sections and previously associated images/diagrams.

5. **Planner Node** (LLM)
   - Given fused input + vision + retrieved docs, produces:
     - Structured diagnosis hypothesis
     - Step-by-step plan (with prerequisites, risks, verification criteria)
     - Required tools/parts

6. **Tool Executor** (subgraph)
   - Can call:
     - `rag_tool(query)` — deeper retrieval
     - `vision_tool(follow_up_image)` — zoom on specific component
     - `safety_tool(plan)` — risk scoring
     - `memory_tool` — store new observation
     - `parts_lookup` (future)

7. **Reflection / Self-Critique Node**
   - LLM reviews plan against retrieved evidence.
   - Scores confidence, detects potential hallucinations.
   - Decides: Proceed, Gather more info, Escalate.

8. **Safety & Compliance Gate**
   - Hard rules (lockout/tagout, PPE, pressure > X → escalate).
   - LLM guardrail (via separate small model or same with system prompt).

9. **Response Synthesizer**
   - Generates:
     - Voice-friendly concise narration
     - Structured checklist / numbered steps
     - Visual highlights (bounding boxes if vision model supports)
     - Citations to manuals/cases

10. **Memory Updater** (parallel)
    - Persists structured case log, updates equipment profile if new facts discovered.

**Edges & Control Flow**:
- Conditional routing based on `state["needs_more_info"]`, `state["confidence"] < threshold`, `state["safety_flags"]`.
- Human-in-the-loop checkpoints (technician confirms "yes" or provides correction via voice).
- Loop back for clarification ("show me a photo of the impeller" → vision tool).

### Tool Definitions (LangChain Tools)

```python
@tool
def analyze_image(image_base64: str, focus: str = "general") -> VisionResult:
    """Identify equipment, detect visible faults, describe condition."""

@tool
def retrieve_knowledge(query: str, equipment_model: str | None = None, top_k: int = 8) -> list[Document]:
    """Hybrid retrieval from technical docs and past cases."""

@tool
def assess_safety(plan_steps: list[str], equipment_type: str) -> SafetyAssessment:
    """Check for high-risk actions and required mitigations."""

@tool
def get_equipment_history(equipment_id: str) -> list[CaseSummary]:
    ...

@tool
def store_case_observation(observation: dict) -> str:
    """Log new finding for future retrieval."""
```

## 4. Data Flow — Typical Multimodal Interaction

1. Technician opens app → starts session (loads last known equipment if GPS or QR scan).
2. Points camera at asset + speaks: "This unit is vibrating badly and leaking from the bottom seal."
3. Client sends:
   - JPEG (or WebP) of current view (periodic or on-demand)
   - Streaming audio chunks (or full utterance)
   - Text fallback
4. Backend:
   - STT → transcript
   - Vision Service → structured JSON + description + confidence
   - Router decides "full agent run" vs quick answer
5. Agent graph executes (can be 3-12 LLM calls depending on complexity).
6. Streaming response back:
   - Immediate: "Identified as Pump Model X-450. Detected seal leak and possible bearing issue."
   - Then: "Step 1: Isolate power and lockout. Confirm with photo."
   - Voice synthesis starts playing while text renders.
7. Technician responds with voice or new photo → loop.

**Streaming Strategy**:
- Use Server-Sent Events (SSE) or WebSocket.
- First token latency prioritized.
- Partial plans and confidence shown early.

## 5. Multimodal RAG Design (Critical Component)

### Ingestion Pipeline (`ai_core/rag/multimodal_ingest.py`)

1. **Document Processing**:
   - PDF → LlamaParse or unstructured.io (tables, diagrams preserved).
   - Each page/section chunked with overlap.
   - Metadata: `{"doc_type": "manual", "model": "X-450", "section": "5.2", "page": 42, "has_diagram": true}`

2. **Image / Diagram Handling**:
   - Extract figures from PDFs.
   - Run VLM on each figure → rich textual description + tags.
   - Store:
     - Text chunk (with image description embedded)
     - Separate image collection in Chroma with CLIP-like or Llama-3.2 visual embeddings (if available)
   - Or: Store only text + pointer to image, and use vision model at query time for reranking.

3. **Past Cases**:
   - Structured JSON + free-text summary.
   - Vectorized on "problem + root cause + resolution".

4. **Hybrid Search**:
   - Vector similarity (nomic-embed or multimodal)
   - BM25 keyword
   - Metadata filtering (equipment model, fault category)
   - Reranker (Cohere or cross-encoder or small LLM)

5. **Multimodal Retrieval at Query Time**:
   - If current image present → embed image (or describe it) and search image collection.
   - Fuse with text retrieval using Reciprocal Rank Fusion or learned fusion.

### Knowledge Base Schema (example)

- `manuals/`: 50-200 PDFs per equipment family
- `sops/`
- `diagrams/`
- `cases/`: 1000+ historical tickets (anonymized)

## 6. Safety, Reliability & Guardrails

- **Layered Defense**:
  1. System prompt + few-shot with "never skip safety steps".
  2. Dedicated `safety_tool` that must pass before final output.
  3. Post-generation hallucination check (LLM-as-judge on "is every claim supported by retrieved docs?").
  4. Confidence threshold → "I'm not sure — recommend consulting supervisor".
  5. Hard-coded rules engine for critical procedures (LOTO, high voltage).
  6. Full audit trail: every agent decision, tool input/output, final response logged with trace ID.

- **Human-in-the-Loop**:
  - Explicit confirmation points ("Do you want to proceed with removing the cover?").
  - Easy "escalate to expert" button.

## 7. Deployment & Runtime Considerations (2026)

- **Local / Edge**:
  - Ollama + quantized models (Q4_K_M or GGUF).
  - Chroma persistent on device or small server.
  - Progressive download of manuals for specific job.

- **Cloud / On-Prem**:
  - vLLM for high throughput.
  - Separate GPU pools: one for VLM (vision heavy), one for reasoning.
  - Horizontal scaling of agent workers (stateless except Redis state).

- **Connectivity Modes**:
  - Full online
  - Degraded (cached knowledge + local models)
  - Offline (pre-cached critical procedures + simple rule-based fallback)

- **Observability**:
  - Every LLM call traced (Langfuse or OpenTelemetry + custom spans for vision/rag).
  - Metrics: latency per node, retrieval precision, safety pass rate, user correction rate.
  - Evaluation harness using RAGAS + domain-specific rubrics.

## 8. Technology Choices Rationale (2026)

| Layer              | Choice                        | Why |
|--------------------|-------------------------------|-----|
| Core Orchestration | LangGraph                     | Mature agentic workflows, excellent state management, debugging |
| LLM Reasoning      | Llama 3.2 / Qwen2.5 / Nemotron 3 open models via Ollama/vLLM | Strong reasoning, tool use, vision variants, permissive license |
| Vision             | Llama-3.2-Vision / Qwen2.5-VL | Native multimodal, good at technical diagrams & real-world equipment |
| Embeddings         | nomic-embed-text + visual     | Excellent quality + speed for RAG |
| Vector DB          | ChromaDB (dev) / PGVector (prod) | Easy local + production-grade |
| Speech             | faster-whisper + Piper        | Fast, accurate, fully local, low resource |
| Backend            | FastAPI                       | Async, great docs, streaming, Python ecosystem |
| Frontend           | Progressive Web App           | No install friction, works on rugged tablets |
| Packaging          | Docker + Docker Compose       | Reproducible, easy to run full stack locally |

**Alternatives Considered**:
- Pure cloud (Bedrock Nova + Bedrock Data Automation): Excellent but not offline-friendly.
- LlamaIndex Workflows instead of LangGraph: Good, but LangGraph currently leads for complex agent control.
- Dedicated CV models (YOLO + CLIP): Use as preprocessing / specialist tools alongside VLM.

## 9. Extensibility Points

- Add new tools easily in `ai_core/agents/tools/`
- Plug different model providers via config (Ollama, OpenAI, Anthropic, vLLM)
- Domain-specific agents (e.g., ElectricalAgent, HydraulicAgent) that inherit base graph
- AR / Glasses integration via same multimodal API
- Continuous learning: feedback loop from technician corrections → fine-tuning dataset

## 10. Open Questions & Future Work

- Best way to ground VLM outputs to physical coordinates (for AR overlays).
- Long-term memory architecture across thousands of jobs.
- Cost-efficient vision inference at the edge.
- Regulatory compliance for safety-critical guidance (auditability is already strong).

This architecture is deliberately **modular**, **observable**, and **agent-first**. It treats vision, voice, documents, and reasoning as first-class citizens that feed a central intelligent orchestrator.

Next: Implementation of the core components.
