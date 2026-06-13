# Multi-Modal Field Service Assistant (MM-FSA)

**AI-Powered Hands-Free Assistant for Field Service Technicians**

A production-grade, well-architected multi-modal AI system that empowers technicians with:
- **Computer Vision**: Real-time equipment identification, fault detection, and visual inspection using mobile camera.
- **Speech Interface**: Fully hands-free voice interaction (STT + TTS) for queries, confirmations, and guidance.
- **Document Intelligence**: Intelligent retrieval and reasoning over technical manuals, SOPs, wiring diagrams, and historical case data.
- **Agentic Reasoning**: Structured, step-by-step guidance, decision trees, risk assessment, and adaptive troubleshooting using tool-calling agents.

## Key Features

- **Multimodal Fusion**: Combines vision analysis + voice query + retrieved knowledge in a single coherent reasoning loop.
- **Agentic Workflows**: LangGraph-powered state machine for complex, multi-step tasks (e.g., "Diagnose overheating pump → retrieve manual → check past cases → propose repair plan").
- **Safety & Guardrails**: Hallucination detection, compliance checks, confidence scoring, and human-in-the-loop escalation.
- **Context-Aware**: Maintains session memory, equipment history, technician profile, and conversation state.
- **Offline/Edge Capable**: Designed for low-connectivity environments (local Ollama models, quantized, progressive sync).
- **Mobile-First**: Optimized for rugged tablets/phones (camera, mic, voice output, minimal UI).
- **Enterprise Ready**: Observability, audit logs, role-based access, evaluation harness.

## Architecture Overview (High-Level)

```
┌─────────────────────────────────────────────────────────────────┐
│                        MOBILE / FIELD DEVICE                     │
│  (Browser PWA / React Native / Native App)                       │
│  - Camera (live + capture)                                      │
│  - Microphone (push-to-talk / always-listen)                    │
│  - Speaker (TTS)                                                │
│  - Touch / Voice UI                                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │ WebSocket / REST (HTTPS)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BACKEND ORCHESTRATION (FastAPI)            │
│  - Session Manager (Redis + Postgres)                           │
│  - Multimodal Router                                            │
│  - API Gateways: /vision, /voice, /chat, /guidance              │
│  - Streaming Responses (SSE / WS)                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  VISION CORE │ │  SPEECH CORE │ │  RAG / DOC   │ │  AGENTIC     │
│  (VLM + CV)  │ │  (STT/TTS)   │ │  INTELLIGENCE│ │  REASONER    │
│  - Llama-3.2 │ │  - faster-   │ │  (Multimodal │ │  (LangGraph) │
│    Vision    │ │    whisper   │ │   RAG)       │ │  - Planner   │
│  - OpenCV    │ │  - Piper TTS │ │  - ChromaDB  │ │  - Tools     │
│  - Fault     │ │              │ │  - LlamaIndex│ │  - Memory    │
│    Detection │ │              │ │  - Hybrid    │ │  - Guardrails│
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   KNOWLEDGE BASE │
              │ - Manuals (PDFs) │
              │ - SOPs           │
              │ - Diagrams       │
              │ - Past Cases     │
              │ - Equipment DB   │
              └──────────────────┘
```

**Core Technologies (2026 Recommended Open-Source Stack)**

- **LLM / VLM / Embeddings**: Ollama (Llama 3.2 / Qwen2.5-VL / Llama-3.2-Vision / nomic-embed)
- **Agent Framework**: LangGraph (LangChain ecosystem)
- **RAG**: LlamaIndex + ChromaDB (multimodal collections for text + image descriptions)
- **Speech**: faster-whisper (STT) + Piper TTS (local, fast)
- **Backend**: FastAPI + Pydantic + Uvicorn + WebSockets + Redis
- **Vector / Metadata DB**: ChromaDB (primary) + PostgreSQL (structured data, audit)
- **CV Preprocessing**: OpenCV, Pillow
- **Frontend**: Progressive Web App (vanilla JS + Tailwind for prototype; Next.js recommended for prod)
- **Orchestration / Deployment**: Docker Compose (local), Kubernetes (prod)
- **Observability**: Langfuse / OpenTelemetry + Prometheus + LangSmith (optional)
- **Evaluation**: RAGAS + custom safety + human feedback loops

## Project Structure

```
multi-modal-fsa/
├── README.md
├── docs/
│   ├── ARCHITECTURE.md          # Detailed C4, flows, decisions
│   ├── API_SPEC.md
│   ├── USER_STORIES.md
│   ├── EVALUATION.md
│   └── DEPLOYMENT.md
├── ai_core/
│   ├── __init__.py
│   ├── agents/
│   │   ├── field_agent.py       # LangGraph definition
│   │   └── tools/
│   │       ├── vision_tool.py
│   │       ├── rag_tool.py
│   │       ├── safety_tool.py
│   │       └── memory_tool.py
│   ├── rag/
│   │   ├── multimodal_ingest.py
│   │   ├── retriever.py
│   │   └── index_manager.py
│   ├── vision/
│   │   └── analyzer.py
│   ├── speech/
│   │   ├── stt.py
│   │   └── tts.py
│   ├── models/
│   │   └── schemas.py           # Pydantic models
│   └── utils/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   └── api_v1/
│   ├── services/
│   │   └── orchestrator.py
│   └── dependencies/
├── frontend/
│   ├── index.html               # Self-contained PWA demo
│   ├── app.js
│   ├── styles.css
│   └── assets/
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.ollama
│   └── docker-compose.yml
├── data/
│   ├── manuals/                 # PDFs / Markdown (synthetic + real)
│   ├── cases/
│   │   └── past_cases.json
│   └── sample_images/
├── scripts/
│   ├── ingest_data.py
│   └── run_demo.py
├── tests/
│   ├── test_agents.py
│   └── test_rag.py
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Quick Start (Development)

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- Ollama (recommended) or OpenAI-compatible endpoint (for rapid prototyping)
- (Optional) NVIDIA GPU for faster inference

### 1. Clone & Setup Environment

```bash
cd multi-modal-fsa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Local Models (Ollama)

```bash
# Install Ollama if not present: https://ollama.com
ollama serve &
ollama pull llama3.2:3b          # or larger
ollama pull llama3.2-vision:11b  # VLM
ollama pull nomic-embed-text     # embeddings  <--- REQUIRED to avoid OpenAI embedding errors
```

### 3. Ingest Knowledge Base

```bash
python scripts/ingest_data.py
```

### 4. Run Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Launch Frontend Demo

Open `frontend/index.html` in browser (or serve with `python -m http.server` from frontend/).

The demo supports:
- Upload or capture photo from camera
- Voice input (browser Web Speech API)
- Real-time chat + guidance
- Simulated full agent flow (connects to backend)

### 6. Full Stack with Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

Access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs
- Ollama: http://localhost:11434

## Example Use Cases (Technician Scenarios)

1. **Visual Diagnosis**:
   - Technician points camera at industrial pump.
   - "This pump is making a grinding noise and the housing feels hot."
   - System: Identifies "Model X-450 Centrifugal Pump", detects "bearing wear" + "seal leak signs", retrieves relevant SOP section + 3 similar past cases, generates 7-step repair plan with safety warnings and parts list.

2. **Hands-Free Troubleshooting**:
   - Voice: "Walk me through replacing the impeller on this unit."
   - System listens, confirms equipment via vision (if image provided), retrieves exact procedure, delivers step-by-step voice + visual overlays. Technician says "next" or "explain more" or "show diagram".

3. **Complex Decision Support**:
   - After multiple steps: "The new seal isn't fitting. What now?"
   - Agent reasons over history + docs + vision (new photo), proposes alternatives, escalates if high risk.

4. **Post-Job Logging**:
   - Auto-summarizes session, suggests updates to knowledge base, logs structured case.

## Non-Functional Requirements

- **Latency**: < 1.5s for vision analysis, < 800ms for voice response start, < 3s full guidance plan.
- **Accuracy**: > 85% equipment ID, > 90% retrieval relevance (measured via RAGAS), low hallucination via guardrails.
- **Reliability**: Graceful degradation (text-only if vision fails), offline caching of critical manuals.
- **Privacy/Security**: On-device preprocessing where possible, PII redaction, encrypted sessions, audit trail.
- **Scalability**: Stateless agents, horizontal scaling of inference (vLLM), sharded vector DB.

## Roadmap & Extensibility

- Phase 1 (Current): Core agent + multimodal RAG + browser demo.
- Phase 2: Real-time streaming STT/TTS (server-side), live video analysis, AR overlays.
- Phase 3: Multi-agent specialization (e.g., SafetyAgent, PartsAgent, DiagnosticsAgent), fine-tuning on domain data.
- Phase 4: Edge deployment (ONNX/TensorRT on rugged devices), federated learning from field data.
- Integrations: CMMS (e.g., SAP, IBM Maximo), IoT sensors, AR glasses (Xreal, etc.).

## Contributing & Evaluation

See `docs/EVALUATION.md` for benchmarks, test harness, and human evaluation protocol.
See `tests/` for automated tests.

Built with care for real-world field conditions — reliable, fast, and trustworthy.

**Status**: Architecture + Core Prototype (2026-06-13)

---

*This system is designed following modern agentic multimodal patterns (LangGraph + Multimodal RAG + VLM + Voice). See docs/ARCHITECTURE.md for deep dive.*
