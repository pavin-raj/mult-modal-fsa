# Getting Started with Multi-Modal Field Service Assistant (MM-FSA)

## Prerequisites (2026)

- Python 3.11+
- Docker + Docker Compose (recommended)
- Ollama (highly recommended for fully local experience)
- Modern browser with camera + microphone access

## Step-by-Step Setup

### 1. Install Ollama and pull models (Strongly Recommended)

```bash
# Install Ollama: https://ollama.com/download
ollama serve &

# Pull recommended models
ollama pull llama3.2:3b
ollama pull llama3.2-vision:11b
ollama pull nomic-embed-text
```

### 2. Set up Python environment

```bash
cd multi-modal-fsa
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ingest the Knowledge Base

```bash
python scripts/ingest_data.py
```

This populates ChromaDB with sample manuals + historical cases.

### 4. Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API will be available at:
- http://localhost:8000
- Interactive docs: http://localhost:8000/docs

### 5. Open the Frontend Demo

Simply open `frontend/index.html` in your browser (double-click or drag into browser).

Or serve it properly:
```bash
python -m http.server 8080 --directory frontend
```

Then visit http://localhost:8080

### 6. (Optional) Full Stack with Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- Ollama: http://localhost:11434

## How to Use the Demo

1. **Start Camera** → Point at equipment (or use a photo of a pump/industrial machine).
2. **Capture Photo** or **Upload** an image.
3. **Speak** (click "Hold to Speak") or type:
   - "This pump is vibrating badly and leaking from the bottom seal"
   - "Walk me through replacing the impeller"
   - "Is this safe to work on?"
4. Watch the **Agentic Guidance** panel populate with:
   - Equipment identification + fault detection (from Vision)
   - Structured step-by-step plan
   - Safety warnings
   - Citations from manuals & past cases
   - Confidence score

5. Click **"Confirm & Continue"** or speak "next step" to simulate multi-turn interaction.

## Enabling Real Models (vs Mock)

The system runs in **MOCK_MODE** by default for instant demo.

To use real local models:

Edit `.env` (copy from `.env.example`):

```env
MOCK_MODE=false
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2:3b
VLM_MODEL=llama3.2-vision:11b
```

Restart the backend.

## Next Steps After Demo

- Replace sample data in `data/manuals/` and `data/cases/` with your real technical content.
- Extend tools in `ai_core/agents/tools/`.
- Add real backend persistence (Postgres + Redis).
- Integrate with your CMMS.
- Deploy using the Docker setup to rugged tablets or edge servers.

## Troubleshooting

- **"Connection refused" on frontend**: Make sure backend is running on port 8000.
- **No vision results**: Start with `MOCK_MODE=true` first. Then ensure Ollama is running with the vision model.
- **Speech not working**: The frontend uses browser Web Speech API. For server-side STT/TTS, the backend services are ready.
- **Slow responses**: Use smaller models (`llama3.2:3b`) or enable GPU in Ollama.

You now have a production-grade foundation for a real Multi-Modal Field Service Assistant.
