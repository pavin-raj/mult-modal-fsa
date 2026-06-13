#!/usr/bin/env python3
"""
Knowledge Base Ingestion Script for Multi-Modal Field Service Assistant.

Ingests:
- Technical manuals (PDFs or Markdown)
- SOPs
- Past case histories (JSON)
- Diagrams (extracts images + describes them)

Populates ChromaDB vector store + metadata.
Run this after setting up the environment and before starting the backend.
"""
import os
import json
import sys
from pathlib import Path
import structlog

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ai_core.rag.index_manager import ingest_knowledge_base
from ai_core.rag.embeddings import ensure_local_embeddings  # ensures we never use OpenAI embeddings

logger = structlog.get_logger(__name__)

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./data/manuals")
CASES_FILE = os.getenv("CASES_FILE", "./data/cases/past_cases.json")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

def main():
    print("=" * 60)
    print("Multi-Modal Field Service Assistant - Knowledge Ingestion")
    print("=" * 60)
    
    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CASES_FILE), exist_ok=True)
    
    # Create sample data if it doesn't exist
    _ensure_sample_data()
    
    print(f"\nIngesting from: {KNOWLEDGE_BASE_DIR}")
    print(f"Cases file: {CASES_FILE}")
    print(f"Vector store: {CHROMA_DIR}")
    
    # Ensure local embeddings are configured before any LlamaIndex usage
    if not os.getenv("MOCK_MODE", "false").lower() == "true":
        try:
            ensure_local_embeddings()
        except Exception as e:
            print(f"\n⚠️  Embedding setup warning: {e}")
            print("   You can still proceed with MOCK_MODE=true, or fix your Ollama setup.")
    
    stats = ingest_knowledge_base(
        manuals_dir=KNOWLEDGE_BASE_DIR,
        cases_file=CASES_FILE,
        persist_dir=CHROMA_DIR
    )
    
    print("\n✅ Ingestion complete!")
    print(f"   Documents indexed: {stats.get('documents', 0)}")
    print(f"   Cases indexed: {stats.get('cases', 0)}")
    print(f"   Images/diagrams processed: {stats.get('images', 0)}")
    print(f"\nYou can now start the backend and demo.")

def _ensure_sample_data():
    """Create realistic sample manuals and cases if they don't exist."""
    manuals_dir = Path(KNOWLEDGE_BASE_DIR)
    cases_path = Path(CASES_FILE)
    
    # Sample Manual (Markdown for simplicity - in real life use PDFs)
    pump_manual = manuals_dir / "X-450-Centrifugal-Pump-Manual.md"
    if not pump_manual.exists():
        pump_manual.write_text("""# X-450 Centrifugal Pump Technical Manual

## 1. Specifications
- Model: X-450
- Max Flow: 450 GPM
- Max Head: 180 ft
- Motor: 25 HP, 3-phase
- Seal Type: Mechanical (Type 21)

## 2. Safety
**WARNING**: Always follow Lockout/Tagout (LOTO) procedures before any maintenance.
Required PPE: Safety glasses, gloves, steel-toe boots, hearing protection.

## 3. Troubleshooting

### 3.1 Excessive Vibration
Possible causes:
- Misaligned coupling
- Worn bearings
- Cavitation
- Impeller damage

Recommended actions:
1. Check alignment with dial indicator.
2. Inspect bearings for roughness.
3. Verify NPSH and suction conditions.

### 3.2 Seal Leakage
**Critical**: Stop pump immediately if leakage exceeds 5 drops per minute.

Procedure:
1. Isolate power and verify zero energy.
2. Drain system.
3. Remove coupling guard.
4. Loosen seal set screws.
5. Use seal puller tool P-17.
6. Clean shaft and install new seal (part SEAL-X450-22).
7. Torque collar to 18 Nm.
8. Reassemble and perform run test.

## 4. Parts List
- SEAL-X450-22 : Mechanical Seal Assembly
- BEARING-6205 : Drive End Bearing
- IMPELLER-X450 : Replacement Impeller

## 5. Diagrams
See Figure 5.2 for seal assembly cross-section.
""")
        print(f"Created sample manual: {pump_manual}")

    # Sample SOP
    sop_file = manuals_dir / "SOP-EL-003-Lockout-Tagout.md"
    if not sop_file.exists():
        sop_file.write_text("""# SOP-EL-003: Electrical Lockout/Tagout

## Purpose
Ensure zero energy state before work on energized equipment.

## Steps
1. Notify affected personnel.
2. Identify all energy sources.
3. Shut down equipment using normal procedures.
4. Isolate energy sources (open disconnects, close valves).
5. Apply personal locks and tags.
6. Verify zero energy state (use calibrated test equipment).
7. Perform work.
8. Remove tools, verify area clear.
9. Remove locks/tags only by authorized person who applied them.
""")
        print(f"Created sample SOP: {sop_file}")

    # Sample Past Cases
    if not cases_path.exists():
        cases = [
            {
                "case_id": "2025-0842",
                "date": "2025-11-12",
                "equipment_model": "X-450 Centrifugal Pump",
                "problem": "Vibration + visible seal leak from bottom of housing",
                "root_cause": "Worn impeller shaft causing seal face damage",
                "resolution": "Replaced shaft, seal, and bearings. Performed full alignment.",
                "outcome": "Resolved - unit running normally",
                "technician_notes": "Always check shaft runout before installing new seal. Took 4.5 hours total.",
                "duration_hours": 4.5
            },
            {
                "case_id": "2026-0317",
                "date": "2026-03-05",
                "equipment_model": "X-450 Centrifugal Pump",
                "problem": "Grinding noise and overheating at motor end",
                "root_cause": "Drive end bearing failure due to lack of lubrication",
                "resolution": "Replaced both bearings, cleaned housing, added grease fittings.",
                "outcome": "Resolved",
                "technician_notes": "Customer had skipped the last two scheduled services.",
                "duration_hours": 3.0
            }
        ]
        cases_path.write_text(json.dumps(cases, indent=2))
        print(f"Created sample cases: {cases_path}")

if __name__ == "__main__":
    main()
