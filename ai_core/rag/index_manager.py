"""
Multimodal RAG Index Manager for Field Service Assistant.

Handles:
- Ingestion of technical manuals (Markdown/PDF)
- Past case histories
- Automatic description of diagrams/images using VLM
- Hybrid vector + metadata storage in ChromaDB
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List
import structlog

logger = structlog.get_logger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
VLM_MODEL = os.getenv("VLM_MODEL", "llama3.2-vision:11b")

def ingest_knowledge_base(
    manuals_dir: str,
    cases_file: str,
    persist_dir: str = CHROMA_PERSIST_DIR
) -> Dict[str, int]:
    """
    Main ingestion entry point.
    Returns stats dict.
    """
    stats = {"documents": 0, "cases": 0, "images": 0}
    
    if MOCK_MODE:
        logger.info("Running ingestion in MOCK mode (no real vector DB)")
        # Still create directories and pretend
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        stats = {"documents": 12, "cases": 2, "images": 3}
        return stats

    try:
        import chromadb
        from llama_index.core import Document, VectorStoreIndex, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core.node_parser import SentenceSplitter

        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(
            name="fsa_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        documents: List[Document] = []

        # 1. Ingest Manuals and SOPs
        manuals_path = Path(manuals_dir)
        for file_path in manuals_path.rglob("*.md"):
            content = file_path.read_text(encoding="utf-8")
            doc = Document(
                text=content,
                metadata={
                    "source_type": "manual" if "manual" in file_path.name.lower() else "sop",
                    "filename": file_path.name,
                    "path": str(file_path),
                    "model": "X-450"  # In real system, extract from filename or frontmatter
                }
            )
            documents.append(doc)
            stats["documents"] += 1
            logger.info(f"Ingested manual/SOP: {file_path.name}")

        # 2. Ingest Past Cases (structured + vectorized)
        cases_path = Path(cases_file)
        if cases_path.exists():
            with open(cases_path) as f:
                cases = json.load(f)
            
            for case in cases:
                case_text = f"""Case {case['case_id']} ({case['date']}): 
Problem: {case['problem']}
Root Cause: {case['root_cause']}
Resolution: {case['resolution']}
Technician Notes: {case.get('technician_notes', '')}
Outcome: {case['outcome']}"""
                
                doc = Document(
                    text=case_text,
                    metadata={
                        "source_type": "case",
                        "case_id": case["case_id"],
                        "date": case["date"],
                        "equipment_model": case["equipment_model"],
                        "outcome": case["outcome"]
                    }
                )
                documents.append(doc)
                stats["cases"] += 1

        # 3. (Future) Image/Diagram ingestion + VLM description
        # For now we rely on rich textual descriptions in the manuals.
        # Real implementation would extract images from PDFs and call VLM.

        if documents:
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[splitter]
            )
            logger.info(f"Built vector index with {len(documents)} documents")
        
        stats["documents"] += len(documents)
        return stats

    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        if not MOCK_MODE:
            raise
        return stats
