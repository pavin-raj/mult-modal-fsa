"""
Improved RAG Index Manager with support for live document uploads.

This version supports:
- Full re-ingestion at startup
- Adding single documents dynamically (for the "Upload Document" feature)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import structlog

from ai_core.rag.embeddings import ensure_local_embeddings

logger = structlog.get_logger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")


def ingest_knowledge_base(
    manuals_dir: str,
    cases_file: str,
    persist_dir: str = CHROMA_PERSIST_DIR
) -> Dict[str, int]:
    """Full re-ingestion (used at startup or when you want a clean rebuild)."""
    stats = {"documents": 0, "cases": 0, "images": 0}

    if MOCK_MODE:
        logger.info("Running ingestion in MOCK mode")
        return {"documents": 20, "cases": 5, "images": 0}

    try:
        import chromadb
        from llama_index.core import Document, VectorStoreIndex, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.readers.file import PDFReader

        ensure_local_embeddings()

        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(
            name="fsa_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        documents: List[Document] = []

        # === Manuals, Markdown, Text, and PDFs ===
        manuals_path = Path(manuals_dir)
        for file_path in manuals_path.rglob("*"):
            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()
            try:
                if suffix in [".md", ".txt"]:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    documents.append(Document(
                        text=content,
                        metadata={
                            "source_type": "manual",
                            "filename": file_path.name,
                            "path": str(file_path)
                        }
                    ))
                    stats["documents"] += 1
                    logger.info(f"Ingested text file: {file_path.name}")

                elif suffix == ".pdf":
                    reader = PDFReader()
                    pdf_docs = reader.load_data(str(file_path))
                    for d in pdf_docs:
                        documents.append(Document(
                            text=d.text,
                            metadata={
                                "source_type": "manual",
                                "filename": file_path.name,
                                "path": str(file_path)
                            }
                        ))
                    stats["documents"] += 1
                    logger.info(f"Ingested PDF: {file_path.name}")

            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

        # === Past Cases (JSON) ===
        cases_path = Path(cases_file)
        if cases_path.exists():
            with open(cases_path) as f:
                cases = json.load(f)

            for case in cases:
                case_text = f"""Case {case.get('case_id', 'unknown')} ({case.get('date', '')}): 
Problem: {case.get('problem', '')}
Root Cause: {case.get('root_cause', '')}
Resolution: {case.get('resolution', '')}
Technician Notes: {case.get('technician_notes', '')}
Outcome: {case.get('outcome', '')}"""

                documents.append(Document(
                    text=case_text,
                    metadata={
                        "source_type": "case",
                        "case_id": case.get("case_id"),
                        "equipment_model": case.get("equipment_model"),
                        "date": case.get("date")
                    }
                ))
                stats["cases"] += 1

        if documents:
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[splitter]
            )
            logger.info(f"Built/rebuilt vector index with {len(documents)} total chunks")

        return stats

    except Exception as e:
        logger.error("Full ingestion failed", error=str(e))
        if not MOCK_MODE:
            raise
        return stats


def ingest_single_document(
    file_path: str,
    original_filename: str,
    persist_dir: str = CHROMA_PERSIST_DIR
) -> bool:
    """
    Add ONE document (PDF, MD or TXT) to the existing ChromaDB.
    Uses pypdf directly for PDFs (more reliable + fewer dependencies).
    Falls back to llama-index PDF reader only if pypdf fails.
    """
    if MOCK_MODE:
        logger.info(f"[MOCK] Would add: {original_filename}")
        return True

    try:
        import chromadb
        from llama_index.core import Document, VectorStoreIndex, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core.node_parser import SentenceSplitter

        ensure_local_embeddings()

        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(name="fsa_knowledge")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        path = Path(file_path)
        documents = []
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            # Preferred: pypdf (light, fast, reliable)
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                if text.strip():
                    documents.append(Document(
                        text=text,
                        metadata={"source_type": "manual", "filename": original_filename}
                    ))
            except Exception as pdf_err:
                logger.warning(f"pypdf failed for {original_filename}, trying fallback: {pdf_err}")
                # Fallback: llama-index reader
                try:
                    from llama_index.readers.file import PDFReader
                    reader = PDFReader()
                    pdf_docs = reader.load_data(str(path))
                    for d in pdf_docs:
                        documents.append(Document(
                            text=d.text,
                            metadata={"source_type": "manual", "filename": original_filename}
                        ))
                except ImportError:
                    logger.error("Neither pypdf nor llama-index-readers-file is available")
                    return False

        else:
            # Markdown / TXT
            content = path.read_text(encoding="utf-8", errors="ignore")
            documents.append(Document(
                text=content,
                metadata={"source_type": "manual", "filename": original_filename}
            ))

        if not documents:
            logger.warning(f"No extractable text in {original_filename}")
            return False

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        nodes = splitter.get_nodes_from_documents(documents)

        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        index.insert_nodes(nodes)

        logger.info(f"✅ Added {original_filename} ({len(nodes)} chunks)")
        return True

    except Exception as e:
        logger.error(f"Failed to ingest {original_filename}", error=str(e))
        return False