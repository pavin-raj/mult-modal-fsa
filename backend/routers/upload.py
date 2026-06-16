"""
Upload router for live document ingestion into the knowledge base.
Supports PDF, MD, TXT added dynamically without full re-ingest.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import structlog

from ai_core.rag.index_manager import ingest_single_document

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/document")
async def upload_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """
    Upload a new manual / SOP / procedure document (PDF, .md, .txt).
    It is immediately added to the live ChromaDB index.
    """
    allowed = {".pdf", ".md", ".txt", ".markdown"}
    suffix = Path(file.filename).suffix.lower()
    
    if suffix not in allowed:
        raise HTTPException(
            400, 
            f"Unsupported file type: {suffix}. Allowed: {allowed}"
        )
    
    # Save to temp, then ingest
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        success = ingest_single_document(
            file_path=tmp_path,
            original_filename=file.filename
        )
        
        if success:
            logger.info("Document uploaded and indexed", filename=file.filename)
            return {
                "success": True,
                "filename": file.filename,
                "message": f"Successfully added {file.filename} to knowledge base. It is now available for RAG queries.",
                "description": description
            }
        else:
            raise HTTPException(500, "Failed to ingest document into vector store")
            
    except Exception as e:
        logger.error("Upload failed", filename=file.filename, error=str(e))
        raise HTTPException(500, f"Upload failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
