"""
Upload Router - Allows users to upload PDFs / manuals directly from the app.
These get added live to the ChromaDB without restarting the server.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from pathlib import Path

from ai_core.rag.index_manager import ingest_single_document

router = APIRouter(prefix="/upload", tags=["Document Upload"])


@router.post("/document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, Markdown, or Text file to be added to the knowledge base.
    """
    allowed_extensions = {".pdf", ".md", ".txt", ".markdown"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(allowed_extensions)}"
        )

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        success = ingest_single_document(
            file_path=tmp_path,
            original_filename=file.filename
        )

        if success:
            return {
                "status": "success",
                "message": f"Document '{file.filename}' was successfully added to the knowledge base.",
                "filename": file.filename
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to process the document.")

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/status")
async def upload_status():
    return {"message": "Upload endpoint is active. POST a PDF/MD/TXT to /upload/document"}