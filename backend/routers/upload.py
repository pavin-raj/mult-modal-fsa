from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tempfile
import os

from ai_core.rag.index_manager import ingest_single_document

router = APIRouter(prefix="/upload", tags=["Document Upload"])

@router.post("/document")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".md", ".txt", ".markdown"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        success = ingest_single_document(tmp_path, file.filename)
        if success:
            return {"status": "success", "message": f"'{file.filename}' added to knowledge base."}
        else:
            raise HTTPException(500, "Failed to process document")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)