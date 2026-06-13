"""
Vision Tool for the Field Service Agent.
Uses Ollama VLM (Llama-3.2-Vision or similar) for equipment identification and fault detection.
Falls back to mock when MOCK_MODE is enabled.
"""
import base64
import time
import os
from typing import Optional
from langchain_core.tools import tool
from ai_core.models.schemas import VisionResult
import structlog

logger = structlog.get_logger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VLM_MODEL = os.getenv("VLM_MODEL", "llama3.2-vision:11b")

try:
    import ollama
    OLLAMA_CLIENT = ollama.Client(host=OLLAMA_BASE_URL)
except ImportError:
    OLLAMA_CLIENT = None
    logger.warning("ollama package not installed. Vision will be mocked.")


def _encode_image(image_b64: str) -> str:
    """Ensure proper base64 format for Ollama."""
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",", 1)[1]
    return image_b64


@tool
def analyze_image(image_b64: str, focus: str = "general", equipment_hint: Optional[str] = None) -> dict:
    """
    Analyze an image of field equipment using a Vision-Language Model.
    Identifies equipment model, visible faults, condition, and provides structured output.
    
    Use this tool whenever the technician provides a photo or live camera feed.
    """
    start = time.time()
    image_b64 = _encode_image(image_b64)

    if MOCK_MODE or OLLAMA_CLIENT is None:
        logger.info("Using MOCK vision analysis")
        result = {
            "equipment_id": "PUMP-X450-001",
            "equipment_model": "X-450 Centrifugal Pump",
            "equipment_type": "Centrifugal Pump",
            "detected_faults": ["seal leakage", "possible bearing wear", "vibration marks on housing"],
            "visual_description": "Industrial centrifugal pump with visible fluid leakage at the mechanical seal area. Housing shows signs of corrosion and heat discoloration near the motor coupling. Nameplate partially visible: 'Model X-450, Serial 78432'.",
            "confidence": 0.87,
            "bounding_boxes": [
                {"label": "mechanical_seal", "bbox": [120, 340, 280, 410]},
                {"label": "motor_coupling", "bbox": [450, 180, 620, 290]}
            ],
            "ocr_text": ["X-450", "Serial: 78432", "Max Flow 450 GPM"],
            "raw_vlm_output": "Mock analysis for demo purposes."
        }
        latency = int((time.time() - start) * 1000)
        return {"success": True, "data": result, "error": None, "tool_name": "analyze_image", "latency_ms": latency}

    try:
        prompt = f"""You are an expert industrial equipment inspector assisting field technicians.

Analyze the provided image carefully.

Focus: {focus}
Equipment hint (if any): {equipment_hint or 'none'}

Return ONLY a valid JSON object with these exact keys (no markdown, no extra text):
{{
  "equipment_id": "string or null",
  "equipment_model": "string or null",
  "equipment_type": "string or null",
  "detected_faults": ["list of specific visible issues"],
  "visual_description": "detailed 2-3 sentence description of what you see, including condition, leaks, damage, labels",
  "confidence": 0.0 to 1.0,
  "bounding_boxes": [ {{"label": "part_name", "bbox": [x1,y1,x2,y2]}} ],
  "ocr_text": ["any readable text from labels or nameplates"]
}}

Be precise and conservative with confidence. If you are unsure about something, use lower confidence or null."""

        response = OLLAMA_CLIENT.chat(
            model=VLM_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_b64]
            }],
            options={"temperature": 0.1, "num_predict": 600}
        )

        content = response['message']['content'].strip()
        
        # Robust JSON extraction (handles occasional markdown or extra text)
        import json, re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
        else:
            parsed = json.loads(content)

        vision = VisionResult(**parsed)
        latency = int((time.time() - start) * 1000)

        return {
            "success": True,
            "data": vision.model_dump(),
            "error": None,
            "tool_name": "analyze_image",
            "latency_ms": latency
        }

    except Exception as e:
        logger.error("Vision analysis failed", error=str(e))
        latency = int((time.time() - start) * 1000)
        return {
            "success": False,
            "data": None,
            "error": str(e),
            "tool_name": "analyze_image",
            "latency_ms": latency
        }
