"""
Speech-to-Text Service.
Uses faster-whisper for high-quality local transcription.
Falls back to mock when MOCK_MODE=true or no audio provided.
"""
import os
import base64
import io
import tempfile
import structlog

logger = structlog.get_logger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
STT_MODEL_SIZE = os.getenv("STT_MODEL", "base")

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    if MOCK_MODE:
        return "mock"
    try:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(STT_MODEL_SIZE, device="cpu", compute_type="int8")
        logger.info(f"Loaded faster-whisper model: {STT_MODEL_SIZE}")
        return _whisper_model
    except Exception as e:
        logger.error("Failed to load faster-whisper", error=str(e))
        return "mock"

def transcribe_audio(audio_b64: str) -> str:
    """
    Transcribe base64-encoded audio (expects WAV or compatible).
    Returns plain text transcript.
    """
    if MOCK_MODE or not audio_b64:
        logger.info("STT running in MOCK mode")
        return "The pump is vibrating and there's fluid leaking from the seal area."

    model = get_whisper_model()
    if model == "mock":
        return "The pump is vibrating and there's fluid leaking from the seal area. (mock transcript)"

    try:
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_b64)
        
        # Write to temp file (faster-whisper expects file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        segments, info = model.transcribe(tmp_path, beam_size=5, language="en")
        transcript = " ".join([seg.text for seg in segments]).strip()
        
        os.unlink(tmp_path)
        logger.info("STT completed", language=info.language, duration=info.duration)
        return transcript

    except Exception as e:
        logger.error("STT failed", error=str(e))
        return "Sorry, I couldn't understand the audio clearly. Please try again or type your question."
