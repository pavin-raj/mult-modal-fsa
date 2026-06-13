"""
Text-to-Speech Service.
Uses Piper TTS for fast, local, high-quality synthesis.
Falls back to mock (returns empty) when MOCK_MODE or Piper not available.
"""
import os
import base64
import tempfile
import structlog

logger = structlog.get_logger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
TTS_VOICE = os.getenv("TTS_VOICE", "en_US-amy-medium")

_piper_voice = None

def get_piper_voice():
    global _piper_voice
    if _piper_voice is not None:
        return _piper_voice
    if MOCK_MODE:
        return "mock"
    try:
        from piper import PiperVoice
        # Assumes voice model is downloaded (see docker setup)
        model_path = f"/app/voices/{TTS_VOICE}.onnx"
        if not os.path.exists(model_path):
            logger.warning(f"Piper voice model not found at {model_path}. Using mock.")
            return "mock"
        _piper_voice = PiperVoice.load(model_path)
        logger.info(f"Loaded Piper TTS voice: {TTS_VOICE}")
        return _piper_voice
    except Exception as e:
        logger.error("Failed to load Piper TTS", error=str(e))
        return "mock"

def synthesize_speech(text: str) -> str:
    """
    Convert text to speech. Returns base64-encoded WAV audio.
    """
    if not text or MOCK_MODE:
        logger.info("TTS running in MOCK mode (no audio generated)")
        return ""  # Frontend can use browser TTS as fallback

    voice = get_piper_voice()
    if voice == "mock":
        return ""

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        # Synthesize
        voice.synthesize(text, tmp_path)
        
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        
        os.unlink(tmp_path)
        
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        logger.info("TTS synthesis complete", text_length=len(text))
        return audio_b64

    except Exception as e:
        logger.error("TTS synthesis failed", error=str(e))
        return ""
