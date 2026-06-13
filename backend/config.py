"""Simple configuration loader."""
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2:3b"
    VLM_MODEL: str = "llama3.2-vision:11b"
    EMBED_MODEL: str = "nomic-embed-text"
    
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    KNOWLEDGE_BASE_DIR: str = "./data/manuals"
    
    MOCK_MODE: bool = False
    ENABLE_VISION: bool = True
    ENABLE_SPEECH: bool = True
    
    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings():
    return Settings()
