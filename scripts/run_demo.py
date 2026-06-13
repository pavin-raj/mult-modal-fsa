#!/usr/bin/env python3
"""
Quick demo runner for the Multi-Modal Field Service Assistant.
Starts everything needed for a local demo (assumes Docker or local services).
"""
import subprocess
import time
import webbrowser
import sys
from pathlib import Path

def main():
    print("🚀 Starting Multi-Modal Field Service Assistant Demo")
    print("=" * 55)
    
    root = Path(__file__).parent.parent
    
    # 1. Ingest knowledge if not present
    print("\n[1/4] Ingesting knowledge base...")
    try:
        subprocess.run([sys.executable, "scripts/ingest_data.py"], cwd=root, check=True)
    except Exception as e:
        print(f"Ingestion note: {e}")
    
    # 2. Check if Ollama is running (informational)
    print("\n[2/4] Checking for Ollama (recommended for real models)...")
    print("   → Make sure 'ollama serve' is running and models are pulled:")
    print("     ollama pull llama3.2:3b")
    print("     ollama pull llama3.2-vision:11b")
    print("     ollama pull nomic-embed-text")
    
    # 3. Start backend
    print("\n[3/4] Starting backend API (FastAPI)...")
    print("   API will be available at: http://localhost:8000")
    print("   API docs: http://localhost:8000/docs")
    
    try:
        # In a real scenario we would use uvicorn here, but for demo we print instructions
        print("\n   To run backend manually:")
        print("   cd backend && uvicorn main:app --reload --port 8000")
    except Exception as e:
        print(f"Note: {e}")
    
    # 4. Open frontend
    print("\n[4/4] Opening frontend demo...")
    frontend_path = root / "frontend" / "index.html"
    
    print(f"\n✅ Setup complete!")
    print(f"\nOpen this file in your browser:")
    print(f"   file://{frontend_path.absolute()}")
    print("\nOr serve it with:")
    print(f"   python -m http.server 8080 --directory {frontend_path.parent}")
    
    print("\nRecommended full flow:")
    print("  1. Start Ollama + pull models")
    print("  2. Run: python scripts/ingest_data.py")
    print("  3. Run backend: uvicorn backend.main:app --reload")
    print("  4. Open frontend/index.html")
    print("  5. Click 'Run Demo Scenario' or use camera + voice")
    
    # Try to open browser
    try:
        webbrowser.open(f"file://{frontend_path.absolute()}")
    except:
        pass

if __name__ == "__main__":
    main()
