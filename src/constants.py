import os


# Ollama is used locally (does not require API key).
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434/")