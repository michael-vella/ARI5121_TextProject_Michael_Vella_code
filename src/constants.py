import os


# Ollama is used locally (does not require API key).
API_TIMEOUT = 300
API_RETRY = 3
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434/")