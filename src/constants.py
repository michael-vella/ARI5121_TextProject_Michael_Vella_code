import os


# Ollama is used locally (does not require API key).
# However, include API key to make solution work with cloud Ollama.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434/")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "") 