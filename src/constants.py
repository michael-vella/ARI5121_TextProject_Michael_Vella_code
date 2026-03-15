import os


# Ollama is used locally (does not require API key).
API_TIMEOUT = 300
API_RETRY = 3
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434/")