# ARI5121 (Applied NLP) Text Assignment

To finalise README.md at the end of the project.

## Ollama Command Cheat Sheet

Assuming that Ollama is already installed on the host machine.

- `ollama serve` starts the Ollama API server, allowing you to run, manage, and interact with local Large Language Models (LLMs) via HTTP requests (usually on http://localhost:11434).
- `ollama run <model_name>` downloads (if necessary), loads, and initiates a local, interactive terminal session with a specified LLM (e.g., Llama 3, Mistral) on your machine.
- `ollama -v` or `ollama --version` checks the installed ollama version.
- `ollama list` or `ollama ls` lists all downloaded models.
- `ollama ps` lists all AI models currently loaded into memory (RAM or GPU). It provides a snapshot of running models, including their name, ID, size, whether they are running on the GPU or CPU, and how long they will remain in memory.
