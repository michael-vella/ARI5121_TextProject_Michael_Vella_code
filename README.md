# ARI5121 (Applied NLP) Text Assignment

To finalise README.md at the end of the project.

## Project Notes

- **Paper to replicate**: [CodeSim: Multi-Agent Code Generation and Problem Solving through Simulation-Driven Planning and Debugging](https://aclanthology.org/2025.findings-naacl.285/).
- **Python version used**: 3.12.3 (refers to Python version used in this project not the Python version used in the original replication paper).
- Requires Python & Ollama to be installed locally.

## Create a Python virtual environment (.venv)

Assuming that Python is already pre-installed on the host machine.

1. Run `python -m venv .venv` to create the Python virtual environment. `python` here refers to the alias of the Python executable path and depends on the alias used on the host machine (full Python path can also be used). Running this command will create a Python virtual environment depending on the base Python version being used to create the environment.
2. Activate virtual environment by running `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux).
3. Upgrade `pip` (Python's package manager) by running `pip install --upgrade pip`.
4. Run `pip install -r requirements.txt` to download any packages required for this project.

## Ollama Command Cheat Sheet

Assuming that Ollama is already installed on the host machine.

- `ollama serve` starts the Ollama API server, allowing you to run, manage, and interact with local Large Language Models (LLMs) via HTTP requests (usually on http://localhost:11434).
- `ollama run <model_name>` downloads (if necessary), loads, and initiates a local, interactive terminal session with a specified LLM (e.g., Llama 3, Mistral) on your machine.
- `ollama -v` or `ollama --version` checks the installed ollama version.
- `ollama list` or `ollama ls` lists all downloaded models.
- `ollama ps` lists all AI models currently loaded into memory (RAM or GPU). It provides a snapshot of running models, including their name, ID, size, whether they are running on the GPU or CPU, and how long they will remain in memory.
