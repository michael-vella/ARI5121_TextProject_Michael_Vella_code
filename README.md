# ARI5121 (Applied NLP) Text Assignment

The purpose of this GitHub repository is to host any code required for the ARI5121 (Text) Applied Natural Language Processing study-unit assignment.

## Project Notes

- **Paper to replicate**: [CodeSim: Multi-Agent Code Generation and Problem Solving through Simulation-Driven Planning and Debugging](https://aclanthology.org/2025.findings-naacl.285/).
- **Python version used**: 3.12.3 (refers to Python version used in this project not the Python version used in the original replication paper).
- **Prerequisites**: Requires Python & Ollama to be installed locally. Ollama is not necessarily required but is needed if you want to run experiments using Ollama models locally.

## Datasets

- [HumanEval](https://arxiv.org/abs/2107.03374) - Downloaded from [HuggingFace](https://huggingface.co/datasets/openai/openai_humaneval).
- [gsm8k](https://arxiv.org/abs/2110.14168) - Downloaded from [HuggingFace](https://huggingface.co/datasets/openai/gsm8k).

## Replication of virtual environment (.venv)

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

## Project Directory

- `datasets/`: Contains datasets used for this assignment.
- `logs/`: Directory to store log files. Not source-controlled until logs are migrated to the `results/` directory.
- `results/`: Directory storing results. Each sub-directory in this directory is named after the Python file that results were gathered from.
- `src/`: Directory containing re-usable python modules used for this assignment.
- `.gitignore`: Git file to ignore certain files from being source-controlled.
- `code_sim_ablation_runner.py`: Python script to run the ablation study of the CodeSim replication paper.
- `code_sim_math_ablation_runner.py`: Python script to run the ablation study after extending the work done in CodeSim and applying it to a Math dataset.
- `code_sim_math_runner.py`: Python script to run the study of extending the work done in CodeSim and applying it to a Math dataset.
- `code_sim_runner.py`: Python script to replicate the CodeSim paper.
- `README.md`: Project guide.
- `requirements.txt`: Text file denoting list of packages used for this assignment.
- `simple_math_runner.py`: Python script to get baseline results on the Math dataset.
- `simple_runner.py`: Python script to get baseline results before replicating the CodeSim paper.

## Replication of results

Important that once project is cloned, we create the `.env` file and the `OPENAI_API_KEY` as an environment variable (should be the secret key to use OpenAI API services).

To attempt to replicate results, follow the below steps.

### Simple Runner

Results directory: `results/simple_runner`

- Deepseek (`deepseek_15032026.txt`): Run `python simple_runner.py --provider ollama --model deepseek-r1:1.5b`
- GPT4o-mini (`gpt4omini_19032026.txt`): Run `python simple_runner.py --provider openai --model gpt-4o-mini-2024-07-18`

### CodeSim Replication

Results directory: `results/code_sim_runner`

- GPT4o-mini 1st run (`gpt4omini_20032026.txt`): Run `python code_sim_runner.py --provider openai --model gpt-4o-mini-2024-07-18 --max_plan_try 1 --max_debug_try 1`
- GPT4o-mini 2nd run (`gpt4omini_21032026.txt`): Run `python code_sim_runner.py --provider openai --model gpt-4o-mini-2024-07-18 --max_plan_try 3 --max_debug_try 3`

### CodeSim Replication (Ablation)

Results directory: `results/code_sim_ablation_runner`

- GPT4o-mini 1st run (`gpt4omini_21032026_A.txt`): Run `python code_sim_ablation_runner.py --provider openai --model gpt-4o-mini-2024-07-18 --max_plan_try 1 --max_debug_try 1`
- GPT4o-mini 2nd run (`gpt4omini_21032026_B.txt`): Run `python code_sim_ablation_runner.py --provider openai --model gpt-4o-mini-2024-07-18 --max_plan_try 3 --max_debug_try 3`

### Simple Math Runner

Results directory: `results/simple_math_runner`

- GPT4o-mini (`gpt4omini_03042026.txt`): Run `python simple_math_runner.py --provider openai --model gpt-4o-mini-2024-07-18`

### CodeSim Math Extension

Results directory: `results/code_sim_math_runner`

- GPT4o-mini (`gpt4omini_05042026.txt`): Run `python code_sim_math_runner.py --provider openai --model gpt-4o-mini-2024-07-18 --max_plan_try 1 --max_debug_try 1`

### CodeSim Math Extension (Ablation)

Results directory: `results/code_sim_math_ablation_runner`

- GPT4o-mini (`gpt4omini_04042026.txt`): Run `python code_sim_math_ablation_runner.py --provider openai --model gpt-4o-mini-2024-07-18 --max_plan_try 1 --max_debug_try 1`