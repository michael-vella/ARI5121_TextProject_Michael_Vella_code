import subprocess
import tempfile
import os
import re

import pandas as pd

from src.models.model_factory import ModelFactory
from src.models.base import BaseModel, PromptResponse
from src.utils.logger import setup_logger


def extract_code(model_output: str) -> str:
    match = re.search(r"```python\n(.*?)```", model_output, re.DOTALL)
    
    if match:
        code = match.group(1)
    else:
        # Fallback: use raw output if no fences found
        code = model_output.strip()

    # Combine prompt + completion so the function name is defined
    return code

def clean_test(test_code: str) -> str:
    idx = test_code.find("def ")
    return test_code[idx:] if idx != -1 else test_code

def add_imports(generated_code: str) -> str:
    return "from typing import *" + "\n\n" + generated_code

def run_test(generated_code: str, test_code: str, entry_point: str) -> bool:
    """
    Combines generated code + test harness, runs it, returns True if it passes.
    """
    full_program = generated_code + "\n\n" + test_code + "\n\n" + f"check({entry_point})"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_program)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            timeout=10,          # prevent infinite loops
            capture_output=True,
            text=True
        )
        return result.returncode == 0   # 0 = all assertions passed
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(tmp_path)


logger = setup_logger()


sleep_time = 0

initial_prompt = f"Complete this Python function. Return only the code, no explanation.\n\n{{code_input}}"
llm: BaseModel = ModelFactory.get_llm("ollama")(
    sleep_time=0,
    model_name="deepseek-r1:1.5b"
)

pd_df: pd.DataFrame = pd.read_parquet("datasets/human_eval/data.parquet")

passed = 0
total_time = 0
total_input_tokens = 0
total_output_tokens = 0
total = len(pd_df)

for index, row in pd_df.iterrows():
    task_id = row["task_id"]
    prompt = initial_prompt.format(code_input=row["prompt"])
    test = clean_test(row["test"])
    entry_point = row["entry_point"]

    logger.info(f"Task ID: {task_id}")
    logger.info(f"\nPrompt:\n'''\n{prompt}'''")
    logger.info(f"\nTests:\n```\n{test}```")

    response: PromptResponse = llm.prompt(prompt=prompt)

    time_taken = response["time_taken"]
    input_token_cnt = response["prompt_tokens"]
    output_token_cnt = response["completion_tokens"]
    code_response = extract_code(response["completion_message"])
    final_code = add_imports(generated_code=code_response)

    logger.info(f"\nTime taken: '{time_taken}'")
    logger.info(f"Input token count: '{input_token_cnt}'")
    logger.info(f"Output token count: '{output_token_cnt}'")
    logger.info(f"\nCode generated:\n```\n{final_code}```")

    success = run_test(final_code, test, entry_point)
    if success:
        logger.info("\nTests passed!")
        passed += 1
    else:
        logger.info("\nTests failed...")
    
    logger.info(f"\n{'-' * 50}\n")

logger.info(f"Passed count: '{passed}'")
logger.info(f"Total count: '{total}'")
logger.info(f"pass@1 score: '{passed / total}'")