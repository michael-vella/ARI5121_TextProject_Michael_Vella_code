import subprocess
import tempfile
import os
import re

import pandas as pd

from src.models.model_factory import ModelFactory
from src.models.base import BaseModel


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

def run_test(generated_code: str, test_code: str, entry_point: str) -> bool:
    """
    Combines generated code + test harness, runs it, returns True if it passes.
    """
    full_program = "from typing import *" + "\n\n" + generated_code + "\n\n" + test_code + "\n\n" + f"check({entry_point})"
    print("Full program:", full_program)

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


sleep_time = 0

initial_prompt = f"Complete this Python function. Return only the code, no explanation.\n\n{{prompt}}"
llm: BaseModel = ModelFactory.get_llm("ollama")(
    sleep_time=0,
    model_name="deepseek-r1:1.5b"
)

pd_df: pd.DataFrame = pd.read_parquet("datasets/human_eval/data.parquet")

passed = 0
total = len(pd_df)

for index, row in pd_df.iterrows():
    task_id = row["task_id"]
    prompt = row["prompt"]
    test = clean_test(row["test"])
    entry_point = row["entry_point"]

    print("Task ID:", task_id)
    print("\nPrompt:", prompt)
    print("\nTest:", test)

    response = llm.prompt(initial_prompt.format(prompt=prompt))
    print("\nResponse:", response)

    code_response = extract_code(response["completion_message"])
    print("\nCode generated:", code_response)

    success = run_test(code_response, test, entry_point)
    print("\nTest result:", success)
    if success:
        print("\nTest Passed!")
        passed += 1
    else:
        print("\nTest Failed...")

print("Passed:", passed)
print("Total:", total)
print("pass@1 score", passed / total)