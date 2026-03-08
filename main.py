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

pd_df: pd.DataFrame = pd.read_parquet("datasets/human_eval/data.parquet")
row_zero = pd_df.iloc[0]
row_zero_prompt = row_zero["prompt"]
row_zero_test = clean_test(row_zero["test"])
row_zero_entry_point = row_zero["entry_point"]

print("\nPrompt for row zero:", row_zero_prompt)
print("\nTest for row zero:", row_zero_test)

sleep_time=0
initial_prompt = f"Complete this Python function. Return only the code, no explanation.\n\n{{prompt}}"

llm: BaseModel = ModelFactory.get_llm("ollama")(
    sleep_time=0,
    model_name="deepseek-r1:1.5b"
)

response = llm.prompt(initial_prompt.format(prompt=row_zero_prompt))
print("Response:", response)

code_response = extract_code(response["completion_message"])
print("Code generated:", code_response)

test_result = run_test(code_response, row_zero_test, row_zero_entry_point)
print("Test result:", test_result)