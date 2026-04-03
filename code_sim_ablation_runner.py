import logging
import re
import argparse

import pandas as pd

from src.dataset_factory import DatasetFactory
from src.utils.code_evaluation import CodeEvaluation
from src.models.model_factory import ModelFactory
from src.models.base import BaseModel, PromptResponse
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsState, Agent


def clean_tests(test_code: str) -> str:
    idx = test_code.find("def ")
    return test_code[idx:] if idx != -1 else test_code

def extract_code(raw_code: str) -> str:
    """
    Retriving the code blocks from the response.
    """

    if '<think>' in raw_code and '</think>' in raw_code:
        raw_code = raw_code.split('</think>')[1]
    
    if raw_code is None:
        return ''
    
    if "```" not in raw_code:
        return raw_code

    code_pattern = r'```((.|\n)*?)```'
    if "```Python" in raw_code:
        code_pattern = r'```Python((.|\n)*?)```'
    if "```Python3" in raw_code:
        code_pattern = r'```Python3((.|\n)*?)```'
    if "```python" in raw_code:
        code_pattern = r'```python((.|\n)*?)```'
    if "```python3" in raw_code:
        code_pattern = r'```python3((.|\n)*?)```'

    code_blocks = re.findall(code_pattern, raw_code, re.DOTALL)

    if type(code_blocks[-1]) == tuple or type(code_blocks[-1]) == list:
        code_str = "\n".join(code_blocks[-1])
    elif type(code_blocks[-1]) == str:
        code_str = code_blocks[-1]
    else:
        code_str = raw_code

    return code_str.strip()

def format_test_logs(failed_tests: list) -> str:
    new_failures = []
    for failed_test in failed_tests:
        new_failures.append(f"Assertion: {failed_test['test']}. Error: {failed_test['error']}")

    failed_test_cases_str = "\n".join(new_failures)
    return f"### Test Cases where the generated code failed to generate the expected output:\n{failed_test_cases_str}"

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM prompts")
    parser.add_argument("--provider", choices=["ollama", "openai"], required=True, help="LLM provider to use")
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument("--max_plan_try", required=True, help="Max. no. of planning agent retry")
    parser.add_argument("--max_debug_try", required=True, help="Max. no. of debugging agent retry")
    return parser.parse_args()

def print_llm_response_metrics(logger: logging.Logger, response: PromptResponse) -> None:
    logger.info(f"Time taken: '{response['time_taken']}'")
    logger.info(f"Input token count: '{response['prompt_tokens']}'")
    logger.info(f"Output token count: '{response['completion_tokens']}'")


PLANNING_PROMPT = """You are a programmer tasked with generating appropriate plan to solve a given problem using the **{language}** programming language.

## Problem

{problem}

**Expected Output:**

Your response must be structured as follows:

### Problem Understanding

- Think about the original problem. Develop an initial understanding about the problem.

### Recall Example Problem

Recall a relevant and distinct problems (different from problem mentioned above) and
- Describe it
- Generate {language} code step by step to solve that problem
- Discuss the algorithm to solve this problem
- Finally generate a planning to solve that problem

### Algorithm to solve the original problem

- Write down the algorithm that is well suited for the original problem
- Give some tutorials to about the algorithm for example:
    - How to approach this type of algorithm
    - Important things to consider

### Plan

- Write down a detailed, step-by-step plan to solve the **original problem**.

--------
**Important Instruction:**
- Strictly follow the instructions.
- Do not generate code.
"""

CODE_GENERATION_PROMPT = """You are a programmer tasked with solving a given problem using the **{language}** programming language. See the plan to solve the plan and implement code to solve it.

{problem_with_plan}

--------
**Important Instructions:**
- Do not add any explanation.
- The generated **{language}** code must be inside a triple backtick (```) code block.
- Do not add testing code for example assert statement in your code.
- Strictly follow the sample input and output format"""


DEBUGGING_PROMPT = """You are a programmer who has received a solution of a problem written in **{language}** that fails to pass certain test cases. Your task is to modify the code in such a way so that it can pass all the test cases. Do not generate same code.

{problem_with_plan}

### Buggy Code
```{language}
{code}
```

{test_log}

**Expected Output:**

Your response must be structured as follows:

### Simulation with failed test case
To detect where is the bug follow following steps:
    - Take a sample test case where it fails.
    - Take the input go through each step according to the plan
    - You will get a output that must be different from the expected output.

### Debugging Notes
- Based on this simulation detect any of the following cases:
    - Plan is wrong
    - Plan is correct but plan to code generation is wrong.
- Finally, discuss how to correct this code.

### Modified Code

```{language}
# Your corrected code, with comments explaining each correction.
```

--------
**Important Instructions:**
- Strictly follow the instructions.
- Do not add testing code for example assert statement in your code.
- Do not be overconfident that the generated code is correct. It is wrong.
- The modified **{language}** code must be enclosed within triple backticks (```).
- Your response must contain **Simulation with failed test case**, **Debugging Notes**,
and **Modified Code** section.
- Strictly follow the sample input and output format
"""


# ollama: deepseek-r1:1.5b
# openai: gpt-4o-mini-2024-07-18
def main():
    args = parse_args()
    logger = setup_logger()

    provider: str = args.provider
    model_name: str = args.model
    # hyperparameters - vary in the OG paper but better pass@1 results with larger values
    max_plan_try: int = int(args.max_plan_try)
    max_debug_try: int = int(args.max_debug_try)

    pd_df: pd.DataFrame = DatasetFactory.get_code_dataset()

    # in the original work, there is support for multiple languages
    # we will only support Python 3 for the sake of simplicity
    language: str = "Python"

    sleep_time = 0
    llm: BaseModel = ModelFactory.get_llm(provider)(
        sleep_time=sleep_time,
        model_name=model_name
    )

    metrics = MetricsState()
    total = len(pd_df)
    pass_count = 0

    logger.info(f"MODEL: '{model_name}'")
    logger.info("HYPERPARAMETERS:")
    logger.info(f"MAX PLANNING AGENT TRIES: '{max_plan_try}'")
    logger.info(f"MAX DEBUGGING AGENT TRIES: '{max_debug_try}'\n")

    # iterate over the data
    for _, row in pd_df.iterrows():
        task_id = row["task_id"]
        problem = row["prompt"]
        tests = clean_tests(row["test"])
        entry_point = row["entry_point"]

        logger.info(f"Task ID: {task_id}")
        logger.info(f"\nProblem:\n```\n{problem}```")
        logger.info(f"\nTests:\n```\n{tests}```")
        

        # Planning, Coding, Debugging
        for plan_no in range(1, max_plan_try + 1):
            # Planning Phase
            logger.info(f"\n--------- PLANNING PHASE ITERATION NUMBER '{plan_no}' ---------")

            # start: initial plan generation
            plan_input_prompt = PLANNING_PROMPT.format(
                language=language,
                problem=problem
            )
            logger.info(f"\n--- INITIAL PLAN GENERATION (LLM INPUT):\n\n{plan_input_prompt}")
            plan_response: PromptResponse = llm.prompt(prompt=plan_input_prompt)
            raw_plan = plan_response["completion_message"]
            logger.info(f"\n--- INITIAL PLAN GENERATION (LLM RESPONSE):\n\n{raw_plan}")

            metrics.record(Agent.PLANNING, task_id, plan_response)
            logger.info(f"\n--- INITIAL PLAN GENERATION (LLM METRICS):")
            print_llm_response_metrics(logger=logger, response=plan_response)

            if "### Plan" not in raw_plan:
                plan = f"### Plan\n\n{raw_plan}"
            else:
                plan = raw_plan[raw_plan.rfind("### Plan"):]
            problem_with_plan = f"### Problem:\n{problem}\n\n{plan}"
            # end: initial plan generation

            # Coding Phase
            # start: code generation
            logger.info("\n--------- CODING PHASE ---------")
            code_generation_input_prompt = CODE_GENERATION_PROMPT.format(
                language=language,
                problem_with_plan=problem_with_plan
            )
            logger.info(f"\n--- CODE GENERATION (LLM INPUT):\n\n{code_generation_input_prompt}")
            code_generation_response: PromptResponse = llm.prompt(prompt=code_generation_input_prompt)
            raw_code = code_generation_response["completion_message"]
            logger.info(f"\n--- CODE GENERATION (LLM RESPONSE):\n\n{raw_code}")

            code = extract_code(raw_code)
            logger.info(f"\n--- CODE GENERATION (AFTER CLEANUP):\n\n{code}")

            metrics.record(Agent.CODE_GENERATION, task_id, code_generation_response)
            logger.info(f"\n--- CODE GENERATION (LLM METRICS):")
            print_llm_response_metrics(logger=logger, response=code_generation_response)
            # end: code generation

            # start: test code generated
            passed, test_log = CodeEvaluation.evaluate_code(
                code=code,
                test_cases=tests,
                entry_point=entry_point
            )
            # end: test code generated

            if passed:
                pass_count += 1
                logger.info(f"\n--- ALL TESTS PASSED FOR TASK '{task_id}'!")
                logger.info(f"\n{'-' * 200}\n")
                break
        
            # Debugging Phase
            for debug_no in range(1, max_debug_try + 1):
                # start: debugging
                logger.info(f"\n--------- DEBUGGING PHASE ITERATION NUMBER '{debug_no}' ---------")

                formatted_test_log = format_test_logs(test_log)
                debugging_input_prompt = DEBUGGING_PROMPT.format(
                    language=language,
                    problem_with_plan=problem_with_plan,
                    code=code,
                    test_log=formatted_test_log
                )
                logger.info(f"\n--- DEBUGGING (LLM INPUT):\n\n{debugging_input_prompt}")
                debugging_response: PromptResponse = llm.prompt(prompt=debugging_input_prompt)
                raw_debugging_str = debugging_response["completion_message"]
                logger.info(f"\n--- DEBUGGING (LLM RESPONSE):\n\n{raw_debugging_str}")

                code = extract_code(raw_debugging_str)
                logger.info(f"\n--- DEBUGGING (CODE AFTER CLEANUP):\n\n{code}")

                metrics.record(Agent.DEBUGGING, task_id, debugging_response)
                logger.info(f"\n--- DEBUGGING (LLM METRICS):")
                print_llm_response_metrics(logger=logger, response=debugging_response)

                # start: test code generated
                passed, test_log = CodeEvaluation.evaluate_code(
                    code=code,
                    test_cases=tests,
                    entry_point=entry_point
                )
                # end: test code generated
                # end: debugging

                # exit debugging loop
                if passed:
                    break
        
            if passed:
                pass_count += 1
                logger.info(f"\n--- ALL TESTS PASSED FOR TASK '{task_id}'!")
                logger.info(f"\n{'-' * 200}\n")
                break
        
        if not passed:
            logger.info(f"\n--- TESTS FAILED FOR TASK '{task_id}' AFTER PERFORMING '{max_plan_try}' PLANNING TRIES AND '{max_debug_try}' DEBUGGING TRIES...")
            logger.info(f"\n{'-' * 200}\n")

    logger.info(metrics.summary())

    logger.info(f"Total count: '{total}'")
    logger.info(f"Passed count: '{pass_count}'")
    logger.info(f"pass@1 score ({pass_count} / {total}): '{pass_count / total}'")

if __name__ == "__main__":
    main()