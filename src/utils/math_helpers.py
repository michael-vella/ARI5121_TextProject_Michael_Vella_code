import re


PLANNING_PROMPT: str = """You are a math expert tasked with generating a step-by-step plan to solve a math word problem.

## Problem

{problem}

**Expected Output:**

Your response must be structured as follows:

### Problem Understanding

- Think about the original problem. Develop an initial understanding of what quantity is being asked for and what information is given.

### Recall Example Problem

Recall a relevant and distinct math word problem (different from the problem above) and:
- Describe it briefly
- Solve it step by step
- Discuss the mathematical strategy used (e.g., unit conversion, setting up equations, ratio/proportion, working backwards)
- Generate a plan for solving that example problem

### Mathematical Strategy

- Identify the mathematical approach best suited for the original problem
- Explain how to apply this strategy (e.g., what to compute first, what intermediate values are needed)

### Plan

- Write a detailed, step-by-step plan to solve the **original problem**.
- Each step should clearly state what to compute and why.

--------
**Important Instructions:**
- Strictly follow the instructions.
- Do not compute the final numerical answer yet.
"""


SIMULATION_PROMPT = """You are a math expert tasked with verifying a plan for solving a math word problem.

{problem_with_plan}

**Expected Output:**

Your response must be structured as follows:

### Simulation

- Follow the plan step by step using the actual numbers from the problem.
- Show all intermediate calculations explicitly.
- State the final answer you arrive at after following the plan.

### Plan Evaluation

- If the simulation produces a logically consistent and correctly computed answer, write **No Need to Modify Plan**.
- If the simulation reveals a missing step, incorrect reasoning, or a calculation that leads nowhere, write **Plan Modification Needed**.
"""


PLAN_REFINEMENT_PROMPT = """You are a math expert tasked with correcting a flawed plan for solving a math word problem.

{problem_with_plan}

## Plan Critique

{critique}

**Expected Output:**

Your response must be structured as follows:

## New Plan

- Write a detailed, step-by-step corrected plan to solve the **original problem**.
- Ensure each step logically follows from the previous one.
- Address the issues identified in the critique.

--------
**Important Instructions:**
- Your response must contain only the new plan.
- Do not add any explanation outside the plan.
- Do not compute the final answer.
"""


MATH_SOLUTION_PROMPT = """You are a math expert tasked with solving a math word problem. Use the plan below to work through the problem carefully and arrive at the correct final numerical answer.

{problem_with_plan}

**Expected Output:**

Your response must be structured as follows:

### Solution

- Follow the plan step by step.
- Show all intermediate calculations explicitly.
- Do not skip steps.

### Final Answer

State only the final numerical answer on the last line in this exact format:
**Answer: <number>**

--------
**Important Instructions:**
- The answer must be a single number (integer or decimal).
- Do not include units or explanation in the Final Answer line.
- Strictly use the format: **Answer: <number>**
"""


DEBUGGING_PROMPT = """You are a math expert who has received a solution to a math word problem that produced an incorrect final answer. Your task is to identify the error and produce a corrected solution.

{problem_with_plan}

### Incorrect Solution

{solution}

### Incorrect Answer Produced

{wrong_answer}

**Expected Output:**

Your response must be structured as follows:

### Error Tracing

- Go through the incorrect solution step by step.
- Identify the specific step where the reasoning or calculation went wrong.
- Explain precisely what the error is (e.g., wrong operation, incorrect unit conversion, missed case, arithmetic mistake).

### Debugging Notes

- Was the plan itself flawed, or was the plan correct but executed incorrectly?
- Describe what must change to get the correct answer.

### Corrected Solution

- Redo the solution, correcting the identified error.
- Show all corrected calculations step by step.

### Final Answer

**Answer: <number>**

--------
**Important Instructions:**
- Do not assume the incorrect solution is almost right — re-examine every step.
- The final answer must follow exactly: **Answer: <number>**
- Your response must contain all four sections: Error Tracing, Debugging Notes, Corrected Solution, and Final Answer.
"""


class MathHelpers:
    @staticmethod
    def keep_only_numbers(text: str):
        match = re.search(r'-?\d+\.?\d*', str(text))
        return match.group(0) if match else 0

    @staticmethod
    def get_expected_answer(answer_txt: str):
        idx = answer_txt.find("#### ")
        idx += 5
        return float(MathHelpers.keep_only_numbers(answer_txt[idx:]) if idx != -1 else answer_txt)
    
    @staticmethod
    def extract_math_answer(raw_response: str) -> str:
        if '<think>' in raw_response and '</think>' in raw_response:
            raw_response = raw_response.split('</think>')[1]

        match = re.search(r'\*\*Answer:\s*([\d,\.]+)\*\*', raw_response)
        if match:
            return float(MathHelpers.keep_only_numbers(match.group(1)))

        # Fallback: last non-empty line
        lines = [l.strip() for l in raw_response.strip().splitlines() if l.strip()]
        return float(MathHelpers.keep_only_numbers(lines[-1]) if lines else 0)

    @staticmethod
    def get_planning_prompt() -> str:
        return PLANNING_PROMPT

    @staticmethod
    def get_simulation_prompt() -> str:
        return SIMULATION_PROMPT
    
    @staticmethod
    def get_plan_refinement_prompt() -> str:
        return PLAN_REFINEMENT_PROMPT
    
    @staticmethod
    def get_math_solution_prompt() -> str:
        return MATH_SOLUTION_PROMPT
    
    @staticmethod
    def get_debugging_prompt() -> str:
        return DEBUGGING_PROMPT