import logging
import argparse

import pandas as pd

from src.dataset_factory import DatasetFactory
from src.utils.math_helpers import MathHelpers
from src.models.model_factory import ModelFactory
from src.models.base import BaseModel, PromptResponse
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsState, Agent


def print_llm_response_metrics(logger: logging.Logger, response: PromptResponse) -> None:
    logger.info(f"Time taken: '{response['time_taken']}'")
    logger.info(f"Input token count: '{response['prompt_tokens']}'")
    logger.info(f"Output token count: '{response['completion_tokens']}'")

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM prompts")
    parser.add_argument("--provider", choices=["ollama", "openai"], required=True, help="LLM provider to use")
    parser.add_argument("--model", required=True, help="Model name to use")
    parser.add_argument("--max_plan_try", required=True, help="Max. no. of planning agent retry")
    parser.add_argument("--max_debug_try", required=True, help="Max. no. of debugging agent retry")
    return parser.parse_args()


# ollama: deepseek-r1:1.5b
# openai: gpt-4o-mini-2024-07-18
def main():
    args = parse_args()
    logger = setup_logger()

    provider: str = args.provider
    model_name: str = args.model
    max_plan_try: int = int(args.max_plan_try)
    max_debug_try: int = int(args.max_debug_try)

    pd_df: pd.DataFrame = DatasetFactory.get_math_dataset()

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

    # get prompts
    planning_prompt = MathHelpers.get_planning_prompt()
    simulation_prompt = MathHelpers.get_simulation_prompt()
    plan_refinement_prompt = MathHelpers.get_plan_refinement_prompt()
    math_solution_prompt = MathHelpers.get_math_solution_prompt()
    debugging_prompt = MathHelpers.get_debugging_prompt()

    # iterate over the data
    for index, row in pd_df.iterrows():
        question = row["question"]
        expected_answer = MathHelpers.get_expected_answer(answer_txt=row["answer"])

        logger.info(f"Question number: {index}")
        logger.info(f"\nQuestion: {question}")
        logger.info(f"Expected Answer: {expected_answer}")
        
        # Planning, Math, Debugging
        for plan_no in range(1, max_plan_try + 1):
            # Planning Phase
            logger.info(f"\n--------- PLANNING PHASE ITERATION NUMBER '{plan_no}' ---------")

            # start: initial plan generation
            plan_input_prompt = planning_prompt.format(problem=question)
            logger.info(f"\n--- INITIAL PLAN GENERATION (LLM INPUT):\n\n{plan_input_prompt}")
            plan_response: PromptResponse = llm.prompt(prompt=plan_input_prompt)
            raw_plan = plan_response["completion_message"]
            logger.info(f"\n--- INITIAL PLAN GENERATION (LLM RESPONSE):\n\n{raw_plan}")

            metrics.record(Agent.PLANNING, index, plan_response)
            logger.info(f"\n--- INITIAL PLAN GENERATION (LLM METRICS):")
            print_llm_response_metrics(logger=logger, response=plan_response)
            # end: initial plan generation

            # start: simulation
            if "### Plan" not in raw_plan:
                plan = f"### Plan\n\n{raw_plan}"
            else:
                plan = raw_plan[raw_plan.rfind("### Plan"):]
            problem_with_plan = f"### Problem:\n{question}\n\n{plan}"

            simulation_input_prompt = simulation_prompt.format(problem_with_plan=problem_with_plan)
            logger.info(f"\n--- SIMULATION (LLM INPUT):\n\n{simulation_input_prompt}")
            simulation_response: PromptResponse = llm.prompt(prompt=simulation_input_prompt)
            raw_simulation = simulation_response["completion_message"]
            logger.info(f"\n--- SIMULATION (LLM RESPONSE):\n\n{raw_simulation}")

            metrics.record(Agent.SIMULATION, index, simulation_response)
            logger.info(f"\n--- SIMULATION (LLM METRICS):")
            print_llm_response_metrics(logger=logger, response=simulation_response)
            # end: simulation

            # start: plan refinement
            if "Plan Modification Needed" in raw_simulation and "No Plan Modification Needed" not in raw_simulation:
                plan_refinement_input_prompt = plan_refinement_prompt.format(
                    problem_with_plan=problem_with_plan,
                    critique=raw_simulation
                )
                logger.info(f"\n--- PLAN REFINEMENT (LLM INPUT):\n\n{plan_refinement_input_prompt}")
                plan_refinement_response: PromptResponse = llm.prompt(prompt=plan_refinement_input_prompt)
                plan = plan_refinement_response["completion_message"]
                logger.info(f"\n--- REFINEMENT RESPONSE (LLM RESPONSE):\n\n{plan}")

                metrics.record(Agent.PLAN_REFINEMENT, index, plan_refinement_response)
                logger.info(f"\n--- PLAN REFINEMENT (LLM METRICS):")
                print_llm_response_metrics(logger=logger, response=plan_refinement_response)
            # end: plan refinement

            # update `problem_with_plan` just in case plan was refined
            problem_with_plan = f"### Problem:\n{question}\n\n{plan}"

            # Math Phase
            # start: math generation
            logger.info("\n--------- MATH PHASE ---------")
            math_generation_input_prompt = math_solution_prompt.format(problem_with_plan=problem_with_plan)
            logger.info(f"\n--- MATH GENERATION (LLM INPUT):\n\n{math_generation_input_prompt}")
            math_generation_response: PromptResponse = llm.prompt(prompt=math_generation_input_prompt)
            raw_math_solution = math_generation_response["completion_message"]
            logger.info(f"\n--- MATH GENERATION (LLM RESPONSE):\n\n{raw_math_solution}")

            actual_answer = MathHelpers.extract_math_answer(raw_math_solution)
            logger.info(f"\n--- MATH GENERATION (AFTER CLEANUP):\n\n{actual_answer}")

            metrics.record(Agent.MATH_GENERATION, index, math_generation_response)
            logger.info(f"\n--- MATH GENERATION (LLM METRICS):")
            print_llm_response_metrics(logger=logger, response=math_generation_response)
            # end: math generation

            # start: evaluate math answer
            passed = expected_answer == actual_answer
            # end: evaluate math answer

            if passed:
                pass_count += 1
                logger.info(f"\n--- CORRECT ANSWER FOR QUESTION NUMBER '{index}'!")
                logger.info(f"\n{'-' * 200}\n")
                break
        
            # Debugging Phase
            for debug_no in range(1, max_debug_try + 1):
                # start: debugging
                logger.info(f"\n--------- DEBUGGING PHASE ITERATION NUMBER '{debug_no}' ---------")

                debugging_input_prompt = debugging_prompt.format(
                    problem_with_plan=problem_with_plan,
                    solution=raw_math_solution,
                    wrong_answer=actual_answer
                )
                logger.info(f"\n--- DEBUGGING (LLM INPUT):\n\n{debugging_input_prompt}")
                debugging_response: PromptResponse = llm.prompt(prompt=debugging_input_prompt)
                raw_debugging_solution = debugging_response["completion_message"]
                logger.info(f"\n--- DEBUGGING (LLM RESPONSE):\n\n{raw_debugging_solution}")

                actual_answer = MathHelpers.extract_math_answer(raw_debugging_solution)
                logger.info(f"\n--- DEBUGGING (MATH ANSWER AFTER CLEANUP):\n\n{actual_answer}")

                metrics.record(Agent.DEBUGGING, index, debugging_response)
                logger.info(f"\n--- DEBUGGING (LLM METRICS):")
                print_llm_response_metrics(logger=logger, response=debugging_response)

                # start: evaluate math answer
                passed = expected_answer == actual_answer
                # end: evaluate math answer
                # end: debugging

                # exit debugging loop
                if passed:
                    break
        
            if passed:
                pass_count += 1
                logger.info(f"\n--- CORRECT ANSWER FOR QUESTION NUMBER '{index}'!")
                logger.info(f"\n{'-' * 200}\n")
                break
        
        if not passed:
            logger.info(f"\n--- INCORRECT ANSWER FOR QUESTION NUMBER '{index}' AFTER PERFORMING '{max_plan_try}' PLANNING TRIES AND '{max_debug_try}' DEBUGGING TRIES...")
            logger.info(f"\n{'-' * 200}\n")

    logger.info(metrics.summary())

    logger.info(f"Total count: '{total}'")
    logger.info(f"Passed count: '{pass_count}'")
    logger.info(f"pass@1 score ({pass_count} / {total}): '{pass_count / total}'")

if __name__ == "__main__":
    main()