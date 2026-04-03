import argparse

import pandas as pd

from src.dataset_factory import DatasetFactory
from src.models.model_factory import ModelFactory
from src.models.base import BaseModel, PromptResponse
from src.utils.math_helpers import MathHelpers
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM prompts")
    parser.add_argument("--provider", choices=["ollama", "openai"], required=True, help="LLM provider to use")
    parser.add_argument("--model", required=True, help="Model name to use")
    return parser.parse_args()

# ollama: deepseek-r1:1.5b
# openai: gpt-4o-mini-2024-07-18
def main():
    args = parse_args()
    logger = setup_logger()

    sleep_time = 0

    initial_prompt = "Work out this math question and output the final answer. Return only the correct number answer. Provide no explanation, no sentences, no metrics, just the final answer number.\n\n{math_input}"
    llm: BaseModel = ModelFactory.get_llm(args.provider)(
        sleep_time=sleep_time,
        model_name=args.model
    )

    pd_df: pd.DataFrame = DatasetFactory.get_math_dataset()

    passed = 0
    total_time = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total = len(pd_df)

    for index, row in pd_df.iterrows():
        question = row["question"]
        prompt = initial_prompt.format(math_input=question)
        expected_answer = MathHelpers.get_expected_answer(answer_txt=row["answer"])

        logger.info(f"Question: {index}")
        logger.info(f"Expected Answer: {expected_answer}")
        logger.info(f"Prompt:\n'''{prompt}'''")
        
        try:
            response: PromptResponse = llm.prompt(prompt=prompt)

            time_taken = response["time_taken"]
            input_token_cnt = response["prompt_tokens"]
            output_token_cnt = response["completion_tokens"]
            raw_answer = response["completion_message"]
            actual_answer = MathHelpers.keep_only_numbers(text=raw_answer)

            logger.info(f"\nTime taken: '{time_taken}'")
            logger.info(f"Input token count: '{input_token_cnt}'")
            logger.info(f"Output token count: '{output_token_cnt}'")
            logger.info(f"(Raw) Answer generated: '{raw_answer}'")
            logger.info(f"\nAnswer generated: '{actual_answer}'")

            # stats across the whole dataset
            total_time += time_taken
            total_input_tokens += input_token_cnt
            total_output_tokens += output_token_cnt

            success = expected_answer == actual_answer
            if success:
                logger.info("\nCorrect answer!")
                passed += 1
            else:
                logger.info("\nIncorrect answer...")
        except Exception as e:
            logger.error(f"\nCould not process question: {index}. Error: {e}")
            logger.info(f"\nCould not process question '{index}'")
        finally:
            logger.info(f"\n{'-' * 50}\n")

    logger.info(f"Total count: '{total}'")
    logger.info(f"Passed count: '{passed}'")
    logger.info(f"Total time taken: '{total_time}'")
    logger.info(f"Total input tokens used: '{total_input_tokens}'")
    logger.info(f"Total output tokens used: '{total_output_tokens}'")

    logger.info(f"pass@1 score ({passed} / {total}): '{passed / total}'")

if __name__ == "__main__":
    main()