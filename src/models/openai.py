import time

from openai import OpenAI

from src.constants import OPENAI_API_KEY, API_TIMEOUT, API_RETRY
from src.models.base import BaseModel, PromptResponse


class OpenAIModel(BaseModel):
    def __init__(self, sleep_time: int, model_name: str) -> None:
        if model_name is None:
            error_msg = "OpenAI model name is required"
            raise Exception(error_msg)

        super().__init__(sleep_time=sleep_time)

        self.__model_name = model_name
        self.__client = OpenAI(api_key=OPENAI_API_KEY, timeout=API_TIMEOUT)
        self.__max_retries = API_RETRY
        self.__backoff_base = 2  # seconds

    def prompt(self, prompt: str) -> PromptResponse:
        self._logger.debug(f"Sleeping for '{self._sleep_time}' seconds")
        time.sleep(self._sleep_time)
        self._logger.debug(f"Slept for '{self._sleep_time}' seconds")

        last_exception: Exception | None = None

        for attempt in range(1, self.__max_retries + 1):
            try:
                self._logger.debug(f"Invoking OpenAI API client (attempt {attempt}/{self.__max_retries})")
                start_time = time.monotonic()

                response = self.__client.chat.completions.create(
                    model=self.__model_name,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                self._logger.debug("Finished OpenAI API client invocation")
                
                time_taken_s = time.monotonic() - start_time

                self._logger.debug("Returning response")
                return PromptResponse(
                    time_taken=time_taken_s,
                    prompt_message=prompt,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_message=response.choices[0].message.content,
                    completion_tokens=response.usage.completion_tokens,
                    model_name=response.model
                )

            except Exception as e:
                last_exception = e

                if attempt == self.__max_retries:
                    self._logger.error(f"All {self.__max_retries} attempts failed. Last error: {e}")
                    break

                backoff_time = self.__backoff_base ** attempt
                self._logger.warning(
                    f"Attempt {attempt}/{self.__max_retries} failed with error: {e}. "
                    f"Retrying in {backoff_time}s..."
                )
                time.sleep(backoff_time)

        raise last_exception