import time

from ollama import Client

from src.constants import OLLAMA_HOST, API_TIMEOUT, API_RETRY
from src.models.base import BaseModel, PromptResponse


class OllamaModel(BaseModel):
    def __init__(self, sleep_time: int, model_name: str) -> None:
        if model_name is None:
            error_msg = "Ollama model name is required"
            self._logger.error(error_msg)
            raise Exception(error_msg)
        
        super().__init__(sleep_time=sleep_time)

        self.__model_name = model_name
        self.__client = Client(host=OLLAMA_HOST, timeout=API_TIMEOUT) # timeout after 5 minutes
        self.__max_retries = API_RETRY
        self.__backoff_base = 2 # seconds

    def prompt(self, prompt: str) -> PromptResponse:
        self._logger.debug(f"Sleeping for '{self._sleep_time}' seconds")
        time.sleep(self._sleep_time)
        self._logger.debug(f"Slept for '{self._sleep_time}' seconds")

        last_exception: Exception | None = None

        # implementing a retry mechanism
        # sometimes Ollama is hanging (sending a request but never recieving a response)
        # most probably, this is due to the limited computed available
        # hence, we implement a timeout and an API retry mechanism
        # we retry 3 times
        # after each retry, we wait 2^n where n is the attempt number to allow the server to (hopefully) heal
        for attempt in range(1, self.__max_retries + 1):
            try:
                self._logger.debug(f"Invoking Ollama API client (attempt {attempt}/{self.__max_retries})")
                response = self.__client.chat(
                    model=self.__model_name,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                self._logger.debug("Finished Ollama API client invocation")

                self._logger.debug("Converting time taken from nano seconds to seconds")
                time_taken_nano_s = response.total_duration
                time_taken_s = time_taken_nano_s / 1_000_000_000

                self._logger.debug("Returning response")
                return PromptResponse(
                    time_taken=time_taken_s,
                    prompt_message=prompt,
                    prompt_tokens=response.prompt_eval_count,
                    completion_message=response.message.content,
                    completion_tokens=response.eval_count,
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