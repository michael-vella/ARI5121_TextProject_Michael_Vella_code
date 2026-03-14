import time

from ollama import Client

from src.constants import OLLAMA_HOST
from src.models.base import BaseModel, PromptResponse


class OllamaModel(BaseModel):
    def __init__(self, sleep_time: int, model_name: str) -> None:
        if model_name is None:
            error_msg = "Ollama model name is required"
            self._logger.error(error_msg)
            raise Exception(error_msg)
        
        super().__init__(sleep_time=sleep_time)

        self.__model_name = model_name
        self.__client = Client(host=OLLAMA_HOST)

    def prompt(self, prompt: str) -> PromptResponse:
        self._logger.debug(f"Sleeping for '{self._sleep_time}' seconds")
        time.sleep(self._sleep_time)
        self._logger.debug(f"Slept for '{self._sleep_time}' seconds")

        self._logger.debug("Invoking Ollama API client")
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