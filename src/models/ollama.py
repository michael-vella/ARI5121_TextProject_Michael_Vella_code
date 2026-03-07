import time

from ollama import Client

from src.constants import OLLAMA_HOST
from src.models.base import BaseModel, PromptResponse


class OllamaModel(BaseModel):
    def __init__(self, sleep_time: int, model_name: str) -> None:
        if model_name is None:
            raise Exception("Ollama model name is required")
        
        super().__init__(sleep_time=sleep_time)

        self.__model_name = model_name
        self.__client = Client(host=OLLAMA_HOST)

    def prompt(self, prompt: str) -> PromptResponse:
        time.sleep(self._sleep_time)

        response = self.__client.chat(
            model=self.__model_name,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return PromptResponse(
            time_taken=response.total_duration,
            prompt_message=prompt,
            prompt_tokens=response.prompt_eval_count,
            completion_message=response.message.content,
            completion_tokens=response.eval_count,
            model_name=response.model
        )