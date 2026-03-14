import logging
from abc import ABC, abstractmethod
from typing import TypedDict


logger = logging.getLogger("my_app")


class PromptResponse(TypedDict):
    """
    Class representing response capture after prompting the LLM.

    Attributes:
    - time_taken (float): Time taken to receive output from model (seconds).
    - prompt_message (str): Input message sent to model.
    - prompt_tokens (int): Number of tokens inside input message.
    - completion_message (str): Message received by the model.
    - completion_tokens (int): Number of tokens inside completion message.
    - model_name (str): Name of the model being used.
    """
    time_taken: float
    prompt_message: str
    prompt_tokens: int
    completion_message: str
    completion_tokens: int
    model_name: str


class BaseModel(ABC):
    def __init__(self, sleep_time: int) -> None:
        self._logger = logger
        self._sleep_time = sleep_time

    @abstractmethod
    def prompt(self, prompt: str) -> PromptResponse:
        pass