import logging

from src.models.base import BaseModel
from src.models.ollama import OllamaModel


logger = logging.getLogger("my_app")


class ModelFactory:
    @staticmethod
    def get_llm(model_provider_name: str) -> BaseModel:
        logger.debug(f"Model provider name: '{model_provider_name}'")
        model_provider_name = model_provider_name.lower()
        logger.debug(f"Model provider name (lower): '{model_provider_name}'")

        match model_provider_name:
            case "ollama":
                logger.debug("Returning a (non-instantiated) Ollama class model")
                return OllamaModel
            case _:
                error_msg = f"Unknown model provider name '{model_provider_name}'"
                logger.error(error_msg)
                raise Exception(error_msg)