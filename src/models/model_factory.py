from src.models.base import BaseModel
from src.models.ollama import OllamaModel


class ModelFactory:
    @staticmethod
    def get_llm(model_provider_name: str) -> BaseModel:
        print("Model provider name:", model_provider_name)
        model_provider_name = model_provider_name.lower()
        print("Model provider name (lower):", model_provider_name)

        match model_provider_name:
            case "ollama":
                print("Returning a (non-instantiated) Ollama class model")
                return OllamaModel
            case _:
                raise Exception(f"Unknown model provider name '{model_provider_name}'")