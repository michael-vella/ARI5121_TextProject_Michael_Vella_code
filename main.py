from src.models.model_factory import ModelFactory
from src.models.base import BaseModel


sleep_time=0

llm: BaseModel = ModelFactory.get_llm("ollama")(
    sleep_time=0,
    model_name="deepseek-r1:1.5b"
)

response = llm.prompt("Why is the sky blue?")
print("Response:", response)