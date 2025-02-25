from dataclasses import dataclass

from api.engine import Engine, EngineConfig

try:
    from tiktoken import encoding_for_model
    from openai import OpenAI
except ImportError:
    print("openai is not installed, please install it to use openai engine")
    raise


@dataclass
class OpenAIConfig(EngineConfig):
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    top_p: float


class OpenAIEngine(Engine):
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.client = OpenAI(api_key=config.api_key)
