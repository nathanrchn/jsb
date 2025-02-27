from guidance import models
from dataclasses import dataclass
from typing import List, Literal, Optional

from api.base import Schema
from api.engine import Engine, EngineConfig, GenerationResponse


@dataclass
class GuidanceConfig(EngineConfig):
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    model_engine: Literal["llamacpp"] = "llamacpp"


class GuidanceEngine(Engine[GuidanceConfig]):
    def __init__(self, config: GuidanceConfig):
        super().__init__(config)

        match self.config.model_engine:
            case "llamacpp":
                from llama_cpp import Llama

                self.model = Llama.from_pretrained(self.config.model)

                self.guidance_model_state = models.LlamaCpp(
                    self.model, echo=False, caching=False
                )

                self.tokenizer = self.guidance_model_state.engine.tokenizer
            case _:
                raise ValueError(
                    f"model engine {self.config.model_engine} not supported"
                )

    def generate(self, prompt: str, schema: Schema) -> GenerationResponse:
        pass

    def encode(self, text: str) -> Optional[List[int]]:
        return self.tokenizer.encode(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> Optional[str]:
        return self.tokenizer.decode(ids).decode("utf-8")
