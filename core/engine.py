from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Generic

from core.types import (
    Schema,
    TokenUsage,
    GenerationState,
)
from core.profile import profile_generation


@dataclass
class EngineConfig:
    pass


T = TypeVar("T", bound=EngineConfig)


class Engine(ABC, Generic[T]):
    def __init__(self, config: T):
        self.config = config
        self.total_usage = TokenUsage()

    @profile_generation
    def generate(
        self,
        task: str,
        prompt: str,
        schema: Schema,
    ) -> GenerationState:
        schema = self.adapt_schema(schema)
        state = GenerationState(task=task, input=prompt, output="", schema=schema)
        state.token_usage.input_tokens = self.count_tokens(state.input)

        self._generate(state)

        self.total_usage += state.token_usage
        return state

    @abstractmethod
    def _generate(
        self,
        state: GenerationState,
    ) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_context_length(self) -> int:
        raise NotImplementedError

    def adapt_schema(self, schema: Schema) -> Schema:
        return schema

    def encode(self, text: str) -> Optional[List[int]]:
        return None

    def decode(self, ids: List[int]) -> Optional[str]:
        return None

    def convert_token_to_id(self, token: str) -> Optional[int]:
        res = self.encode(token)
        return res[0] if res else None

    def convert_id_to_token(self, id: int) -> Optional[str]:
        res = self.decode([id])
        return res[0] if res else None

    def count_tokens(self, text: str) -> int:
        res = self.encode(text)
        return len(res) if res else 0
    
    def close(self):
        # Close the model and the sampler for LlamaCpp
        # https://github.com/abetlen/llama-cpp-python/issues/1610
        if hasattr(self.model, "_sampler"):
            self.model._sampler.close()
            self.model.close()

