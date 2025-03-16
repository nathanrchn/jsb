from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Generic

from core.types import (
    Schema,
    TokenUsage,
    GenerationResult,
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
        prompt: str,
        schema: Schema,
        task: str,
    ) -> GenerationResult:
        schema = self.adapt_schema(schema)
        result = self._generate(prompt, schema)

        self.total_usage += result.token_usage
        result.json_schema = schema
        result.task = task
        return result

    @abstractmethod
    def _generate(
        self,
        prompt: str,
        schema: Schema,
    ) -> GenerationResult:
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
        pass
