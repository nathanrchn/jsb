from uuid import uuid4
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TypeVar, Generic

from utils import (
    TokenUsage,
    Token,
    GenerationMetadata,
    PerfMetrics,
    TokenizationAnalysis,
    profile_generation,
)
from api.base import Schema


@dataclass
class EngineConfig:
    pass


T = TypeVar("T", bound=EngineConfig)


@dataclass
class GenerationResult:
    input: str
    output: str
    label: Optional[str] = None
    json_schema: Optional[Schema] = None
    id: str = field(default_factory=lambda: str(uuid4()))
    generated_tokens: List[Token] = field(default_factory=list)
    top_tokens: List[List[Token]] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    perf_metrics: PerfMetrics = field(default_factory=PerfMetrics)
    metadata: GenerationMetadata = field(default_factory=GenerationMetadata)
    tokenization_analysis: TokenizationAnalysis = field(
        default_factory=TokenizationAnalysis
    )


class Engine(ABC, Generic[T]):
    def __init__(self, config: T):
        self.config = config
        self.total_usage = TokenUsage()

    @profile_generation
    def generate(
        self,
        prompt: str,
        schema: Schema,
    ) -> GenerationResult:
        schema = self.adapt_schema(schema)
        result = self._generate(prompt, schema)

        self.total_usage += result.token_usage
        result.json_schema = schema
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

    @abstractmethod
    def adapt_schema(self, schema: Schema) -> Schema:
        return schema

    def encode(self, text: str) -> Optional[List[int]]:
        return None

    def decode(self, ids: List[int]) -> Optional[str]:
        return None

    def convert_token_to_id(self, token: str) -> Optional[int]:
        res = self.encode(token)
        return res[0] if len(res) == 1 else None

    def convert_id_to_token(self, id: int) -> Optional[str]:
        res = self.decode([id])
        return res[0] if len(res) == 1 else None

    def count_tokens(self, text: str) -> int:
        res = self.encode(text)
        return len(res) if res else 0
