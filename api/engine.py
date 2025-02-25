from uuid import uuid4
from typing import List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from utils import (
    TokenUsage,
    Token,
    GenerationMetadata,
    PerfMetrics,
    TokenizationAnalysis,
)
from api.base import Schema

@dataclass
class EngineConfig:
    pass


@dataclass
class GenerationResponse:
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


class Engine(ABC):
    def __init__(self, config: EngineConfig):
        self.config = config
        self.total_usage = TokenUsage()

    @abstractmethod
    def generate(
        self,
        prompt: str,
        schema: Schema,
    ) -> GenerationResponse:
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
        return res[0] if len(res) == 1 else None

    def convert_id_to_token(self, id: int) -> Optional[str]:
        res = self.decode([id])
        return res[0] if len(res) == 1 else None

    def count_tokens(self, text: str) -> int:
        res = self.encode(text)
        return len(res) if res else 0
