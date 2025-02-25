from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from utils import (
    TokenUsage,
    Token,
    GenerationMetadata,
    PerfMetrics,
    TokenizationAnalysis,
)


@dataclass
class EngineConfig:
    pass


@dataclass
class GenerationResponse:
    input: str
    output: str
    id: Optional[str] = None
    json_schema: Optional[Dict[str, Any]] = None
    label: Optional[str] = None
    generated_tokens: List[Token] = field(default_factory=list)
    top_tokens: List[List[Token]] = field(default_factory=list)
    metadata: GenerationMetadata = field(default_factory=GenerationMetadata)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    perf_metrics: PerfMetrics = field(default_factory=PerfMetrics)
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
        schema: Optional[Dict[str, Any]] = None,
    ) -> GenerationResponse:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_context_length(self) -> int:
        raise NotImplementedError

    def adapt_schema(
        self, schema: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
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
