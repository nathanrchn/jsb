from enum import Enum
from uuid import uuid4
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional

from core.utils import safe_divide, safe_subtract


Schema = Dict[str, Any]
FormatPrompt = Callable[[Schema], str]


class CompileStatusCode(Enum):
    TBD = -1
    OK = 0
    UNSUPPORTED_SCHEMA = 1
    RUNTIME_GRAMMAR_ERROR = 2
    API_BAD_RESPONSE = 3
    PROMPT_TOO_LONG = 4
    COMPILE_TIMEOUT = 5
    RUNTIME_TIMEOUT = 6
    UNKOWN_ERROR = 7


class DecodingStatusCode(Enum):
    TBD = -1
    OK = 0
    EXCEEDING_MAX_CTX = 1
    DECODING_TIMEOUT = 2
    BAD_API_RESPONSE = 3
    UNKOWN_ERROR = 4


@dataclass
class CompileStatus:
    code: CompileStatusCode = CompileStatusCode.TBD
    message: Optional[str] = None


@dataclass
class DecodingStatus:
    code: DecodingStatusCode = DecodingStatusCode.TBD
    message: Optional[str] = None


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    ff_output_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            ff_output_tokens=self.ff_output_tokens + other.ff_output_tokens,
        )

    def __repr__(self) -> str:
        return (
            f"token usage: {self.input_tokens:,} input, {self.output_tokens:,} output."
        )


@dataclass
class Token:
    id: Optional[int] = None
    text: Optional[str] = None
    logprob: Optional[float] = None
    unmasked_logprob: Optional[float] = None


@dataclass
class GenerationMetadata:
    system_fingerprint: Optional[str] = None
    first_token_arrival_time: Optional[float] = None
    grammar_compilation_end_time: Optional[float] = None
    compile_status: Optional[CompileStatus] = field(default_factory=CompileStatus)
    decoding_status: Optional[DecodingStatus] = field(default_factory=DecodingStatus)


@dataclass
class PerfMetrics:
    ttft: Optional[float] = None  # Time to first token in s
    tpot: Optional[float] = None  # Time per output token in ms
    tgt: Optional[float] = None  # Total generation time in s
    gct: Optional[float] = None  # Grammar compilation time in s
    prft: Optional[float] = None  # Prefilling time in s
    peak_memory: Optional[float] = None  # In MB

    @classmethod
    def from_timestamps(
        cls,
        start_time: float,
        grammar_compilation_end_time: Optional[float],
        first_token_arrival_time: Optional[float],
        end_time: float,
        num_output_tokens: int,
    ):
        ttft = safe_subtract(first_token_arrival_time, start_time)
        tpot = safe_divide(
            safe_subtract(end_time, first_token_arrival_time),
            safe_subtract(num_output_tokens, 1),
        )
        tgt = safe_subtract(end_time, start_time)
        gct = safe_subtract(grammar_compilation_end_time, start_time)
        prft = safe_subtract(first_token_arrival_time, grammar_compilation_end_time)
        return cls(
            ttft=ttft,
            tpot=tpot * 1000 if tpot is not None else None,
            tgt=tgt,
            gct=gct,
            prft=prft,
        )


@dataclass
class GenerationResult:
    input: str
    output: str
    json_schema: Optional[Schema] = None
    id: str = field(default_factory=lambda: str(uuid4()))
    generated_tokens: List[Token] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    perf_metrics: PerfMetrics = field(default_factory=PerfMetrics)
    metadata: GenerationMetadata = field(default_factory=GenerationMetadata)
