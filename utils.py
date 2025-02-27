from time import time
from enum import Enum
from functools import wraps
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, TYPE_CHECKING, Type

from engines.openai import OpenAIEngine
from engines.guidance import GuidanceEngine

if TYPE_CHECKING:
    from api.engine import Engine, GenerationResult

COMPILATION_TIMEOUT = 40
GENERATION_TIMEOUT = 60


ENGINE_TO_CLASS: Dict[str, Type[Engine]] = {
    "openai": OpenAIEngine,
    "guidance": GuidanceEngine,
}


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


class JsonSchemaMatchingCode(Enum):
    TBD = -1
    MATCH = 0
    SYNTAX_ERROR = 1
    SEMANTIC_ERROR = 2
    JSON_NOT_FOUND_ERROR = 3
    # INVALID_REF_JSON_SCHEMA_ERROR = -1
    UNKOWN_ERROR = 4
    SKIPPED = 5
    EMPTY_INPUT_OR_BAD_FORMAT = 6


class ExactMatchStatusCode(Enum):
    TBD = -1
    MATCH = 0
    MISMATCH = 1
    EMPTY_INPUT_OR_BAD_FORMAT = 2


@dataclass
class CompileStatus:
    code: CompileStatusCode = CompileStatusCode.TBD
    message: str = "unknown"


@dataclass
class DecodingStatus:
    code: DecodingStatusCode = DecodingStatusCode.TBD
    message: str = "unknown"


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
        return f"token usage: {self.input_tokens:,} input, {self.output_tokens:,} output."


@dataclass
class TokenDiff:
    index: int
    original_id: int
    original_token: str
    encoded_id: int
    encoded_token: str


@dataclass
class TokenizationAnalysis:
    match: bool = False
    original_length: int = 0
    encoded_length: int = 0
    num_differing_tokens: int = 0
    first_divergence_index: Optional[int] = None
    differing_token_indices: List[int] = field(default_factory=list)
    token_diffs: List[TokenDiff] = field(default_factory=list)


@dataclass
class Token:
    id: int
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
    prft: Optional[float] = None  #  prefilling time in s
    peak_memory: Optional[float] = None  # in MB

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
class Conversation:
    system_message: Optional[Dict[str, str]] = None
    user_messages: Optional[List[Dict[str, str]]] = []

    def to_messages(self) -> List[Dict[str, str]]:
        messages = []
        if self.system_message:
            messages.append(self.system_message)
        if self.user_messages:
            messages.extend(self.user_messages)
        return messages


def safe_divide(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safely divides a by b, returning None if either input is None."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def safe_subtract(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safely subtracts b from a, returning None if either input is None."""
    if a is None or b is None:
        return None
    return a - b


def profile_generation(
    generate: Callable[["Engine", str, Dict[str, Any]], "GenerationResult"],
) -> Callable[["Engine", str, Dict[str, Any]], "GenerationResult"]:
    @wraps(generate)
    def wrapper(
        engine: "Engine", prompt: str, schema: Dict[str, Any]
    ) -> "GenerationResult":
        gen_start_time: float = time()
        result: "GenerationResult" = generate(engine, prompt, schema)
        gen_end_time: float = time()

        perf_metrics: PerfMetrics = PerfMetrics.from_timestamps(
            start_time=gen_start_time,
            grammar_compilation_end_time=result.metadata.grammar_compilation_end_time,
            first_token_arrival_time=result.metadata.first_token_arrival_time,
            end_time=gen_end_time,
            num_output_tokens=result.token_usage.output_tokens,
        )

        result.perf_metrics = perf_metrics
        return result

    return wrapper
