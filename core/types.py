from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field


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
        return (
            f"token usage: {self.input_tokens:,} input, {self.output_tokens:,} output."
        )


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
    prft: Optional[float] = None  # Prefilling time in s
    peak_memory: Optional[float] = None  # In MB
