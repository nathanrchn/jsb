from time import time
from functools import wraps
from typing import Callable, Dict, Any, TYPE_CHECKING

from core.types import PerfMetrics

if TYPE_CHECKING:
    from api.engine import Engine, GenerationResult

COMPILATION_TIMEOUT = 40
GENERATION_TIMEOUT = 60


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
