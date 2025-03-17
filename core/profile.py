from time import time
from functools import wraps
from typing import Callable, Dict, Any, TYPE_CHECKING

from core.types import PerfMetrics, DecodingStatusCode

if TYPE_CHECKING:
    from core.engine import Engine, GenerationState


def profile_generation(
    generate: Callable[["Engine", str, str, Dict[str, Any]], "GenerationState"],
) -> Callable[["Engine", str, str, Dict[str, Any]], "GenerationState"]:
    @wraps(generate)
    def wrapper(
        engine: "Engine", task: str, prompt: str, schema: Dict[str, Any]
    ) -> "GenerationState":
        gen_start_time: float = time()
        state: "GenerationState" = generate(engine, task, prompt, schema)
        gen_end_time: float = time()

        if state.metadata.decoding_status.code == DecodingStatusCode.UNKOWN_ERROR:
            return state

        perf_metrics: PerfMetrics = PerfMetrics.from_timestamps(
            start_time=gen_start_time,
            grammar_compilation_end_time=state.metadata.grammar_compilation_end_time,
            first_token_arrival_time=state.metadata.first_token_arrival_time,
            end_time=gen_end_time,
            num_output_tokens=state.token_usage.output_tokens,
        )

        state.perf_metrics = perf_metrics
        return state

    return wrapper
