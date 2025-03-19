from time import time
from functools import wraps
from typing import Callable, Dict, Any, TYPE_CHECKING

from core.types import PerfMetrics

if TYPE_CHECKING:
    from core.engine import Engine, GenerationOutput


def profile_generation(
    generate: Callable[["Engine", str, str, Dict[str, Any]], "GenerationOutput"],
) -> Callable[["Engine", str, str, Dict[str, Any]], "GenerationOutput"]:
    @wraps(generate)
    def wrapper(
        engine: "Engine", task: str, prompt: str, schema: Dict[str, Any]
    ) -> "GenerationOutput":
        gen_start_time: float = time()
        output: "GenerationOutput" = generate(engine, task, prompt, schema)
        gen_end_time: float = time()

        perf_metrics: PerfMetrics = PerfMetrics.from_timestamps(
            start_time=gen_start_time,
            grammar_compilation_end_time=output.metadata.grammar_compilation_end_time,
            first_token_arrival_time=output.metadata.first_token_arrival_time,
            end_time=gen_end_time,
            num_output_tokens=output.token_usage.output_tokens,
        )

        output.perf_metrics = perf_metrics
        return output

    return wrapper
