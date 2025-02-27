import json
from enum import IntEnum
from typing import List, Tuple
from fastjsonschema import compile
from prettytable import PrettyTable

from api.engine import GenerationResult, PerfMetrics


class EvaluationCode(IntEnum):
    MATCH = 0
    MISMATCH = 1
    SYNTAX_ERROR = 2
    SEMANTIC_ERROR = 3
    EMPTY_INPUT_OR_BAD_FORMAT = 4
    JSON_NOT_FOUND_ERROR = 5
    UNKNOWN_ERROR = 6


def evaluate(
    results: List[GenerationResult],
) -> Tuple[List[EvaluationCode], PerfMetrics]:
    scores = []

    for result in results:
        output = result.output
        schema = result.json_schema

        if schema is None:
            scores.append(EvaluationCode.JSON_NOT_FOUND_ERROR)
            continue

        if output is None:
            scores.append(EvaluationCode.EMPTY_INPUT_OR_BAD_FORMAT)
            continue

        try:
            json_object = json.loads(output)
        except (json.JSONDecodeError, RecursionError, ValueError) as e:
            scores.append(EvaluationCode.SYNTAX_ERROR)
            continue

        schema_validator = compile(schema)
        try:
            schema_validator(json_object)
        except Exception:
            scores.append(EvaluationCode.SEMANTIC_ERROR)
            continue

        scores.append(EvaluationCode.MATCH)

    average_ttft = float("inf")
    total_ttft = 0
    average_tpot = float("inf")
    total_tpot = 0
    average_tgt = float("inf")
    total_tgt = 0
    average_gct = float("inf")
    total_gct = 0

    for result in results:
        if result.perf_metrics.ttft is not None:
            average_ttft += result.perf_metrics.ttft
            total_ttft += 1
        if result.perf_metrics.tpot is not None:
            average_tpot += result.perf_metrics.tpot
            total_tpot += 1
        if result.perf_metrics.tgt is not None:
            average_tgt += result.perf_metrics.tgt
            total_tgt += 1
        if result.perf_metrics.gct is not None:
            average_gct += result.perf_metrics.gct
            total_gct += 1

    return scores, PerfMetrics(
        ttft=average_ttft / total_ttft,
        tpot=average_tpot / total_tpot,
        tgt=average_tgt / total_tgt,
        gct=average_gct / total_gct,
    )


def detect_inf(value: float) -> str:
    if value == float("inf"):
        return "n/a"
    return f"{value:.2f}"


def print_scores(
    scores: List[Tuple[List[EvaluationCode], PerfMetrics]], tasks: List[str]
) -> None:
    table = PrettyTable(
        [
            "Task",
            "Accuracy",
            "Time to first token (ms)",
            "Time per output token (ms)",
            "Grammar compilation time (s)",
            "Generation time (s)",
        ]
    )
    for task, (task_scores, perf_metrics) in zip(tasks, scores):
        accuracy = sum([score == EvaluationCode.MATCH for score in task_scores]) / len(
            task_scores
        )
        table.add_row(
            [
                task,
                f"{accuracy:.2%}",
                detect_inf(perf_metrics.ttft),
                detect_inf(perf_metrics.tpot),
                detect_inf(perf_metrics.gct),
                detect_inf(perf_metrics.tgt),
            ]
        )
    print(table)
