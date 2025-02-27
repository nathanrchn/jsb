import json
from enum import IntEnum
from typing import List, Dict
from fastjsonschema import compile
from prettytable import PrettyTable

from api.engine import GenerationResult


class EvaluationCode(IntEnum):
    MATCH = 0
    MISMATCH = 1
    SYNTAX_ERROR = 2
    SEMANTIC_ERROR = 3
    EMPTY_INPUT_OR_BAD_FORMAT = 4
    JSON_NOT_FOUND_ERROR = 5
    UNKNOWN_ERROR = 6


def evaluate(results: List[GenerationResult]) -> List[EvaluationCode]:
    scores = []

    for result in results:
        output = result.output
        schema = result.json_schema

        if schema is None:
            scores.append(EvaluationCode.JSON_NOT_FOUND_ERROR)

        try:
            if output is None:
                scores.append(EvaluationCode.EMPTY_INPUT_OR_BAD_FORMAT)
                continue

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

    return scores


def aggregate_scores(scores: List[EvaluationCode]) -> Dict[EvaluationCode, int]:
    return {code: scores.count(code) for code in EvaluationCode}


def print_scores(scores: Dict[EvaluationCode, int]) -> None:
    table = PrettyTable(["Code", "Count"])
    for code, count in scores.items():
        table.add_row([code.name, count])
    print(table)
