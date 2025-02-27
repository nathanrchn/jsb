import os
from typing import List
from argparse import ArgumentParser

from api.dataset import Dataset, DatasetConfig
from api.engine import Engine, GenerationResult
from engines.openai import OpenAIConfig, OpenAIEngine
from api.base import FormatPrompt, DEFAULT_FORMAT_PROMPT
from core.evaluator import evaluate, aggregate_scores, print_scores


def bench(
    engine: Engine, dataset: Dataset, prompt_fn: FormatPrompt = DEFAULT_FORMAT_PROMPT
) -> List[GenerationResult]:
    results = []
    for prompt, schema in dataset.iter(prompt_fn):
        schema = engine.adapt_schema(schema)
        result = engine.generate(prompt, schema)
        results.append(result)

    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--limit", type=int, required=False)
    args = parser.parse_args()

    engine = OpenAIEngine(
        OpenAIConfig(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    )
    dataset = Dataset(DatasetConfig(args.task, limit=args.limit))

    results = bench(engine, dataset)
    scores = evaluate(results)
    aggregated_scores = aggregate_scores(scores)
    print_scores(aggregated_scores)
