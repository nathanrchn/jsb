import os
from tqdm import tqdm
from argparse import ArgumentParser

from api.engine import Engine
from api.dataset import Dataset, DatasetConfig
from engines.openai import OpenAIConfig, OpenAIEngine
from api.base import FormatPrompt, DEFAULT_FORMAT_PROMPT
from core.evaluator import evaluate, aggregate_scores, print_scores


def bench(
    engine: Engine, dataset: Dataset, prompt_fn: FormatPrompt = DEFAULT_FORMAT_PROMPT
) -> None:
    results = []
    for prompt, schema in tqdm(
        dataset.iter(prompt_fn), total=dataset.config.limit or len(dataset)
    ):
        schema = engine.adapt_schema(schema)
        result = engine.generate(prompt, schema)
        results.append(result)

    scores = evaluate(results)
    aggregated_scores = aggregate_scores(scores)
    print_scores(aggregated_scores)


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

    bench(engine, dataset)
