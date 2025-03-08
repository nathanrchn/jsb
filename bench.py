from tqdm import tqdm
from typing import List, Optional
from argparse import ArgumentParser

from api.engine import Engine
from core.utils import load_config
from api.dataset import Dataset, DatasetConfig
from core.evaluator import evaluate, print_scores
from api.base import FormatPrompt, DEFAULT_FORMAT_PROMPT
from core.registry import get_engine_class, get_engine_config


def bench(
    engine: Engine,
    tasks: List[str],
    limit: Optional[int] = None,
    prompt_fn: FormatPrompt = DEFAULT_FORMAT_PROMPT,
) -> None:
    declared_coverage = []
    empirical_coverage = []
    perf_metrics = []

    for task in tasks:
        task_results = []
        dataset = Dataset(DatasetConfig(task, limit=limit))
        for prompt, schema in tqdm(
            dataset.iter(prompt_fn), total=limit or len(dataset), desc=task
        ):
            schema = engine.adapt_schema(schema)
            result = engine.generate(prompt, schema)
            task_results.append(result)
        dc, ec, pm = evaluate(task_results)
        declared_coverage.append(dc)
        empirical_coverage.append(ec)
        perf_metrics.append(pm)

    print_scores(declared_coverage, empirical_coverage, perf_metrics, tasks)
    print(engine.total_usage)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--limit", type=int, required=False)
    args = parser.parse_args()

    engine = get_engine_class(args.engine)(
        load_config(get_engine_config(args.engine), args.config)
    )

    bench(engine, args.tasks.split(","), args.limit, DEFAULT_FORMAT_PROMPT)
