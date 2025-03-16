import os
import sys
from tqdm import tqdm
from json import dumps
from dataclasses import asdict
from typing import List, Optional, Union

from core.engine import Engine
from core.dataset import Dataset, DatasetConfig
from core.evaluator import evaluate, print_scores
from core.types import FormatPrompt, GenerationResult
from core.utils import DEFAULT_FORMAT_PROMPT, disable_print, nanoid


def bench(
    engine: Engine,
    tasks: List[str],
    limit: Optional[int] = None,
    prompt_fn: Union[FormatPrompt, List[FormatPrompt]] = DEFAULT_FORMAT_PROMPT,
    close_engine: bool = True,
    save_results: bool = False,
) -> List[List[GenerationResult]]:
    id = nanoid()

    perf_metrics = []
    declared_coverage = []
    empirical_coverage = []

    if not isinstance(prompt_fn, list):
        prompt_fn = [prompt_fn] * len(tasks)

    all_results = []
    for task, pf in zip(tasks, prompt_fn):
        task_results = []
        dataset = Dataset(DatasetConfig(task, limit=limit))
        for prompt, schema in tqdm(
            dataset.iter(pf), total=limit or len(dataset), desc=task, file=sys.stdout
        ):
            with disable_print():
                schema = engine.adapt_schema(schema)
                result = engine.generate(prompt, schema, task)
                task_results.append(result)
        dc, ec, pm = evaluate(task_results)
        declared_coverage.append(dc)
        empirical_coverage.append(ec)
        perf_metrics.append(pm)
        all_results.append(task_results)

    print_scores(declared_coverage, empirical_coverage, perf_metrics, tasks)
    print(engine.total_usage)

    if close_engine:
        engine.close()

    if save_results:
        if not os.path.exists("results"):
            os.makedirs("results")

        with open(f"results/{id}.jsonl", "w") as f:
            for results in all_results:
                for result in results:
                    f.write(dumps(asdict(result)))

        print(f"Results saved to results/{id}.jsonl")

    return all_results
