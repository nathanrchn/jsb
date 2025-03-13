from tqdm import tqdm
from typing import List, Optional, Union

from core.engine import Engine
from core.types import FormatPrompt
from core.utils import DEFAULT_FORMAT_PROMPT
from core.dataset import Dataset, DatasetConfig
from core.evaluator import evaluate, print_scores


def bench(
    engine: Engine,
    tasks: List[str],
    limit: Optional[int] = None,
    prompt_fn: Union[FormatPrompt, List[FormatPrompt]] = DEFAULT_FORMAT_PROMPT,
    close_engine: bool = True,
) -> None:
    declared_coverage = []
    empirical_coverage = []
    perf_metrics = []

    if isinstance(prompt_fn, FormatPrompt):
        prompt_fn = [prompt_fn] * len(tasks)

    for task, pf in zip(tasks, prompt_fn):
        task_results = []
        dataset = Dataset(DatasetConfig(task, limit=limit))
        for prompt, schema in tqdm(
            dataset.iter(pf), total=limit or len(dataset), desc=task
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

    if close_engine:
        engine.close()
