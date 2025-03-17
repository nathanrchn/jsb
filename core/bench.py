import os
import sys
from tqdm import tqdm
from json import dumps
from dataclasses import asdict
from typing import List, Optional, Union

from core.engine import Engine
from core.dataset import Dataset, DatasetConfig
from core.evaluator import evaluate, print_scores
from core.types import FormatPrompt, GenerationState
from core.utils import DEFAULT_FORMAT_PROMPT, disable_print, nanoid


def bench(
    engine: Engine,
    tasks: List[str],
    limit: Optional[int] = None,
    prompt_fn: Union[FormatPrompt, List[FormatPrompt]] = DEFAULT_FORMAT_PROMPT,
    close_engine: bool = True,
    save_states: bool = False,
) -> List[List[GenerationState]]:
    id = nanoid()

    compliance = []
    perf_metrics = []
    declared_coverage = []
    empirical_coverage = []

    if not isinstance(prompt_fn, list):
        prompt_fn = [prompt_fn] * len(tasks)

    all_states = []
    for task, pf in zip(tasks, prompt_fn):
        task_states = []
        dataset = Dataset(DatasetConfig(task, limit=limit))
        for prompt, schema in tqdm(
            dataset.iter(pf), total=limit or len(dataset), desc=task, file=sys.stdout
        ):
            with disable_print():
                schema = engine.adapt_schema(schema)
                result = engine.generate(task, prompt, schema)
                task_states.append(result)
        dc, ec, cl, pm = evaluate(task_states)
        declared_coverage.append(dc)
        empirical_coverage.append(ec)
        compliance.append(cl)
        perf_metrics.append(pm)
        all_states.append(task_states)

    print_scores(declared_coverage, empirical_coverage, compliance, perf_metrics, tasks)
    print(engine.total_usage)

    if close_engine:
        engine.close()

    if save_states:
        if not os.path.exists("states"):
            os.makedirs("states")

        with open(f"states/{id}.jsonl", "w") as f:
            for states in all_states:
                for state in states:
                    f.write(f"{dumps(asdict(state))}\n")

        print(f"States saved to states/{id}.jsonl")

    return all_states
