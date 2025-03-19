import os
import sys
from tqdm import tqdm
from json import dumps
from dataclasses import asdict
from typing import List, Optional, Union

from core.engine import Engine
from core.dataset import Dataset, DatasetConfig
from core.evaluator import evaluate, print_scores
from core.types import FormatPrompt, GenerationOutput
from core.utils import DEFAULT_FORMAT_PROMPT, disable_print, nanoid, safe_min


def bench(
    engine: Engine,
    tasks: List[str],
    limit: Optional[int] = None,
    prompt_fn: Union[FormatPrompt, List[FormatPrompt]] = DEFAULT_FORMAT_PROMPT,
    close_engine: bool = True,
    save_outputs: bool = False,
) -> List[List[GenerationOutput]]:
    """Benchmarks an engine with specified tasks and datasets.

    :param engine: Engine
        The engine to benchmark.
    :param tasks: List[str]
        The tasks to benchmark.
    :param limit: Optional[int]
        The limit on the number of samples to benchmark.
    :param prompt_fn: Union[FormatPrompt, List[FormatPrompt]]
        The function(s) to format the schema into a prompt. If a single
        function is provided, it will be used for all tasks. If a list of
        functions is provided, each function will be used for the corresponding
        task. The default format prompt is:
            You need to generate a JSON object that matches the schema below.
            Do not include the schema in the output and DIRECTLY return the
            JSON object without any additional information. The schema is:
            {dumps(schema)}
    :param close_engine: bool
        Whether to close the engine after the benchmark.
    :param save_outputs: bool
        Whether to save the generation outputs after the benchmark.

    :return: List[List[GenerationOutput]]
        The generation outputs for each sample for each task.
    """
    id = nanoid()

    if not isinstance(prompt_fn, list):
        prompt_fn = [prompt_fn] * len(tasks)

    all_outputs = []
    for task, pf in zip(tasks, prompt_fn):
        task_outputs = []
        dataset = Dataset(DatasetConfig(task, limit=limit))
        for prompt, schema in tqdm(
            dataset.iter(pf),
            total=safe_min(len(dataset), limit),
            desc=task,
            file=sys.stdout,
        ):
            with disable_print():
                schema = engine.adapt_schema(schema)
                result = engine.generate(task, prompt, schema)
                task_outputs.append(result)
        all_outputs.append(task_outputs)

    compliance = []
    perf_metrics = []
    declared_coverage = []
    empirical_coverage = []
    for outputs in all_outputs:
        dc, ec, cl, pm = evaluate(outputs)

        compliance.append(cl)
        perf_metrics.append(pm)
        declared_coverage.append(dc)
        empirical_coverage.append(ec)

    print_scores(declared_coverage, empirical_coverage, compliance, perf_metrics, tasks)
    print(engine.total_usage)

    if save_outputs:
        if not os.path.exists("outputs"):
            os.makedirs("outputs")

        if not os.path.exists(f"outputs/{engine.name}"):
            os.makedirs(f"outputs/{engine.name}")

        with open(f"outputs/{engine.name}/{id}.jsonl", "w") as f:
            for outputs in all_outputs:
                for output in outputs:
                    f.write(f"{dumps(asdict(output))}\n")

        print(f"Outputs saved to outputs/{engine.name}/{id}.jsonl")

    if close_engine:
        engine.close()

    return all_outputs
