from json import loads
from dacite import from_dict, Config
from typing import Dict, List
from argparse import ArgumentParser

from core.types import GenerationState
from core.evaluator import evaluate, print_scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--states", type=str, required=True)
    args = parser.parse_args()

    dacite_config = Config(check_types=False)
    with open(args.states, "r") as f:
        states = [
            from_dict(GenerationState, loads(line), config=dacite_config) for line in f
        ]

    task_states: Dict[str, List[GenerationState]] = {}
    for state in states:
        if state.task not in task_states:
            task_states[state.task] = []
        task_states[state.task].append(state)

    compliance = []
    perf_metrics = []
    declared_coverage = []
    empirical_coverage = []
    for states in task_states.values():
        dc, ec, cl, pm = evaluate(states)

        compliance.append(cl)
        perf_metrics.append(pm)
        declared_coverage.append(dc)
        empirical_coverage.append(ec)

    print_scores(
        declared_coverage,
        empirical_coverage,
        compliance,
        perf_metrics,
        list(task_states.keys()),
    )
