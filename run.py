from argparse import ArgumentParser

from core.bench import bench
from core.utils import load_config, disable_print
from core.registry import ENGINE_TO_CLASS, ENGINE_TO_CONFIG


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--limit", type=int, required=False)
    parser.add_argument("--save_states", action="store_true")
    args = parser.parse_args()

    with disable_print():
        engine = ENGINE_TO_CLASS[args.engine](
            load_config(ENGINE_TO_CONFIG[args.engine], args.config)
        )

    bench(
        engine=engine,
        tasks=args.tasks.split(","),
        limit=args.limit,
        save_states=args.save_states,
    )
