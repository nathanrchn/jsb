from argparse import ArgumentParser

from api.engine import Engine
from api.dataset import Dataset


def bench(engine: Engine, dataset: Dataset):
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    bench()
