from typing import List
from argparse import ArgumentParser

from api.dataset import Dataset
from api.engine import Engine, GenerationResponse


def bench(engine: Engine, dataset: Dataset) -> List[GenerationResponse]:
    responses = []
    for schema in dataset:
        schema = engine.adapt_schema(schema)
        prompt = dataset.schema_to_prompt(schema)
        response = engine.generate(prompt, schema)
        responses.append(response)

    return responses


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    bench()
