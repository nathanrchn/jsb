from json import loads, dumps
from dataclasses import dataclass
from datasets import load_dataset
from typing import Dict, Any, Callable


DATASET_SCHEMA_COLUMN = "json_schema"
DATASET_HUGGINGFACE_PATH = "epfl-dlab/JSONSchemaBench"


@dataclass
class PromptConfig:
    pass


@dataclass
class DatasetConfig:
    dataset_name: str
    prompt_config: PromptConfig


class Dataset:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.dataset = load_dataset(
            path=DATASET_HUGGINGFACE_PATH, split=config.dataset_name
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def filter(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> None:
        self.dataset = self.dataset.filter(
            lambda x: filter_fn(loads(x[DATASET_SCHEMA_COLUMN]))
        )

    def map(self, map_fn: Callable[[Dict[str, Any]], Any]) -> None:
        self.dataset = self.dataset.map(
            lambda x: map_fn(loads(x[DATASET_SCHEMA_COLUMN]))
        )

    def shuffle(self) -> None:
        self.dataset = self.dataset.shuffle()

    def __iter__(self):
        for item in self.dataset:
            yield item

    def schema_to_prompt(self, schema: Dict[str, Any]) -> str:
        return dumps(schema)
