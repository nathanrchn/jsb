from json import loads
from dataclasses import dataclass
from datasets import load_dataset
from typing import Callable, Iterator

from api.base import Schema, FormatPrompt, DEFAULT_FORMAT_PROMPT

DATASET_SCHEMA_COLUMN = "json_schema"
DATASET_HUGGINGFACE_PATH = "epfl-dlab/JSONSchemaBench"


@dataclass
class PromptConfig:
    format_prompt: FormatPrompt = DEFAULT_FORMAT_PROMPT


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

    def filter(self, filter_fn: Callable[[Schema], bool]) -> None:
        self.dataset = self.dataset.filter(
            lambda x: filter_fn(loads(x[DATASET_SCHEMA_COLUMN]))
        )

    def map(self, map_fn: Callable[[Schema], Schema]) -> None:
        self.dataset = self.dataset.map(
            lambda x: map_fn(loads(x[DATASET_SCHEMA_COLUMN]))
        )

    def shuffle(self) -> None:
        self.dataset = self.dataset.shuffle()

    def __iter__(self) -> Iterator[Schema]:
        for item in self.dataset:
            yield item

    def schema_to_prompt(self, schema: Schema) -> str:
        return self.config.prompt_config.format_prompt(schema)
