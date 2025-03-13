from dataclasses import dataclass

from .engine import EngineConfig
from .dataset import DatasetConfig


@dataclass
class Config:
    engine_config: EngineConfig
    dataset_config: DatasetConfig
