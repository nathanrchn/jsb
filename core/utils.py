import os
import sys
from json import dumps
from omegaconf import OmegaConf
from contextlib import contextmanager
from typing import List, Optional, TypeVar, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from core.types import FormatPrompt


T = TypeVar("T")


def load_config(config_type: Type[T], config_path: str) -> T:
    config = OmegaConf.load(config_path)
    return OmegaConf.merge(config, OmegaConf.structured(config_type, config))


DEFAULT_FORMAT_PROMPT: "FormatPrompt" = (
    lambda schema: f" You need to generate a JSON object that matches the schema below.  Do not include the schema in the output and DIRECTLY return the JSON object without any additional information.  The schema is: {dumps(schema)}"
)


def safe_divide(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safely divides a by b, returning None if either input is None."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def safe_subtract(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safely subtracts b from a, returning None if either input is None."""
    if a is None or b is None:
        return None
    return a - b


def median(values: List[float]) -> float:
    if len(values) == 0:
        return None
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]


def detect_none(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


@contextmanager
def disable_print():
    stdout = sys.stdout
    stderr = sys.stderr
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout = stdout
        sys.stderr = stderr
