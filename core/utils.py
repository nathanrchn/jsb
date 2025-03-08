from omegaconf import OmegaConf
from typing import Optional, TypeVar, Type


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


T = TypeVar("T")


def load_config(config_type: Type[T], config_path: str) -> T:
    config = OmegaConf.load(config_path)
    return OmegaConf.structured(config_type, config)
