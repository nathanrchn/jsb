from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from api.engine import Engine, EngineConfig

ENGINE_TO_CLASS: Dict[str, Type["Engine"]] = {}
ENGINE_TO_CONFIG: Dict[str, Type["EngineConfig"]] = {}


def register_engine(
    name: str, engine_class: Type["Engine"], config_class: Type["EngineConfig"]
):
    ENGINE_TO_CLASS[name] = engine_class
    ENGINE_TO_CONFIG[name] = config_class


def get_engine_class(name: str) -> Type["Engine"]:
    if name not in ENGINE_TO_CLASS:
        available_engines = ", ".join(ENGINE_TO_CLASS.keys())
        raise KeyError(f"Engine '{name}' not found. Available engines: {available_engines}")
    return ENGINE_TO_CLASS[name]


def get_engine_config(name: str) -> Type["EngineConfig"]:
    if name not in ENGINE_TO_CONFIG:
        available_engines = ", ".join(ENGINE_TO_CONFIG.keys())
        raise KeyError(f"Engine '{name}' not found. Available engines: {available_engines}")
    return ENGINE_TO_CONFIG[name]
