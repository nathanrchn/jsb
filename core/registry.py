from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from api.engine import Engine, EngineConfig

ENGINE_TO_CLASS: Dict[str, Type["Engine"]] = {}
ENGINE_TO_CONFIG: Dict[str, Type["EngineConfig"]] = {}


def register_engine(
    name: str, engine_class: Type["Engine"], config_class: Type["EngineConfig"]
):
    global ENGINE_TO_CLASS, ENGINE_TO_CONFIG
    
    ENGINE_TO_CLASS[name] = engine_class
    ENGINE_TO_CONFIG[name] = config_class
