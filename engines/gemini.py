from api.engine import Engine, EngineConfig


class GeminiEngineConfig(EngineConfig):
    pass


class GeminiEngine(Engine):
    def __init__(self, config: GeminiEngineConfig):
        super().__init__(config)
