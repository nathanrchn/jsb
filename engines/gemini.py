from typing import List, Optional

from core.registry import register_engine
from engines.openai import OpenAIEngine, OpenAIConfig


class GeminiEngine(OpenAIEngine):
    def __init__(self, config: OpenAIConfig):
        config.api_key_variable_name = "GEMINI_API_KEY"
        config.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        super().__init__(config)

    def encode(self, _: str) -> Optional[List[int]]:
        return None

    def decode(self, _: List[int]) -> Optional[str]:
        return None

    @property
    def max_context_length(self) -> int:
        max_context_length_dict = {
            "models/gemini-2.0-flash": 1_048_576,
            "models/gemini-2.0-flash-lite": 1_048_576,
            "models/gemini-1.5-flash": 1_048_576,
            "models/gemini-1.5-flash-8b": 1_048_576,
            "models/gemini-1.5-pro": 2_097_152,
        }
        return max_context_length_dict[self.config.model]


register_engine("gemini", GeminiEngine, OpenAIConfig)
