from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Generic

from core.types import (
    Schema,
    TokenUsage,
    GenerationState,
)
from core.profile import profile_generation


@dataclass
class EngineConfig:
    pass


T = TypeVar("T", bound=EngineConfig)


class Engine(ABC, Generic[T]):
    def __init__(self, config: T):
        """Defines the interface that should be implemented by all engines.
        Engines are assumed to take a schema and generate a JSON object that
        matches the schema.

        :param config: EngineConfig
            Configuration for the engine. This config is passed to the
            engine constructor and is used to configure the engine.
        """

        self.config = config
        self.total_usage = TokenUsage()

    @profile_generation
    def generate(
        self,
        task: str,
        prompt: str,
        schema: Schema,
    ) -> GenerationState:
        """Generates a JSON object that matches the schema.

        This method is used to generate a JSON object that matches the schema.
        It is a wrapper around the `_generate` method.

        :param task: str
            The task to generate the JSON object for.
        :param prompt: str
            The prompt to generate the JSON object for.
        :param schema: Schema
            The schema to generate the JSON object for.
        :return: GenerationState
            The generation state.
        """

        schema = self.adapt_schema(schema)
        state = GenerationState(task=task, input=prompt, output="", schema=schema)
        state.token_usage.input_tokens = self.count_tokens(state.input)

        self._generate(state)

        self.total_usage += state.token_usage
        return state

    @abstractmethod
    def _generate(
        self,
        state: GenerationState,
    ) -> None:
        """The method that should be implemented by all engines. It takes
        a generation state and modifies it in place.

        :param state: GenerationState
            The generation state.
        :return: None
            The generation state is modified in place.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """The maximum context length of the engine.

        :return: int
            The maximum context length.
        """
        raise NotImplementedError

    def adapt_schema(self, schema: Schema) -> Schema:
        """Adapts the schema to the engine. This should be implemented if the
        engine needs to modify the schema in some way before generating.

        :param schema: Schema
            The schema to adapt.
        :return: Schema
            The adapted schema.
        """
        return schema

    def encode(self, text: str) -> Optional[List[int]]:
        """Encodes a text string into a list of tokens.

        :param text: str
            The text to encode.
        :return: Optional[List[int]]
            The encoded tokens.
        """
        return None

    def decode(self, ids: List[int]) -> Optional[str]:
        """Decodes a list of tokens into a text string.

        :param ids: List[int]
            The tokens to decode.
        :return: Optional[str]
            The decoded text.
        """
        return None

    def convert_token_to_id(self, token: str) -> Optional[int]:
        """Converts a token to an id.

        :param token: str
            The token to convert.
        :return: Optional[int]
            The id of the token.
        """
        res = self.encode(token)
        return res[0] if res else None

    def convert_id_to_token(self, id: int) -> Optional[str]:
        """Converts an id to a token.

        :param id: int
            The id to convert.
        :return: Optional[str]
            The token.
        """
        res = self.decode([id])
        return res[0] if res else None

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in a text string. This can be
        implemented by the engines if they don't provide a tokenizer.

        :param text: str
            The text to count the tokens in.
        :return: int
            The number of tokens in the text.
        """
        res = self.encode(text)
        return len(res) if res else 0
    
    def close(self):
        # Close the model and the sampler for LlamaCpp
        # https://github.com/abetlen/llama-cpp-python/issues/1610
        if hasattr(self.model, "_sampler"):
            self.model._sampler.close()
            self.model.close()

