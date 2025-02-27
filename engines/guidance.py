import stopit
from time import time
from dataclasses import dataclass
from guidance import models, json
from typing import List, Literal, Optional

from api.base import Schema
from api.engine import Engine, EngineConfig, GenerationResult
from utils import (
    TokenUsage,
    GenerationMetadata,
    CompileStatus,
    CompileStatusCode,
    DecodingStatus,
    DecodingStatusCode,
)


COMPILATION_TIMEOUT = 30
GENERATION_TIMEOUT = 60


@dataclass
class GuidanceConfig(EngineConfig):
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    model_engine: Literal["llamacpp"] = "llamacpp"


class GuidanceEngine(Engine[GuidanceConfig]):
    def __init__(self, config: GuidanceConfig):
        super().__init__(config)

        match self.config.model_engine:
            case "llamacpp":
                from llama_cpp import Llama

                self.model = Llama.from_pretrained(self.config.model)

                self.guidance_model_state = models.LlamaCpp(
                    self.model, echo=False, caching=False
                )

                self.tokenizer = self.guidance_model_state.engine.tokenizer
            case _:
                raise ValueError(
                    f"model engine {self.config.model_engine} not supported"
                )

    def _generate(self, prompt: str, schema: Schema) -> GenerationResult:
        metadata = GenerationMetadata()

        try:
            with stopit.ThreadingTimeout(COMPILATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    generation_op = json(
                        schema=schema,
                        name="generated_object",
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        top_p=self.config.top_p,
                    )
                    metadata.grammar_compilation_end_time = time()
                    metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.COMPILE_TIMEOUT,
                    message="Schema compilation timed out",
                )
                return GenerationResult(input=prompt, output="", metadata=metadata)

        except Exception as e:
            metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
            return GenerationResult(input=prompt, output="", metadata=metadata)

        try:
            with stopit.ThreadingTimeout(GENERATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    state_iterator = (
                        self.guidance_model_state.stream() + prompt + generation_op
                    )
                    for i, state in enumerate(state_iterator):
                        if i == 0:
                            metadata.first_token_arrival_time = time()
                    final_state = state

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                metadata.decoding_status = DecodingStatus(
                    code=DecodingStatusCode.DECODING_TIMEOUT,
                    message="Generation timed out",
                )
                return GenerationResult(input=prompt, output="", metadata=metadata)

        except Exception as e:
            metadata.decoding_status = DecodingStatus(
                code=DecodingStatusCode.UNKOWN_ERROR, message=str(e)
            )
            return GenerationResult(input=prompt, output="", metadata=metadata)

        try:
            generation = final_state["generated_object"]
            metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)
        except KeyError:
            metadata.decoding_status = DecodingStatus(
                code=DecodingStatusCode.UNKOWN_ERROR,
                message="Failed to extract generated object",
            )
            generation = ""

        input_tokens = len(self.encode(prompt))
        output_tokens = len(self.encode(generation)) if generation else 0

        token_usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

        self.guidance_model_state.reset()

        return GenerationResult(
            input=prompt,
            output=generation,
            json_schema=schema,
            token_usage=token_usage,
            metadata=metadata,
        )

    def encode(self, text: str) -> Optional[List[int]]:
        return self.tokenizer.encode(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> Optional[str]:
        return self.tokenizer.decode(ids).decode("utf-8")

    @property
    def max_context_length(self) -> int:
        return self.model.n_ctx()
