import stopit
from time import time
from typing import List, Optional
from dataclasses import dataclass

from core.registry import register_engine
from engines.llama_cpp import LlamaCppConfig
from core.engine import Engine, EngineConfig
from core.evaluator import is_json_schema_valid
from core.types import (
    Schema,
    CompileStatus,
    DecodingStatus,
    GenerationState,
    CompileStatusCode,
    DecodingStatusCode,
)


GENERATION_TIMEOUT = 60
COMPILATION_TIMEOUT = 30


@dataclass
class GuidanceConfig(EngineConfig):
    model_engine_config: LlamaCppConfig
    max_tokens: int = 100000000


class GuidanceEngine(Engine[GuidanceConfig]):
    def __init__(self, config: GuidanceConfig):
        super().__init__(config)

        from llama_cpp import Llama
        from guidance.models import LlamaCpp

        self.model = Llama.from_pretrained(
            self.config.model_engine_config.model,
            filename=self.config.model_engine_config.filename,
            n_ctx=self.config.model_engine_config.n_ctx,
            verbose=self.config.model_engine_config.verbose,
            n_gpu_layers=self.config.model_engine_config.n_gpu_layers,
        )

        self.guidance_model_state = LlamaCpp(self.model, echo=False, caching=False)

        self.tokenizer = self.guidance_model_state.engine.tokenizer

    def _generate(self, state: GenerationState) -> None:
        from guidance import json as guidance_json

        try:
            with stopit.ThreadingTimeout(COMPILATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    generation_op = guidance_json(
                        schema=state.schema,
                        name="generated_object",
                        temperature=self.config.model_engine_config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    state.metadata.grammar_compilation_end_time = time()
                    state.metadata.compile_status = CompileStatus(
                        code=CompileStatusCode.OK
                    )

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                state.metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.COMPILE_TIMEOUT,
                    message="Schema compilation timed out",
                )
                return

        except Exception as e:
            state.metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
            return

        try:
            with stopit.ThreadingTimeout(GENERATION_TIMEOUT) as to_ctx_mgr:
                from time import sleep

                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    state_iterator = (
                        self.guidance_model_state.stream() + state.input + generation_op
                    )
                    for i, guidance_state in enumerate(state_iterator):
                        if i == 0:
                            state.metadata.first_token_arrival_time = time()
                        sleep(5)

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                state.metadata.decoding_status = DecodingStatus(
                    code=DecodingStatusCode.DECODING_TIMEOUT,
                    message="Generation timed out",
                )

                # unset the first token arrival time avoid false performance metrics
                state.metadata.first_token_arrival_time = None
                self.guidance_model_state.engine.model_obj.reset()
                return

        except Exception as e:
            state.metadata.decoding_status = DecodingStatus(
                code=DecodingStatusCode.UNKOWN_ERROR, message=str(e)
            )
            self.guidance_model_state.engine.model_obj.reset()
            return

        try:
            generation = guidance_state["generated_object"]
            state.metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)
        except KeyError:
            state.metadata.decoding_status = DecodingStatus(
                code=DecodingStatusCode.UNKOWN_ERROR,
                message="Failed to extract generated object",
            )
            generation = ""

        state.output = generation
        state.token_usage.output_tokens = self.count_tokens(generation)

        self.guidance_model_state.engine.model_obj.reset()
        return

    def encode(self, text: str) -> Optional[List[int]]:
        return self.tokenizer.encode(text.encode("utf-8"))

    def decode(self, ids: List[int]) -> Optional[str]:
        return self.tokenizer.decode(ids).decode("utf-8")

    def adapt_schema(self, schema: Schema) -> Schema:
        if "type" not in schema:
            schema["type"] = "object"

        if not is_json_schema_valid(schema):
            print("The JSON schema after adaptation is no longer valid.")
        return schema

    @property
    def max_context_length(self) -> int:
        return self.model.n_ctx()

    def close(self):
        self.model.close()


register_engine("guidance", GuidanceEngine, GuidanceConfig)
