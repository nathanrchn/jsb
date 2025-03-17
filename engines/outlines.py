import stopit
from time import time
from json import dumps
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from core.registry import register_engine
from engines.llama_cpp import LlamaCppConfig
from core.engine import Engine, EngineConfig
from core.utils import COMPILATION_TIMEOUT, GENERATION_TIMEOUT
from core.types import (
    Token,
    Schema,
    TokenUsage,
    CompileStatus,
    DecodingStatus,
    GenerationState,
    CompileStatusCode,
    DecodingStatusCode,
    GenerationMetadata,
)

if TYPE_CHECKING:
    from outlines.generate.api import SequenceGeneratorAdapter


@dataclass
class OutlinesConfig(EngineConfig):
    model_engine_config: LlamaCppConfig
    grammar_cache_enabled: bool = False
    hf_tokenizer_id: Optional[str] = None


class OutlinesEngine(Engine[OutlinesConfig]):
    def __init__(self, config: OutlinesConfig):
        super().__init__(config)

        from llama_cpp.llama_tokenizer import LlamaHFTokenizer
        from outlines.models import llamacpp as outlines_llamacpp

        if self.config.hf_tokenizer_id:
            tokenizer = LlamaHFTokenizer.from_pretrained(self.config.hf_tokenizer_id)
        else:
            tokenizer = None

        self.model = outlines_llamacpp(
            repo_id=self.config.model_engine_config.model,
            filename=self.config.model_engine_config.filename,
            tokenizer=tokenizer,
            n_ctx=self.config.model_engine_config.n_ctx,
            n_gpu_layers=self.config.model_engine_config.n_gpu_layers,
        )

    def _generate(self, state: GenerationState) -> None:
        generator = self._compile_grammar(state.schema, state.metadata)

        if (
            state.metadata.compile_status.code != CompileStatusCode.OK
            or generator is None
        ):
            return

        try:
            with stopit.ThreadingTimeout(GENERATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    token_iterator = generator.stream(
                        state.input,
                        temperature=self.config.model_engine_config.temperature,
                        max_tokens=self.config.model_engine_config.max_tokens,
                        stop_at="```\n",
                    )

                    tokens_str = []
                    for i, token in enumerate(token_iterator):
                        if i == 0:
                            state.metadata.first_token_arrival_time = time()
                        tokens_str.append(token)

                    state.metadata.decoding_status = DecodingStatus(
                        code=DecodingStatusCode.OK
                    )

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                state.metadata.decoding_status = DecodingStatus(
                    code=DecodingStatusCode.DECODING_TIMEOUT,
                    message="Generation timed out",
                )

        except Exception as e:
            state.metadata.decoding_status = DecodingStatus(
                code=DecodingStatusCode.UNKOWN_ERROR, message=str(e)
            )

            self.model.model.reset()
            return

        output = "".join(tokens_str)
        state.token_usage.output_tokens = self.count_tokens(output)

        state.output = output
        state.generated_tokens = [
            Token(id=self.convert_token_to_id(token), text=token)
            for token in tokens_str
        ]

        self.model.model.reset()
        return

    def _compile_grammar(
        self, schema: Schema, metadata: GenerationMetadata
    ) -> Optional["SequenceGeneratorAdapter"]:
        from outlines.caching import cache_disabled
        from outlines.generate import json as outlines_json

        try:
            with stopit.ThreadingTimeout(COMPILATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    if not self.config.grammar_cache_enabled:
                        with cache_disabled():
                            generator = outlines_json(
                                self.model, schema_object=dumps(schema)
                            )
                    else:
                        generator = outlines_json(
                            self.model, schema_object=dumps(schema)
                        )

                    metadata.grammar_compilation_end_time = time()
                    metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.COMPILE_TIMEOUT,
                    message="Grammar compilation timed out",
                )
                return None

        except BaseException as e:
            metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
            return None

        return generator

    def encode(self, text: str) -> List[int]:
        return self.model.model.tokenizer().encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.model.model.tokenizer().decode(ids)

    @property
    def max_context_length(self) -> int:
        return self.config.model_engine_config.n_ctx

    def close(self):
        self.model.model.close()


register_engine("outlines", OutlinesEngine, OutlinesConfig)
