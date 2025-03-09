import stopit
from time import time
from json import dumps
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from api.base import Schema
from core.registry import register_engine
from engines.llama_cpp import LlamaCppConfig
from api.engine import Engine, EngineConfig, GenerationResult
from core.types import (
    TokenUsage,
    GenerationMetadata,
    CompileStatus,
    CompileStatusCode,
    DecodingStatus,
    DecodingStatusCode,
)

if TYPE_CHECKING:
    from outlines.generate.api import SequenceGeneratorAdapter

COMPILATION_TIMEOUT = 30
GENERATION_TIMEOUT = 60


@dataclass
class OutlinesConfig(EngineConfig):
    model_engine_config: LlamaCppConfig
    grammar_cache_enabled: bool = False
    hf_tokenizer_id: Optional[str] = None


class OutlinesEngine(Engine[OutlinesConfig]):
    def __init__(self, config: OutlinesConfig):
        super().__init__(config)

        from outlines.models import llamacpp
        from llama_cpp.llama_tokenizer import LlamaHFTokenizer

        if self.config.hf_tokenizer_id:
            tokenizer = LlamaHFTokenizer.from_pretrained(self.config.hf_tokenizer_id)
        else:
            tokenizer = None

        self.model = llamacpp(
            repo_id=self.config.model_engine_config.model,
            filename=self.config.model_engine_config.filename,
            tokenizer=tokenizer,
            n_ctx=self.config.model_engine_config.n_ctx,
            n_gpu_layers=self.config.model_engine_config.n_gpu_layers,
        )

    def _generate(self, prompt: str, schema: Schema) -> GenerationResult:
        metadata = GenerationMetadata()

        generator = self._compile_grammar(schema, metadata)

        if metadata.compile_status.code != CompileStatusCode.OK:
            return GenerationResult(input=prompt, output="", metadata=metadata)

        try:
            with stopit.ThreadingTimeout(GENERATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    token_iterator = generator.stream(
                        prompt,
                        temperature=self.config.model_engine_config.temperature,
                        max_tokens=self.config.model_engine_config.max_tokens,
                    )

                    tokens = []
                    for i, token in enumerate(token_iterator):
                        print(token)
                        if i == 0:
                            metadata.first_token_arrival_time = time()
                        tokens.append(token)

                    output = "".join(tokens)
                    metadata.decoding_status = DecodingStatus(
                        code=DecodingStatusCode.OK
                    )

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

        if isinstance(output, dict):
            output = dumps(output)
        elif output is None:
            output = ""
        elif not isinstance(output, str):
            output = str(output)

        token_usage = TokenUsage(
            input_tokens=self.count_tokens(prompt),
            output_tokens=self.count_tokens(output),
        )

        self.model.model.reset()

        return GenerationResult(
            input=prompt,
            output=output,
            json_schema=schema,
            token_usage=token_usage,
            metadata=metadata,
        )

    def _compile_grammar(
        self, schema: Schema, metadata: GenerationMetadata
    ) -> "SequenceGeneratorAdapter":
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

        except Exception as e:
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
        self.model.close()


register_engine("outlines", OutlinesEngine, OutlinesConfig)
