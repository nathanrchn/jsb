import json
import time
import stopit
from dataclasses import dataclass
from typing import List, Optional

import outlines
from outlines.caching import cache_disabled
from outlines.generate.api import SequenceGeneratorAdapter

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
class OutlinesConfig(EngineConfig):
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    n_ctx: int = 4096
    grammar_cache_enabled: bool = False
    hf_tokenizer_id: Optional[str] = None


class OutlinesEngine(Engine[OutlinesConfig]):
    def __init__(self, config: OutlinesConfig):
        super().__init__(config)

        # Initialize LlamaCpp model
        import llama_cpp

        # Use HF tokenizer if specified
        if self.config.hf_tokenizer_id:
            tokenizer = llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
                self.config.hf_tokenizer_id
            )
        else:
            tokenizer = None

        # Initialize the model
        self.model = outlines.models.llamacpp(
            repo_id=self.config.model,
            tokenizer=tokenizer,
            n_ctx=self.config.n_ctx,
        )

    def _generate(self, prompt: str, schema: Schema) -> GenerationResult:
        metadata = GenerationMetadata()

        # Determine generation mode based on schema
        if schema:
            # JSON schema mode
            generator = self._compile_grammar(schema, metadata)

            # If compilation failed, return early
            if metadata.compile_status.code != CompileStatusCode.OK:
                return GenerationResult(input=prompt, output="", metadata=metadata)
        else:
            # Free text mode
            generator = outlines.generate.text(self.model)
            metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)
            metadata.grammar_compilation_end_time = time.time()

        # Generate text with timeout
        try:
            with stopit.ThreadingTimeout(GENERATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    # Stream tokens to capture first token time
                    token_iterator = generator.stream(
                        prompt,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        top_p=self.config.top_p,
                    )

                    tokens = []
                    for i, token in enumerate(token_iterator):
                        if i == 0:
                            metadata.first_token_arrival_time = time.time()
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

        # Convert dict output to JSON string if needed
        if isinstance(output, dict):
            output = json.dumps(output)
        elif output is None:
            output = ""
        elif not isinstance(output, str):
            output = str(output)

        # Calculate token usage
        token_usage = TokenUsage(
            input_tokens=self._count_tokens(prompt),
            output_tokens=self._count_tokens(output),
        )

        # Reset model state
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
    ) -> SequenceGeneratorAdapter:
        """Compile JSON schema into an Outlines grammar."""
        try:
            with stopit.ThreadingTimeout(COMPILATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    # Disable caching if not enabled
                    if not self.config.grammar_cache_enabled:
                        with cache_disabled():
                            generator = outlines.generate.json(
                                self.model, schema_object=json.dumps(schema)
                            )
                    else:
                        generator = outlines.generate.json(
                            self.model, schema_object=json.dumps(schema)
                        )

                    metadata.grammar_compilation_end_time = time.time()
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

    def _count_tokens(self, text: str) -> int:
        """Count tokens in a string."""
        if not isinstance(text, str):
            text = str(text)
        return len(self.model.model.tokenizer().encode(text))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.model.model.tokenizer().encode(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.model.model.tokenizer().decode(ids)

    @property
    def max_context_length(self) -> int:
        """Return the maximum context length."""
        return self.config.n_ctx

    def adapt_schema(self, schema: Schema) -> Schema:
        """Adapt schema if needed before processing."""
        return schema
