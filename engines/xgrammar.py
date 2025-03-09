import time
import torch
import stopit
import xgrammar as xgr
from json import dumps
from typing import List, Optional
from dataclasses import dataclass

from transformers import LogitsProcessor
from transformers.generation import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from api.base import Schema
from api.engine import Engine, EngineConfig, GenerationResult
from core.types import (
    TokenUsage,
    GenerationMetadata,
    CompileStatus,
    CompileStatusCode,
    DecodingStatus,
    DecodingStatusCode,
)
from core.registry import register_engine

COMPILATION_TIMEOUT = 30
GENERATION_TIMEOUT = 60


class TimingLogitsProcessor(LogitsProcessor):
    """Logits processor that records timestamps for token generation."""

    def __init__(self):
        super().__init__()
        self.timestamps = []

    def __call__(self, _, scores):
        self.timestamps.append(time.time())
        return scores


@dataclass
class XGrammarConfig(EngineConfig):
    model: str
    temperature: float = 0
    max_tokens: Optional[int] = None
    grammar_cache_enabled: bool = False


class XGrammarEngine(Engine[XGrammarConfig]):
    def __init__(self, config: XGrammarConfig):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model, torch_dtype=torch.bfloat16, device_map="auto"
        )

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.model.config.vocab_size
        )
        self.grammar_compiler = xgr.GrammarCompiler(
            tokenizer_info, cache_enabled=self.config.grammar_cache_enabled
        )

    def _generate(self, prompt: str, schema: Schema) -> GenerationResult:
        metadata = GenerationMetadata()
        timing_processor = TimingLogitsProcessor()
        logits_processors = [timing_processor]

        try:
            with stopit.ThreadingTimeout(COMPILATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    json_schema_str = dumps(schema)
                    compiled_grammar = self.grammar_compiler.compile_json_schema(
                        json_schema_str
                    )
                    metadata.grammar_compilation_end_time = time.time()
                    metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)
                    logits_processors.append(
                        xgr.contrib.hf.LogitsProcessor(compiled_grammar)
                    )

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.COMPILE_TIMEOUT,
                    message="Grammar compilation timed out",
                )
                return GenerationResult(input=prompt, output="", metadata=metadata)

        except Exception as e:
            metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
            return GenerationResult(input=prompt, output="", metadata=metadata)

        model_input = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
        ).to("cuda")

        input_length = model_input["input_ids"].shape[1]

        try:
            with stopit.ThreadingTimeout(GENERATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    model_output = self.model.generate(
                        model_input["input_ids"],
                        generation_config=GenerationConfig(
                            max_new_tokens=self.config.max_tokens,
                            temperature=self.config.temperature,
                            do_sample=self.config.temperature > 0,
                        ),
                        attention_mask=model_input["attention_mask"],
                        tokenizer=self.tokenizer,
                        logits_processor=logits_processors,
                    )
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

        generated_sequences = model_output.sequences[:, input_length:]
        generated_texts = self.tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )

        output_text = generated_texts[0] if generated_texts else ""

        generated_tokens = []
        if output_text:
            token_ids = self.encode(output_text)
            for token_id in token_ids:
                token_text = self.tokenizer.decode([token_id])
                generated_tokens.append({"id": token_id, "text": token_text})

        if timing_processor.timestamps:
            metadata.first_token_arrival_time = timing_processor.timestamps[0]

        token_usage = TokenUsage(
            input_tokens=input_length, output_tokens=len(generated_tokens)
        )

        return GenerationResult(
            input=prompt,
            output=output_text,
            json_schema=schema,
            token_usage=token_usage,
            metadata=metadata,
            generated_tokens=generated_tokens,
        )

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def max_context_length(self) -> int:
        return self.tokenizer.model_max_length


register_engine("xgrammar", XGrammarEngine, XGrammarConfig)
