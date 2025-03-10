import os
import time
import stopit
from json import dumps
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from api.base import Schema
from core.registry import register_engine
from api.engine import Engine, EngineConfig, GenerationResult
from core.custom_types import (
    TokenUsage,
    GenerationMetadata,
    CompileStatus,
    CompileStatusCode,
    DecodingStatus,
    DecodingStatusCode,
    Token,
)

if TYPE_CHECKING:
    from llama_cpp.llama_grammar import LlamaGrammar

COMPILATION_TIMEOUT = 30
GENERATION_TIMEOUT = 60


@dataclass
class LlamaCppConfig(EngineConfig):
    model: str
    filename: str
    n_ctx: int = 4096
    verbose: bool = True
    n_gpu_layers: int = -1
    temperature: float = 0.2
    max_tokens: Optional[int] = None


class LlamaCppEngine(Engine[LlamaCppConfig]):
    def __init__(self, config: LlamaCppConfig):
        super().__init__(config)

        from llama_cpp import Llama

        self.model = Llama.from_pretrained(
            repo_id=self.config.model,
            filename=self.config.filename,
            n_ctx=self.config.n_ctx,
            verbose=self.config.verbose,
            n_gpu_layers=self.config.n_gpu_layers,
        )

    def _generate(self, prompt: str, schema: Schema) -> GenerationResult:
        from llama_cpp.llama_grammar import LlamaGrammar

        metadata = GenerationMetadata()

        grammar = None
        try:
            with stopit.ThreadingTimeout(COMPILATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    grammar = LlamaGrammar.from_json_schema(
                        dumps(schema), verbose=False
                    )
                    metadata.grammar_compilation_end_time = time.time()
                    metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.COMPILE_TIMEOUT,
                    message="Grammar compilation timed out",
                )
                return GenerationResult(input=prompt, output="", metadata=metadata)

            segfault_check = self._check_grammar_safety(grammar)
            if not segfault_check["success"]:
                metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.UNSUPPORTED_SCHEMA,
                    message=f"Failed to add grammar to sampler: {segfault_check}",
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
                    generator = self.model.create_chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                        logprobs=True,
                        grammar=grammar,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )

                    tokens_str = []
                    generated_tokens = []

                    for i, chunk in enumerate(generator):
                        if i == 0:
                            metadata.first_token_arrival_time = time.time()

                        if (
                            len(chunk["choices"]) == 0
                            or chunk["choices"][0]["finish_reason"] is not None
                        ):
                            continue

                        chunk_content = chunk["choices"][0]["delta"].get("content", "")
                        if chunk_content:
                            tokens_str.append(chunk_content)

                            if "logprobs" in chunk["choices"][0]:
                                logprob_info = chunk["choices"][0]["logprobs"]
                                if logprob_info and "token_logprobs" in logprob_info:
                                    for j, token_text in enumerate(
                                        logprob_info["tokens"]
                                    ):
                                        token_id = self.convert_token_to_id(token_text)
                                        logprob = logprob_info["token_logprobs"][j]
                                        generated_tokens.append(
                                            Token(
                                                id=token_id,
                                                text=token_text,
                                                logprob=logprob,
                                            )
                                        )

                    generation = "".join(tokens_str)
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

        input_tokens = len(self.encode(prompt))
        output_tokens = len(self.encode(generation)) if generation else 0

        token_usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

        self.model.reset()

        return GenerationResult(
            input=prompt,
            output=generation,
            json_schema=schema,
            token_usage=token_usage,
            metadata=metadata,
            generated_tokens=generated_tokens,
        )

    def _check_grammar_safety(self, grammar: "LlamaGrammar") -> Dict[str, Any]:
        def child_process():
            import signal
            from llama_cpp._internals import LlamaSampler

            signal.signal(signal.SIGALRM, lambda _, __: os._exit(2))
            signal.alarm(15)
            try:
                LlamaSampler().add_grammar(self.model._model, grammar)
                os._exit(0)
            except Exception:
                os._exit(1)

        id = os.fork()
        if id == 0:
            child_process()
        else:
            _, status = os.waitpid(id, 0)
            if os.WIFEXITED(status):
                exit_code = os.WEXITSTATUS(status)
                return {"success": exit_code == 0, "exit_code": exit_code}
            return {"success": False, "error": "Unknown status"}

    def encode(self, text: str) -> List[int]:
        byte_string = text.encode("utf-8")
        return self.model.tokenize(byte_string)

    def decode(self, ids: List[int]) -> str:
        byte_string = self.model.detokenize(ids)
        return byte_string.decode("utf-8")

    @property
    def max_context_length(self) -> int:
        return self.model.n_ctx()

    def close(self):
        self.model.close()


register_engine("llama_cpp", LlamaCppEngine, LlamaCppConfig)
