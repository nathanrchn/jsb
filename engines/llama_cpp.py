import os
import time
import stopit
from json import dumps
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from core.registry import register_engine
from core.engine import Engine, EngineConfig
from core.utils import COMPILATION_TIMEOUT, GENERATION_TIMEOUT
from core.types import (
    Token,
    CompileStatus,
    DecodingStatus,
    GenerationState,
    CompileStatusCode,
    DecodingStatusCode,
)

if TYPE_CHECKING:
    from llama_cpp.llama_grammar import LlamaGrammar


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

    def _generate(self, state: GenerationState) -> None:
        from llama_cpp.llama_grammar import LlamaGrammar

        grammar = None
        try:
            with stopit.ThreadingTimeout(COMPILATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    grammar = LlamaGrammar.from_json_schema(
                        dumps(state.schema), verbose=False
                    )
                    state.metadata.grammar_compilation_end_time = time.time()
                    state.metadata.compile_status = CompileStatus(
                        code=CompileStatusCode.OK
                    )

            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                state.metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.COMPILE_TIMEOUT,
                    message="Grammar compilation timed out",
                )
                return

            segfault_check = self._check_grammar_safety(grammar)
            if not segfault_check["success"]:
                state.metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.UNSUPPORTED_SCHEMA,
                    message=f"Failed to add grammar to sampler: {segfault_check}",
                )
                return

        except Exception as e:
            state.metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
            return

        try:
            with stopit.ThreadingTimeout(GENERATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    generator = self.model.create_chat_completion(
                        messages=[{"role": "user", "content": state.input}],
                        stream=True,
                        grammar=grammar,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )

                    tokens_str = []
                    for i, chunk in enumerate(generator):
                        if i == 0:
                            state.metadata.first_token_arrival_time = time.time()

                        if (
                            len(chunk["choices"]) == 0
                            or chunk["choices"][0]["finish_reason"] is not None
                        ):
                            continue

                        chunk_content = chunk["choices"][0]["delta"].get("content", "")
                        if chunk_content:
                            tokens_str.append(chunk_content)

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

            self.model.reset()
            return

        generation = "".join(tokens_str)

        state.output = generation
        state.token_usage.output_tokens = self.count_tokens(generation)
        state.generated_tokens = [
            Token(id=self.convert_token_to_id(token), text=token)
            for token in tokens_str
        ]

        self.model.reset()
        return

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


register_engine("llama_cpp", LlamaCppEngine, LlamaCppConfig)
