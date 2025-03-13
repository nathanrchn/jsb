import os, sys

from core.bench import bench
from engines.gemini import GeminiEngine
from engines.openai import OpenAIEngine, OpenAIConfig
from engines.guidance import GuidanceEngine, GuidanceConfig
from engines.outlines import OutlinesEngine, OutlinesConfig
from engines.xgrammar import XGrammarEngine, XGrammarConfig
from engines.llama_cpp import LlamaCppEngine, LlamaCppConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # openai
# openai_engine = OpenAIEngine(OpenAIConfig(model="gpt-4o-mini"))
# bench(openai_engine, ["Github_easy"], limit=10)

# # gemini
# gemini_engine = GeminiEngine(OpenAIConfig(model="models/gemini-2.0-flash-lite"))
# bench(gemini_engine, ["Github_easy"], limit=10)

# guidance
guidance_engine = GuidanceEngine(
    GuidanceConfig(
        model_engine_config=LlamaCppConfig(
            model="bartowski/google_gemma-3-1b-it-GGUF", filename="*Q8_0.gguf"
        )
    )
)
bench(guidance_engine, ["Github_easy"], close_engine=True)

# llama_cpp
llama_cpp_engine = LlamaCppEngine(
    LlamaCppConfig(model="bartowski/google_gemma-3-1b-it-GGUF", filename="*Q8_0.gguf")
)
bench(llama_cpp_engine, ["Github_easy"], close_engine=True)

# outlines
outlines_engine = OutlinesEngine(
    OutlinesConfig(
        model_engine_config=LlamaCppConfig(
            model="bartowski/google_gemma-3-1b-it-GGUF", filename="*Q8_0.gguf"
        ),
        hf_tokenizer_id="google/gemma-3-1b-it",
    )
)
bench(outlines_engine, ["Github_easy"], close_engine=True)

# xgrammar
xgrammar_engine = XGrammarEngine(XGrammarConfig(model="google/gemma-3-1b-it"))
bench(xgrammar_engine, ["Github_easy"], close_engine=True)
