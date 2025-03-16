from core.bench import bench
from engines.gemini import GeminiEngine
from engines.openai import OpenAIEngine, OpenAIConfig
from engines.guidance import GuidanceEngine, GuidanceConfig
from engines.outlines import OutlinesEngine, OutlinesConfig
from engines.xgrammar import XGrammarEngine, XGrammarConfig
from engines.llama_cpp import LlamaCppEngine, LlamaCppConfig

# openai
openai_engine = OpenAIEngine(OpenAIConfig(model="gpt-4o-mini"))
bench(
    openai_engine,
    ["Glaiveai2K", "Github_easy", "Snowplow", "Github_medium"],
    limit=100,
    save_results=True,
)

# gemini
gemini_engine = GeminiEngine(OpenAIConfig(model="models/gemini-2.0-flash-lite"))
bench(
    gemini_engine,
    ["Glaiveai2K", "Github_easy", "Snowplow", "Github_medium"],
    limit=100,
    save_results=True,
)

# guidance
guidance_engine = GuidanceEngine(
    GuidanceConfig(
        model_engine_config=LlamaCppConfig(
            model="bartowski/Llama-3.2-1B-Instruct-GGUF", filename="*f16.gguf"
        )
    )
)
bench(
    guidance_engine,
    ["Glaiveai2K", "Github_easy", "Snowplow", "Github_medium"],
    limit=100,
    close_engine=True,
    save_results=True,
)

# llama_cpp
llama_cpp_engine = LlamaCppEngine(
    LlamaCppConfig(model="bartowski/Llama-3.2-1B-Instruct-GGUF", filename="*f16.gguf")
)
bench(
    llama_cpp_engine,
    ["Glaiveai2K", "Github_easy", "Snowplow", "Github_medium"],
    limit=100,
    close_engine=True,
    save_results=True,
)

# outlines
outlines_engine = OutlinesEngine(
    OutlinesConfig(
        model_engine_config=LlamaCppConfig(
            model="bartowski/Llama-3.2-1B-Instruct-GGUF", filename="*f16.gguf"
        ),
        hf_tokenizer_id="meta-llama/Llama-3.2-1B-Instruct",
    )
)
bench(
    outlines_engine,
    ["Glaiveai2K", "Github_easy", "Snowplow", "Github_medium"],
    limit=100,
    close_engine=True,
    save_results=True,
)

# xgrammar
xgrammar_engine = XGrammarEngine(
    XGrammarConfig(model="meta-llama/Llama-3.2-1B-Instruct")
)
bench(
    xgrammar_engine,
    ["Glaiveai2K", "Github_easy", "Snowplow", "Github_medium"],
    limit=100,
    close_engine=True,
    save_results=True,
)
