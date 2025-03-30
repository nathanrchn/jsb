from core.bench import bench
from engines.gemini import GeminiEngine
from engines.openai import OpenAIEngine, OpenAIConfig
from engines.guidance import GuidanceEngine, GuidanceConfig
from engines.outlines import OutlinesEngine, OutlinesConfig
from engines.xgrammar import XGrammarEngine, XGrammarConfig
from engines.llama_cpp import LlamaCppEngine, LlamaCppConfig
from engines.huggingface import HuggingFaceEngine, HuggingFaceConfig

# guidance
guidance_engine = GuidanceEngine(
    GuidanceConfig(
        model_engine_config=HuggingFaceConfig(
            model="meta-llama/Llama-3.1-8B-Instruct"
        )
    )
)

bench(
    guidance_engine,
    ["Glaiveai2K", "Github_easy", "Snowplow", "Github_medium", "Kubernetes", "WashingtonPost", "Github_hard"],
    limit=38,
    save_outputs=True,
)
