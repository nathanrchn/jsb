# Engines

JSONSchemaBench supports multiple language model engines. This page documents the available engines and their configurations.

## OpenAI Engine

The OpenAI engine allows you to benchmark OpenAI models like GPT-4, GPT-4o, and GPT-3.5.

### Configuration

```python
@dataclass
class OpenAIConfig(EngineConfig):
    api_key: str
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None
```

**Parameters:**
- `api_key`: Your OpenAI API key
- `model`: The model to use (e.g., "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo")
- `temperature`: Controls randomness (default: None, uses OpenAI's default)
- `max_tokens`: Maximum number of tokens to generate (default: None, uses OpenAI's default)
- `base_url`: Optional base URL for the API (default: None, uses OpenAI's default)

### Example Usage

```python
import os
from utils import ENGINE_TO_CLASS
from engines.openai import OpenAIConfig

engine = ENGINE_TO_CLASS["openai"](
    OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        temperature=0.0
    )
)
```

### Supported Models

The OpenAI engine supports various models with different context lengths:

- `gpt-4o`: 128,000 tokens
- `gpt-4o-mini`: 128,000 tokens
- `gpt-4o-2024-05-13`: 128,000 tokens
- `gpt-4o-2024-08-06`: 128,000 tokens

## Guidance Engine

The Guidance engine allows you to benchmark models using the [Guidance](https://github.com/guidance-ai/guidance) library.

### Configuration

```python
@dataclass
class GuidanceConfig(EngineConfig):
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    model_engine: Literal["llamacpp"] = "llamacpp"
```

**Parameters:**
- `model`: The model to use
- `temperature`: Controls randomness
- `max_tokens`: Maximum number of tokens to generate
- `top_p`: Controls diversity via nucleus sampling
- `model_engine`: The underlying model engine (currently only "llamacpp" is supported)

### Example Usage

```python
from utils import ENGINE_TO_CLASS
from engines.guidance import GuidanceConfig

engine = ENGINE_TO_CLASS["guidance"](
    GuidanceConfig(
        model="path/to/your/model",
        temperature=0.0,
        max_tokens=1024,
        top_p=0.9
    )
)
```

## Adding Custom Engines

You can add custom engines by implementing the `Engine` abstract base class:

1. Create a new engine class that inherits from `Engine`
2. Implement the required abstract methods:
   - `_generate(prompt: str, schema: Schema) -> GenerationResult`
   - `max_context_length` property
   - `adapt_schema(schema: Schema) -> Schema`
3. Optionally implement tokenization methods:
   - `encode(text: str) -> Optional[List[int]]`
   - `decode(ids: List[int]) -> Optional[str]`

Example skeleton for a custom engine:

```python
from dataclasses import dataclass
from typing import List, Optional

from api.engine import Engine, EngineConfig, GenerationResult
from api.base import Schema

@dataclass
class MyCustomEngineConfig(EngineConfig):
    # Define your configuration parameters here
    model: str
    temperature: float

class MyCustomEngine(Engine[MyCustomEngineConfig]):
    def __init__(self, config: MyCustomEngineConfig):
        super().__init__(config)
        # Initialize your model here
        
    def _generate(self, prompt: str, schema: Schema) -> GenerationResult:
        # Implement generation logic here
        pass
        
    @property
    def max_context_length(self) -> int:
        # Return the maximum context length
        return 4096
        
    def adapt_schema(self, schema: Schema) -> Schema:
        # Adapt the schema if needed
        return schema
        
    def encode(self, text: str) -> Optional[List[int]]:
        # Implement tokenization if available
        pass
        
    def decode(self, ids: List[int]) -> Optional[str]:
        # Implement detokenization if available
        pass
```

Then register your engine in the `ENGINE_TO_CLASS` dictionary in `utils.py`:

```python
ENGINE_TO_CLASS: Dict[str, Type[Engine]] = {
    "openai": OpenAIEngine,
    "guidance": GuidanceEngine,
    "my_custom_engine": MyCustomEngine,
}
``` 