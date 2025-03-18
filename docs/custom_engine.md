# Custom Engine Tutorial

This guide explains how to implement your own engine for the benchmark system.

## Overview

An engine is responsible for generating responses based on a schema. To create a custom engine, you need to:

1. Create a configuration class that extends `EngineConfig`
2. Implement the `Engine` abstract class
3. Implement the required abstract methods

## Step 1: Create a Configuration Class

First, create a configuration class that extends `EngineConfig`:

```python
from core.engine import EngineConfig

class MyEngineConfig(EngineConfig):
    model_name: str
    temperature: float = 0.7
```

## Step 2: Implement the Engine Class

Next, implement the `Engine` abstract class:

```python
from core.engine import Engine
from core.types import GenerationState, Schema

class MyEngine(Engine[MyEngineConfig]):
    def __init__(self, config: MyEngineConfig):
        super().__init__(config)
        
        # ...
    
    def _generate(self, state: GenerationState) -> None:
        """Generate content based on the prompt and schema"""
        # Implement the generation logic
        prompt = state.input
        response = self.model.generate(
            prompt, 
            temperature=self.config.temperature
        )
        
        # Update the state with the response
        state.output = response
        state.token_usage.output_tokens = self.count_tokens(response)
    
    @property
    def max_context_length(self) -> int:
        """Return the maximum context length for your model"""
        return 4096  # Example value
    
    def encode(self, text: str) -> list[int]:
        """Optional: Implement token encoding"""
        return self.model.tokenizer.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        """Optional: Implement token decoding"""
        return self.model.tokenizer.decode(ids)
```

## Step 3: Use Your Custom Engine

Once you've implemented your engine, you can use it in your benchmarking:

```python
from core.bench import bench

# Create your engine configuration
config = MyEngineConfig(model_name="my-model", temperature=0.7)

# Initialize your engine
engine = MyEngine(config)

# Run benchmark
tasks = ["task1", "task2", "task3"]
states = bench(engine, tasks, limit=10, save_states=True)
```

## Required Abstract Methods

Your engine must implement these abstract methods:

1. `_generate(state: GenerationState) -> None`: Core generation logic
2. `max_context_length() -> int`: Returns the maximum context length

## Optional Methods

You can optionally override these methods for better functionality:

- `adapt_schema(schema: Schema) -> Schema`: Modify the schema for your engine
- `encode(text: str) -> List[int]`: Convert text to tokens
- `decode(ids: List[int]) -> str`: Convert tokens to text
- `count_tokens(text: str) -> int`: Count tokens in text
- `close() -> None`: Cleanup resources
