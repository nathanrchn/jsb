# Examples

This page provides examples of how to use JSONSchemaBench for various use cases.

## Basic Benchmarking

This example shows how to benchmark an OpenAI model on a single task:

```python
import os
from utils import ENGINE_TO_CLASS
from engines.openai import OpenAIConfig
from bench import bench

# Set up the OpenAI engine
engine = ENGINE_TO_CLASS["openai"](
    OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
)

# Run the benchmark on the "simple" task with a limit of 10 schemas
bench(engine, ["simple"], limit=10)
```

## Benchmarking Multiple Tasks

This example shows how to benchmark a model on multiple tasks:

```python
import os
from utils import ENGINE_TO_CLASS
from engines.openai import OpenAIConfig
from bench import bench

# Set up the OpenAI engine
engine = ENGINE_TO_CLASS["openai"](
    OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
)

# Run the benchmark on multiple tasks
bench(engine, ["simple", "medium", "hard", "very_hard"], limit=10)
```

## Custom Prompt Format

This example shows how to use a custom prompt format:

```python
import os
from json import dumps
from typing import Callable
from utils import ENGINE_TO_CLASS
from engines.openai import OpenAIConfig
from bench import bench
from api.base import Schema, FormatPrompt

# Define a custom prompt format
def custom_prompt_format(schema: Schema) -> str:
    return f"""
    I need you to generate a valid JSON object that follows this schema:
    
    {dumps(schema, indent=2)}
    
    Please provide ONLY the JSON object without any explanations or markdown formatting.
    """

# Set up the OpenAI engine
engine = ENGINE_TO_CLASS["openai"](
    OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
)

# Run the benchmark with the custom prompt format
bench(engine, ["simple"], limit=10, prompt_fn=custom_prompt_format)
```

## Comparing Multiple Models

This example shows how to compare multiple models:

```python
import os
from utils import ENGINE_TO_CLASS
from engines.openai import OpenAIConfig
from bench import bench

# Define the models to compare
models = [
    "gpt-4o-mini",
    "gpt-4o"
]

# Run benchmarks for each model
for model in models:
    print(f"\nBenchmarking {model}...")
    engine = ENGINE_TO_CLASS["openai"](
        OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model
        )
    )
    bench(engine, ["simple", "medium"], limit=10)
```

## Using the Guidance Engine

This example shows how to use the Guidance engine:

```python
from utils import ENGINE_TO_CLASS
from engines.guidance import GuidanceConfig
from bench import bench

# Set up the Guidance engine
engine = ENGINE_TO_CLASS["guidance"](
    GuidanceConfig(
        model="path/to/your/model",
        temperature=0.0,
        max_tokens=1024,
        top_p=0.9
    )
)

# Run the benchmark
bench(engine, ["simple"], limit=5)
```

## Working with Datasets

This example shows how to work with datasets directly:

```python
from api.dataset import Dataset, DatasetConfig
from api.base import DEFAULT_FORMAT_PROMPT

# Load a dataset
dataset = Dataset(DatasetConfig("simple", limit=5))

# Print the number of schemas
print(f"Number of schemas: {len(dataset)}")

# Print the first schema
print(f"First schema: {dataset[0]}")

# Iterate through the dataset
for prompt, schema in dataset.iter(DEFAULT_FORMAT_PROMPT):
    print(f"Prompt: {prompt[:100]}...")
    print(f"Schema: {schema}")
    print("---")
```

## Custom Schema Filtering

This example shows how to filter schemas based on custom criteria:

```python
from api.dataset import Dataset, DatasetConfig

# Load a dataset
dataset = Dataset(DatasetConfig("medium"))

# Define a filter function
def filter_by_type(schema):
    # Only keep schemas with string properties
    return "type" in schema and schema["type"] == "object" and "properties" in schema and any(
        "type" in prop and prop["type"] == "string" 
        for prop in schema["properties"].values()
    )

# Apply the filter
dataset.filter(filter_by_type)

# Print the number of schemas after filtering
print(f"Number of schemas after filtering: {len(dataset)}")
```

## Command Line Usage

This example shows how to use the command line interface:

```bash
# Basic usage
python bench.py --engine openai --config default --tasks simple --limit 10

# Multiple tasks
python bench.py --engine openai --config default --tasks simple,medium,hard --limit 10

# Using a specific configuration file
python bench.py --engine openai --config my_config.json --tasks simple
``` 