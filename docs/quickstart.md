# Quick Start Guide

This guide will help you get started with JSONSchemaBench quickly.

## Basic Usage

Here's a simple example of how to use JSONSchemaBench to evaluate an OpenAI model:

```python
import os
from api.engine import Engine
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

# Run the benchmark on a specific task
# Available tasks: "simple", "medium", "hard", "very_hard"
bench(engine, ["simple"], limit=10)
```

## Command Line Usage

You can also run benchmarks from the command line:

```bash
python bench.py --engine openai --config default --tasks simple,medium --limit 10
```

## Understanding the Output

The benchmark will output a table with the following metrics:

- **Declared coverage**: The percentage of schemas for which the model attempted to generate JSON
- **Empirical coverage**: The percentage of schemas for which the model generated valid JSON
- **TTFT (Time to First Token)**: The median time in seconds to generate the first token
- **TPOT (Time Per Output Token)**: The median time in milliseconds per output token
- **TGT (Total Generation Time)**: The median total time in seconds for generation
- **GCT (Generation Completion Time)**: The median time in seconds from first to last token

Example output:
```
+----------+-------------------+--------------------+----------+-----------+---------+---------+
|   Task   | Declared coverage | Empirical coverage | TTFT (s) | TPOT (ms) | TGT (s) | GCT (s) |
+----------+-------------------+--------------------+----------+-----------+---------+---------+
|  simple  |       1.00        |        0.95        |   0.52   |   12.34   |   1.23  |   0.71  |
+----------+-------------------+--------------------+----------+-----------+---------+---------+
```

## Next Steps

- Learn about the [Core Concepts](./core_concepts.md) of JSONSchemaBench
- Explore the [API Reference](./api_reference.md) for more detailed information
- Check out the [Examples](./examples.md) for more advanced usage scenarios 