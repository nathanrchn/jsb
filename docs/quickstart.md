# Quick Start Guide

## Installation

To install the library, follow these steps:

1. Create a conda environment:
    ```bash
    conda create -n "jsb" python=3.12
    conda activate jsb
    ```

2. Install the core dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install engines libraries:
   ```bash
    # Install OpenAI and Gemini
    pip install openai
    pip install google-generativeai

    # Install Guidance
    pip install git+https://github.com/guidance-ai/guidance.git@514a5eb16b9d29ad824d9357732ba66e5e767642
    
    # Install llama-cpp-python with CUDA support
    CMAKE_ARGS="-DGGML_CUDA=on" pip install git+https://github.com/abetlen/llama-cpp-python.git
    
    # Install Outlines and XGrammar
    pip install git+https://github.com/dottxt-ai/outlines.git
    pip install git+https://github.com/mlc-ai/xgrammar.git
   ```

For GPU acceleration, ensure you have the appropriate CUDA drivers installed on your system.

## Using the CLI

You can run the benchmark using the command-line interface:

```bash
python3 -m run --engine <engine> --tasks <tasks> --limit <limit> --save_states
```

### Parameters:
- `engine`: The engine implementation to benchmark
- `tasks`: The tasks to run
- `limit`: Maximum number of samples to run on each task
- `save_states`: Save execution states for later analysis

## Analyzing Results

If you have saved states, you can generate a report:

```bash
python3 -m analyze --states <states_path>
```

## Using the Python API

You can also create a Python script to use the library directly. This approach allows you to create a custom engine and run the benchmark with more flexibility.

```python
from core.bench import bench
from core.engine import Engine

# Initialize your engine with configuration
engine = Engine(config=config)

# Run benchmark
states = bench(engine, tasks, limit=limit, save_states=True)
```

For instructions on creating your custom engine, see the [Custom Engine Tutorial](/docs/custom_engine.md).
