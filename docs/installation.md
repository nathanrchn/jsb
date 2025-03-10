# Installation

This guide will help you install JSONSchemaBench and its dependencies.

## Requirements

- Python 3.8 or higher
- pip (Python package installer)

## Basic Installation

You can install JSONSchemaBench directly from GitHub:

```bash
git clone https://github.com/guidance-ai/jsonschemabench.git
cd jsonschemabench
pip install -e .
```

## Dependencies

JSONSchemaBench requires the following main dependencies:

- `datasets`: For loading and managing benchmark datasets
- `jsonschema`: For JSON schema validation
- `tqdm`: For progress bars
- `prettytable`: For formatted output display

## Engine-specific Dependencies

Depending on which engines you want to use, you'll need to install additional dependencies:

### OpenAI Engine

```bash
pip install openai tiktoken
```

### Guidance Engine

```bash
pip install guidance
```

### Outlines Engine

```bash
pip install outlines
```

### Llama.cpp Engine

```bash
pip install llama-cpp-python # this doesn't include CUDA support
```

For CUDA and other accelerators, you'll need to install the appropriate version of `llama-cpp-python`, check the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)


## Environment Setup

For the OpenAI engine, you'll need to set up your API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Or you can set it in your Python code:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

## Verifying Installation

To verify that JSONSchemaBench is installed correctly, you can run a simple benchmark:

```python
import os
from api.engine import Engine
from utils import ENGINE_TO_CLASS
from engines.openai import OpenAIConfig

# Set up the engine
engine = ENGINE_TO_CLASS["openai"](
    OpenAIConfig(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
)

# Run a simple test
from bench import bench
bench(engine, ["simple"], limit=1)
```

If the installation is successful, you should see benchmark results printed to the console. 