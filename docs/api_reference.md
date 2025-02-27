# API Reference

This page provides detailed documentation for the main classes and functions in JSONSchemaBench.

## Core API

### `bench` Function

```python
def bench(
    engine: Engine,
    tasks: List[str],
    limit: Optional[int] = None,
    prompt_fn: FormatPrompt = DEFAULT_FORMAT_PROMPT,
) -> None
```

Runs benchmarks on the specified engine for the given tasks.

**Parameters:**
- `engine`: An instance of an `Engine` class
- `tasks`: A list of task names to benchmark (e.g., ["simple", "medium"])
- `limit`: Optional limit on the number of schemas to test per task
- `prompt_fn`: A function that formats JSON schemas into prompts

**Returns:**
- None (prints results to console)

## Engine API

### `Engine` Class

```python
class Engine(ABC, Generic[T]):
    def __init__(self, config: T):
        ...
```

Abstract base class for language model engines.

**Methods:**
- `generate(prompt: str, schema: Schema) -> GenerationResult`: Generates text based on a prompt and schema
- `_generate(prompt: str, schema: Schema) -> GenerationResult`: Abstract method implemented by subclasses
- `adapt_schema(schema: Schema) -> Schema`: Adapts a schema for the specific engine
- `encode(text: str) -> Optional[List[int]]`: Encodes text to token IDs
- `decode(ids: List[int]) -> Optional[str]`: Decodes token IDs to text
- `count_tokens(text: str) -> int`: Counts the number of tokens in text

**Properties:**
- `max_context_length`: Returns the maximum context length for the model

### `GenerationResult` Class

```python
@dataclass
class GenerationResult:
    input: str
    output: str
    label: Optional[str] = None
    json_schema: Optional[Schema] = None
    ...
```

Represents the result of a generation request.

## Dataset API

### `Dataset` Class

```python
class Dataset:
    def __init__(self, config: DatasetConfig):
        ...
```

Handles loading and managing benchmark datasets.

**Methods:**
- `__len__() -> int`: Returns the number of schemas in the dataset
- `__getitem__(idx: int) -> Schema`: Returns the schema at the given index
- `filter(filter_fn: Callable[[Schema], bool]) -> None`: Filters the dataset
- `map(map_fn: Callable[[Schema], Schema]) -> None`: Maps the schemas in the dataset
- `shuffle() -> None`: Shuffles the dataset
- `iter(prompt_fn: FormatPrompt) -> Iterator[Tuple[str, Schema]]`: Iterates through the dataset

## Evaluation API

### `evaluate` Function

```python
def evaluate(
    results: List[GenerationResult],
) -> Tuple[float, float, PerfMetrics]
```

Evaluates generation results.

**Parameters:**
- `results`: A list of `GenerationResult` objects

**Returns:**
- A tuple containing:
  - Declared coverage (float)
  - Empirical coverage (float)
  - Performance metrics (PerfMetrics)

### `print_scores` Function

```python
def print_scores(
    declared_coverage: List[float],
    empirical_coverage: List[float],
    perf_metrics: List[PerfMetrics],
    tasks: List[str],
) -> None
```

Prints benchmark scores in a formatted table.

## Utility Types

### `Schema` Type

```python
Schema = Dict[str, Any]
```

Represents a JSON schema.

### `FormatPrompt` Type

```python
FormatPrompt = Callable[[Schema], str]
```

A function that formats a JSON schema into a prompt.

### `DEFAULT_FORMAT_PROMPT` Function

```python
DEFAULT_FORMAT_PROMPT: FormatPrompt = (
    lambda schema: f" You need to generate a JSON object that matches the schema below. Do not include the schema in the output and DIRECTLY return the JSON object without any additional information. The schema is: {dumps(schema)}"
)
```

The default prompt formatting function.
