# Core Concepts

This page explains the core concepts and components of JSONSchemaBench.

## JSON Schema

[JSON Schema](https://json-schema.org/) is a vocabulary that allows you to annotate and validate JSON documents. In the context of JSONSchemaBench, JSON Schemas are used to define the structure that language models should generate.

## Benchmark Tasks

JSONSchemaBench includes several benchmark tasks of varying difficulty:

- **simple**: Basic JSON schemas with simple types and structures
- **medium**: More complex schemas with nested objects and arrays
- **hard**: Schemas with complex validation rules and dependencies
- **very_hard**: Highly complex schemas with advanced features and constraints

## Key Components

### Engine

The `Engine` class is an abstract base class that defines the interface for language model engines. It provides methods for:

- Generating text based on a prompt and schema
- Adapting schemas for specific engine requirements
- Tokenization and token counting
- Tracking token usage

Each specific engine (like OpenAI or Guidance) implements this interface.

### Dataset

The `Dataset` class handles loading and managing benchmark datasets. It provides methods for:

- Loading datasets from the Hugging Face hub
- Filtering and mapping schemas
- Iterating through schemas with formatted prompts

### Evaluation

The evaluation system measures how well language models can generate valid JSON according to given schemas. Key metrics include:

- **Declared Coverage**: The percentage of schemas for which the model attempted to generate JSON
- **Empirical Coverage**: The percentage of schemas for which the model generated valid JSON that conforms to the schema
- **Performance Metrics**: Various timing metrics to measure generation speed and efficiency

## Prompt Formatting

JSONSchemaBench uses a prompt formatting function to convert JSON schemas into natural language prompts for the language model. The default prompt is:

```
You need to generate a JSON object that matches the schema below. 
Do not include the schema in the output and DIRECTLY return the JSON object without any additional information. 
The schema is: {schema}
```

You can customize this prompt by providing your own formatting function.

## Token Usage Tracking

JSONSchemaBench tracks token usage for each generation, including:

- Input tokens (prompt)
- Output tokens (generated JSON)
- Total tokens used

This information is useful for estimating costs and comparing efficiency across different models. 