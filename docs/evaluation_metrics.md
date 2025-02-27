# Evaluation Metrics

JSONSchemaBench uses several metrics to evaluate language model performance on JSON schema generation tasks. This page explains these metrics in detail.

## Coverage Metrics

### Declared Coverage

Declared coverage measures the percentage of schemas for which the model attempted to generate JSON. A model "declares" it can handle a schema when it returns a response without errors.

```python
declared_coverage = successful_generations / total_schemas
```

Where:
- `successful_generations` is the number of schemas for which the model returned a response
- `total_schemas` is the total number of schemas in the benchmark

### Empirical Coverage

Empirical coverage measures the percentage of schemas for which the model generated valid JSON that conforms to the schema.

```python
empirical_coverage = valid_generations / successful_generations
```

Where:
- `valid_generations` is the number of schemas for which the model generated valid JSON
- `successful_generations` is the number of schemas for which the model returned a response

## Performance Metrics

### TTFT (Time to First Token)

TTFT measures the time in seconds from when the request is sent to when the first token is generated. This metric is important for assessing the model's latency.

### TPOT (Time Per Output Token)

TPOT measures the average time in milliseconds to generate each output token. This metric is useful for assessing the model's generation speed.

### TGT (Total Generation Time)

TGT measures the total time in seconds for the entire generation process, from sending the request to receiving the complete response.

### GCT (Generation Completion Time)

GCT measures the time in seconds from when the first token is generated to when the last token is generated. This metric is useful for assessing the model's throughput.

## Token Usage Metrics

JSONSchemaBench also tracks token usage for each generation:

- **Input Tokens**: The number of tokens in the prompt
- **Output Tokens**: The number of tokens in the generated response
- **Total Tokens**: The sum of input and output tokens

## Validation Process

The validation process for JSON schema conformance follows these steps:

1. Check if the model returned a response without errors
2. Parse the response as JSON
3. Validate the JSON against the schema using the `jsonschema` library
4. Count the response as valid only if it passes all validation rules

## Interpreting Results

When interpreting benchmark results, consider the following:

- **High Declared Coverage, Low Empirical Coverage**: The model is attempting to generate JSON for most schemas but often produces invalid JSON.
- **Low Declared Coverage, High Empirical Coverage**: The model is selective about which schemas it attempts but produces valid JSON when it does.
- **High TTFT, Low TPOT**: The model has high latency but generates tokens quickly once started.
- **Low TTFT, High TPOT**: The model starts quickly but generates tokens slowly.

## Example Output

```
+-----------------+-------------------+--------------------+----------+-----------+---------+---------+
|       Task      | Declared coverage | Empirical coverage | TTFT (s) | TPOT (ms) | TGT (s) | GCT (s) |
+-----------------+-------------------+--------------------+----------+-----------+---------+---------+
|   Github_easy   |       1.00        |        0.95        |   0.52   |   12.34   |   1.23  |   0.71  |
|  Github_medium  |       0.98        |        0.87        |   0.58   |   15.67   |   1.45  |   0.87  |
+-----------------+-------------------+--------------------+----------+-----------+---------+---------+
```

This table shows that the model performs well on simple schemas but struggles with more complex ones. The performance metrics also show that generation takes longer for more complex schemas. 