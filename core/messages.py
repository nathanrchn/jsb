from json import dumps
from typing import Callable, Dict, Any, List, Tuple

from core.types import Schema

Message = Dict[str, Any]
MessagesFormatter = Callable[[str, Schema], List[Message]]


def default_messages_formatter(_: str, schema: Schema) -> List[Message]:
    return [
        {
            "role": "system",
            "content": "You need to generate a JSON object that matches the schema below.",
        },
        {"role": "user", "content": dumps(schema)},
    ]


def few_shots_messages_formatter(task: str, schema: Schema) -> List[Message]:
    examples = EXAMPLES_FOR_TASK[task]

    messages = [
        {
            "role": "system",
            "content": "You need to generate a JSON object that matches the schema below.",
        }
    ]

    for input, output in examples:
        messages.append({"role": "user", "content": input})
        messages.append({"role": "assistant", "content": output})

    messages.append({"role": "user", "content": dumps(schema)})
    return messages


EXAMPLES_FOR_TASK: Dict[str, List[Tuple[str, str]]] = {
    "Snowplow": [
        (
            '{\n    "additionalProperties": false,\n    "description": "Schema for a JSON Paths file for loading Redshift from JSON or Avro, http://docs.aws.amazon.com/redshift/latest/dg/copy-parameters-data-format.html#copy-json-jsonpaths",\n    "properties": {\n        "jsonpaths": {\n            "items": {\n                "type": "string"\n            },\n            "minItems": 1,\n            "type": "array"\n        }\n    },\n    "required": [\n        "jsonpaths"\n    ],\n    "self": {\n        "format": "jsonschema",\n        "name": "jsonpaths_file",\n        "vendor": "com.amazon.aws.redshift",\n        "version": "1-0-0"\n    },\n    "type": "object"\n}',
            '{"jsonpaths": ["$.user.id", "$.user.name", "$.user.address.street"]}',
        ),
        (
            '{\n    "additionalProperties": false,\n    "description": "Schema for a Google Analytics enhanced e-commerce product impression custom metric entity",\n    "properties": {\n        "customMetricIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "listIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "productIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "value": {\n            "type": [\n                "integer",\n                "null"\n            ]\n        }\n    },\n    "self": {\n        "format": "jsonschema",\n        "name": "product_impression_custom_metric",\n        "vendor": "com.google.analytics.measurement-protocol",\n        "version": "1-0-0"\n    },\n    "type": "object"\n}',
            '{"customMetricIndex": 120, "listIndex": 45, "productIndex": 10, "value": 300}',
        ),
    ]
}

DEFAULT_MESSAGES_FORMATTER: MessagesFormatter = default_messages_formatter
FEW_SHOTS_MESSAGES_FORMATTER: MessagesFormatter = few_shots_messages_formatter
